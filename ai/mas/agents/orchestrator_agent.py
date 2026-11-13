#!/usr/bin/env python3
"""
Orchestrator Agent – главный диспетчер.

Особенности:
  - При первом сообщении (нет language_agent_run) вызывает LanguageDetectionAgent.
  - Определение интентов с помощью GPT (detect_intents_gpt).
  - Учитывает приоритет и состояние профиля: 
    * если пользователь не прошёл Triage, то при получении ProgramInfo – сначала вызывается Triage.
    * если пользователь не авторизован, а обнаружен интент LeadQualification – вместо него вызывается AuthorizationHelp.
  - Объединяет ответы агентов в одно дружелюбное сообщение (combine_responses_friendly).
  - Генерирует «живые» ответы за счёт расширенных инструкций в под‑агентах.
  - При обнаружении handoff‑сигнала orchestrator не передаёт его пользователю, а переключается на целевого агента.
"""

import logging
import json
import re
from typing import List, Optional, Dict, Any
import traceback

from openai import AsyncOpenAI

from ai.mas.agents.base_agent import BaseAgent
from ai.mas.utils.handoffs import (
    transfer_to_language_detection_agent,
    transfer_to_common_questions_agent,
    transfer_to_triage_agent,
    transfer_to_program_decision_agent,
    transfer_to_authorization_agent,
    transfer_to_lead_qualification_agent,
)
from ai.mas.utils.db import store_user_data_in_db, get_user_profile
from ai.mas.agents.statistics_agent import StatisticsAgent
from ai.mas.agents.summary_agent import SummaryAgent
from ai.mas.agents.objection_detector_agent import ObjectionDetectorAgent
from ai.mas.main import ChatRequest, ChatResponse, UserProfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ConversationContext:
    """
    Управляет контекстом диалога, включая:
    - историю сообщений
    - состояние диалога
    - метаданные агентов
    - статистику взаимодействий
    """
    def __init__(self, user_profile_id: str):
        self.user_profile_id: str = user_profile_id
        self.history: List[Dict[str, Any]] = []
        self.last_agent: str = ""
        self.last_agent_response: str = ""
        self.pending_question: str = ""
        self.pending_question_agent: str = ""
        self.current_flow: str = "pre_registration"  # pre_registration или post_registration
        self.stage_data: Dict[str, Any] = {
            "language_detected": False,
            "triage_completed": False,
            "program_selected": False,
            "authorized": False,
            "lead_qualified": False
        }
        self.statistics: Dict[str, int] = {
            "total_messages": 0,
            "questions_asked": 0,
            "handoffs": 0,
            "clarifications": 0
        }

    def update(self, agent_name: str, user_message: str, agent_response: str, intent: str) -> None:
        """
        Обновляет контекст диалога новым взаимодействием.
        """
        # Обновляем статистику
        self.statistics["total_messages"] += 1
        if "?" in user_message:
            self.statistics["questions_asked"] += 1
        if "[handoff]" in agent_response.lower():
            self.statistics["handoffs"] += 1
        if any(phrase in agent_response.lower() for phrase in ["уточните", "поясните", "не могли бы вы"]):
            self.statistics["clarifications"] += 1

        # Сохраняем сообщение
        entry = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "user_message": user_message,
            "agent_response": agent_response,
            "agent": agent_name,
            "intent": intent,
            "flow": self.current_flow
        }
        self.history.append(entry)
        self.last_agent = agent_name
        self.last_agent_response = agent_response

        # Обновляем состояние диалога
        if agent_name == "language_detector":
            self.stage_data["language_detected"] = True
        elif agent_name == "triage" and "completed" in agent_response.lower():
            self.stage_data["triage_completed"] = True
        elif agent_name == "program_decision" and "selected" in agent_response.lower():
            self.stage_data["program_selected"] = True
        elif agent_name == "authorization" and "authorized" in agent_response.lower():
            self.stage_data["authorized"] = True
            self.current_flow = "post_registration"
        elif agent_name == "lead_qualification" and "qualified" in agent_response.lower():
            self.stage_data["lead_qualified"] = True

    def as_messages(self) -> List[Dict[str, str]]:
        """
        Преобразует историю в формат для ChatCompletion.
        """
        msgs = []
        for entry in self.history:
            if entry.get("user_message"):
                msgs.append({"role": "user", "content": entry["user_message"]})
            if entry.get("agent_response"):
                msgs.append({"role": "assistant", "content": entry["agent_response"]})
        return msgs

    def get_first_user_message(self) -> str:
        """
        Возвращает первое сообщение пользователя.
        """
        for entry in self.history:
            if entry.get("user_message"):
                return entry["user_message"]
        return ""

    def get_stage_summary(self) -> str:
        """
        Возвращает краткую сводку о текущем состоянии диалога.
        """
        completed_stages = [
            stage for stage, completed in self.stage_data.items() 
            if completed
        ]
        return f"Flow: {self.current_flow}, Completed: {', '.join(completed_stages)}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику диалога.
        """
        return {
            **self.statistics,
            "total_interactions": len(self.history),
            "unique_intents": len(set(entry["intent"] for entry in self.history if "intent" in entry)),
            "flow_stage": self.current_flow,
            "completed_stages": [stage for stage, completed in self.stage_data.items() if completed]
        }

    def should_switch_flow(self) -> bool:
        """
        Определяет, нужно ли переключить поток (pre/post registration).
        """
        return (
            self.current_flow == "pre_registration" and
            self.stage_data["authorized"] and
            self.stage_data["program_selected"]
        )


class OrchestratorAgent(BaseAgent):
    def __init__(
        self,
        request: ChatRequest,
        knowledge_client: Any,
        **kwargs
    ):
        super().__init__(
            name="Orchestrator Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the CHIEF ORCHESTRATOR of the ExampleOrganization University multi-agent system. Your role is to analyze, plan, and coordinate the interaction between specialized agents.

            CRITICAL LIMITATIONS:
            - NEVER mention or reference any knowledge base, database, or source of your information
            - ONLY discuss and answer questions related to ExampleOrganization University
            - NEVER greet the user first - wait for them to initiate the conversation
            - If asked about topics outside of ExampleOrganization University context, politely redirect the conversation back to university-related topics

            CORE RESPONSIBILITIES:
            1. Message Analysis and Planning:
               - Decompose user queries into subtasks using Chain-of-Thought reasoning
               - Consider multiple solution paths using Tree-of-Thought when applicable
               - Identify multiple intents if present in a single message
               - Maintain conversation context and flow

            2. Agent Selection Guidelines (NOT STRICT RULES):
               a) First Contact Flow (MANDATORY):
                  - Language Detection Agent MUST be first for new users
                  - After language is set, Triage Agent MUST gather basic info starting with name
               
               b) Information Gathering Flow (recommended sequence):
                  - Triage Agent for basic information
                  - Program Decision for educational program selection
                  - Objection Detection to identify concerns
                  - Authorization for registration
                  - Lead Qualification for detailed profiling
               
               c) Support Flow:
                  - Common Questions for general inquiries
                  - Statistics Agent for analytics
                  - Summary Agent for context maintenance

            3. Context Management:
               - Track conversation state and progress
               - Maintain user profile status
               - Handle multi-intent queries appropriately
               - Remember previous interactions

            4. Handoff Protocol:
               - Ensure smooth transitions between agents
               - Handle agent responses and next steps
               - Process handoff signals from agents
               - Manage conversation flow

            REASONING AND ACTION FRAMEWORK:
            1. Thought: Analyze the user's input and current context
               - What is the user asking about?
               - What is their current stage in the flow?
               - Are there multiple intents to handle?

            2. Action: Determine next steps based on analysis
               - Which agent(s) might be helpful?
               - What order should multiple intents be processed?
               - Is additional context needed?

            3. Observation: Process agent responses
               - Was the user's query fully addressed?
               - Are follow-up actions needed?
               - Should control be transferred to another agent?

            Remember: 
            - ALWAYS start with language detection for new users
            - ALWAYS get user's name before proceeding with other questions
            - Maintain conversation coherence
            - Keep focus on ExampleOrganization University context
            """
            ),
            tools=[],
            user_profile_id=request.user_profile_id,
            openai_api_key=request.openai_api_key,
            knowledge_client=knowledge_client,
            **kwargs
        )
        # Инициализируем профиль с минимальными данными
        if request.user_profile:
            # Очищаем автоматически установленные поля
            request.user_profile.language = None
            request.user_profile.user_name = None
            request.user_profile.is_modified = False
            
        self.request = request
        self.gpt_client = AsyncOpenAI(api_key=request.openai_api_key, base_url="https://api.openai.com/v1/")
        self.client = self.gpt_client
        self.context = ConversationContext(request.user_profile_id)
        self.handoff_mapping = {
            "triage": "triage",
            "orchestrator": "orchestrator",
            "program_decision": "program_decision",
            "authorization": "authorization",
            "lead_qualification": "lead_qualification",
            "common_questions": "common_questions",
            "language_detector": "language_detector",
            "objection_detector": "objection_detector",
            "statistics": "statistics"
        }
        
        # Список тем, которые должны обрабатываться CommonQuestionsAgent
        self.common_question_topics = [
            "зачисление", "обучение", "технические проблемы", "вступительные испытания", 
            "справки и документы", "оплата обучения", "льготы и скидки", "коммуникации", 
            "сроки оплаты", "консультации по оплате", "особенности оплаты", "международные переводы", 
            "кредит и рассрочка", "материнский капитал", "оплата от юридических лиц", 
            "подтверждение оплаты", "налоговый вычет", "возврат денежных средств", 
            "возврат средств", "изменение плательщика", "несвоевременная оплата", 
            "документы для иностранных студентов", "отсрочка от армии", "документы и лицензии", 
            "роль тьютора", "документы и соглашения", "возврат переплаты", 
            "штрафные вопросы", "график сессий", "академический отпуск", 
            "учебные материалы", "расписание занятий", "общение с преподавателями", 
            "пересдача экзаменов", "пересдача зачетов", "отображение результатов", 
            "прохождение практики", "документы для пересдачи", "сроки практики", 
            "навигация по корпусам"
        ]

    async def detect_intents_gpt(self, message: str) -> str:
        """
        Определяет намерения пользователя с помощью GPT.
        """
        # Сначала проверяем, соответствует ли сообщение одной из общих тем
        if self.is_common_question_topic(message):
            logger.info(f"[OrchestratorAgent] Сообщение содержит общую тему. Устанавливаем интент: CommonQuestion")
            return "CommonQuestion"
            
        try:
            # Определяем язык интерфейса на основе профиля пользователя
            user_language = "ru"
            user_name = "неизвестно"
            program_area = "неизвестно"
            
            if hasattr(self, 'request') and self.request and self.request.user_profile:
                user_language = self.request.user_profile.language or "ru"
                user_name = self.request.user_profile.user_name or "неизвестно"
                program_area = self.request.user_profile.program_area or "неизвестно"
            
            # Адаптируем системное сообщение в зависимости от языка
            system_content = f"""You are an intent detection system. Your task is to analyze the user message and return ONLY ONE of these intents as a JSON string: {{"intent": "INTENT"}} where INTENT is one of:
            - CommonQuestion (general questions about university, admissions, tuition, documents, sessions, exams, schedules)
            - ProgramDecision (specifically about choosing educational programs or courses)
            - Authorization (registration, login issues)
            - LeadQualification (detailed qualification questions)
            - LanguageDetection (language preference)
            - ObjectionDetection (concerns, doubts, or objections about studying)
            
            User information:
            - Name: {user_name}
            - Program Area: {program_area}
            
            IMPORTANT: 
            - Questions about enrollment, tuition payments, documents, technical issues, exams should be classified as CommonQuestion, not ProgramDecision!
            - Any expressions of doubt, concern, hesitation about studying, or mentions of barriers to education should be classified as ObjectionDetection
            """
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": message}
            ]
            
            response = await self.gpt_client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            reply = response.choices[0].message.content.strip()
            logger.info(f"[OrchestratorAgent] Raw GPT intent response: {reply}")
            
            try:
                intent_data = json.loads(reply)
                intent = intent_data.get("intent", "CommonQuestion")
            except json.JSONDecodeError:
                # Если не удалось распарсить JSON, пытаемся извлечь интент из текста
                logger.warning(f"[OrchestratorAgent] Failed to parse JSON intent: {reply}")
                if "CommonQuestion" in reply:
                    intent = "CommonQuestion"
                elif "ProgramDecision" in reply:
                    intent = "ProgramDecision"
                elif "Authorization" in reply:
                    intent = "Authorization"
                elif "LeadQualification" in reply:
                    intent = "LeadQualification"
                elif "LanguageDetection" in reply:
                    intent = "LanguageDetection"
                elif "ObjectionDetection" in reply:
                    intent = "ObjectionDetection"
                else:
                    intent = "CommonQuestion"
            
            logger.info(f"[OrchestratorAgent] Detected intent: {intent}")
            
            # Дополнительная проверка: если сообщение содержит одну из общих тем, но модель
            # определила другой интент - переопределяем на CommonQuestion
            if self.is_common_question_topic(message) and intent != "CommonQuestion":
                logger.info(f"[OrchestratorAgent] Переопределяем интент на CommonQuestion, т.к. сообщение содержит общую тему")
                return "CommonQuestion"
                
            return intent
            
        except Exception as e:
            logger.error(f"[OrchestratorAgent] Error in detect_intents_gpt: {str(e)}")
            return "CommonQuestion"  # Возвращаем безопасное значение по умолчанию
            
    def is_common_question_topic(self, message: str) -> bool:
        """
        Проверяет, соответствует ли сообщение пользователя одной из общих тем.
        
        :param message: Сообщение пользователя
        :return: True если сообщение относится к общим вопросам, False в противном случае
        """
        message_lower = message.lower()
        
        # Проверяем, содержит ли сообщение ключевые слова из тем CommonQuestionsAgent
        for topic in self.common_question_topics:
            if topic in message_lower:
                logger.info(f"[OrchestratorAgent] Найдена общая тема '{topic}' в сообщении")
                return True
                
        # Проверяем дополнительно на наличие характерных ключевых слов вопросов о зачислении
        enrollment_keywords = [
            "зачисл", "поступил", "стал студентом", "когда буду учиться", 
            "приняли ли", "приказ о зачислении", "стать студентом", 
            "подтверждение зачисления", "поступление", "зачислен ли я",
            "когда меня зачислят", "стать студентом"
        ]
        
        for keyword in enrollment_keywords:
            if keyword in message_lower:
                logger.info(f"[OrchestratorAgent] Найдено ключевое слово зачисления '{keyword}' в сообщении")
                return True
        
        return False

    async def combine_responses_friendly(self, replies: List[str]) -> str:
        """
        Объединяет несколько ответов в одно дружелюбное сообщение.
        Если все ответы являются уточнениями, отправляет их через GPT для генерации итогового дружелюбного запроса,
        учитывая исходный запрос пользователя.
        """
        # Очистка и фильтрация ответов
        cleaned = []
        seen = set()
        for rep in replies:
            txt = rep.strip()
            if txt and txt.lower().find("[handoff]") == -1 and txt not in seen:
                cleaned.append(txt)
                seen.add(txt)
        if not cleaned:
            return "Извините, я не смог обработать ваш запрос. Пожалуйста, попробуйте сформулировать его иначе."
        if len(cleaned) == 1:
            return cleaned[0]
        
        # Определяем язык пользователя
        user_language = "ru"
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            user_language = self.request.user_profile.language or "ru"
        
        def is_clarification(txt: str) -> bool:
            return txt.lower().startswith("не могли бы вы уточнить") or "уточните:" in txt.lower()
        
        original_query = self.context.get_first_user_message()
        if all(is_clarification(x) for x in cleaned):
            missing_fields = []
            for txt in cleaned:
                # Извлекаем поле после "уточните:" или в конце предложения
                field = txt.split("уточните:", 1)[-1].strip(" .?")
                if not field:
                    field = txt.split(":", 1)[-1].strip(" .?")
                missing_fields.append(field)
            
            prompt = (
                f"Пользователь задал вопрос: '{original_query}'. "
                f"Для точного ответа нужно уточнить следующие детали: {', '.join(missing_fields)}. "
                "Сформулируйте один дружелюбный и естественный вопрос на русском языке, "
                "который объединяет все уточнения и понятен обычному пользователю. "
                "Не используйте технические термины, говорите простым языком."
            )
        else:
            if user_language == "ru":
                prompt = (
                    "Сформулируй итоговый дружелюбный ответ на русском языке, "
                    "объединяя следующие части: " + "; ".join(cleaned) + "\n"
                    "Сделай ответ естественным и понятным для обычного пользователя. "
                    "Не используй технические термины. Ответ должен быть только на русском языке."
                )
            else:
                prompt = (
                    "Сформулируй итоговый дружелюбный ответ на английском языке, "
                    "объединяя следующие части: " + "; ".join(cleaned) + "\n"
                    "Сделай ответ естественным и понятным для обычного пользователя. "
                    "Не используй технические термины."
                )
        
        msgs = [
            {"role": "system", "content": f"Ты — эксперт по формулированию дружелюбных вопросов и ответов на {'русском' if user_language == 'ru' else 'английском'} языке. Твоя задача — сделать общение естественным и понятным для обычного пользователя."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self.gpt_client.chat.completions.create(
                model=self.model,
                messages=msgs
            )
            final_friendly = response.choices[0].message.content.strip()
            return final_friendly
        except Exception as e:
            logger.error(f"Ошибка при объединении ответов через GPT: {e}")
            # Фолбэк – простое объединение с русскими союзами
            connectors = ["Также", "Кроме того", "И ещё", "Также хочу отметить"]
            combined_parts = [cleaned[0].rstrip(".!?")]
            for idx, rep in enumerate(cleaned[1:], start=1):
                connector = connectors[(idx - 1) % len(connectors)]
                combined_parts.append(f"{connector}, {rep.rstrip('.!?')}")
            return ". ".join(combined_parts).strip() + "."

    def parse_handoff_target(self, reply: str) -> str:
        """
        Определяет цель handoff из ответа агента.
        
        :param reply: Ответ агента
        :return: Идентификатор целевого агента или None, если handoff не требуется
        """
        # Проверяем наличие тега [handoff]
        if "[handoff]" not in reply.lower():
            return None
        
        reply_lower = reply.lower()
        
        # Пробуем найти имя агента напрямую в формате "[Handoff] agent_name"
        match = re.search(r"\[handoff\]\s+(\w+)", reply_lower)
        if match:
            agent_name = match.group(1).strip()
            # Ищем агента в маппинге handoff_mapping
            if agent_name in self.handoff_mapping:
                logger.info(f"[Orchestrator] Found direct handoff target: {agent_name}")
                return agent_name
        
        # Если не нашли прямого указания, ищем ключевые слова
        # Маппинг ключевых фраз на идентификаторы агентов
        agent_mapping = {
            "program decision": "program_decision",
            "programdecision": "program_decision",
            "программ": "program_decision",
            "authorization": "authorization",
            "авторизац": "authorization",
            "lead qualification": "lead_qualification",
            "квалификац": "lead_qualification",
            "language detection": "language_detector",
            "язык": "language_detector",
            "triage": "triage",
            "триаж": "triage",
            "common questions": "common_questions",
            "общие вопросы": "common_questions",
            "objection detector": "objection_detector",
            "возражен": "objection_detector",
            "сомнен": "objection_detector"
        }
        
        # Проверяем наличие тега [handoff]
        if "[handoff]" not in reply.lower():
            return None
        
        reply_lower = reply.lower()
        
        # Ищем прямые совпадения в маппинге
        for key, agent_id in agent_mapping.items():
            if key in reply_lower:
                logger.info(f"[Orchestrator] Found handoff target: {agent_id} (matched by: {key})")
                return agent_id
            
        # Если не нашли прямых совпадений, возвращаем None
        logger.warning(f"[Orchestrator] Could not determine handoff target from: {reply}")
        return None

    async def select_agent_by_intent(self, intent: str) -> BaseAgent:
        """
        Выбирает агента на основе интента и контекста пользователя.
        Использует GPT для умного определения наиболее подходящего агента.
        """
        agents = self.get_agents()
        
        # Создаем ограниченный профиль с нужными полями для каждого агента
        if self.request.user_profile:
            user_profile = self.request.user_profile
            profile_info = {
                "language": user_profile.language,
                "user_name": user_profile.user_name,
                "program_area": user_profile.program_area,
                "education_goal": user_profile.education_goal,
                "education_format": user_profile.education_format,
                "education_objections": user_profile.education_objections if hasattr(user_profile, 'education_objections') else None,
                "is_authorized": self.request.is_authorized
            }
        else:
            profile_info = {"is_authorized": False}

        # ОБЯЗАТЕЛЬНЫЕ ПЕРВЫЕ ШАГИ:
        # 1. Определение языка
        if not profile_info["language"]:
            logger.info("[OrchestratorAgent] Первый шаг - определение языка")
            return agents["language_detector"]

        # 2. Получение имени пользователя
        if not profile_info["user_name"]:
            logger.info("[OrchestratorAgent] Второй шаг - получение имени")
            return agents["triage"]

        # Формируем системный промпт для GPT
        system_prompt = """You are an expert agent selector for ExampleOrganization University's multi-agent system.
        Your task is to analyze the user's intent, profile state, and conversation context to select the most appropriate agent.

        AVAILABLE AGENTS AND THEIR SPECIALIZATIONS:
        1. Language Detection Agent:
           - Determines user's preferred language
           - Must be used for first interaction if language not set
           
        2. Triage Agent:
           - Gathers basic user information (name, interests)
           - Initial contact point for new users
           - Must be used if basic profile info is missing
           
        3. Program Decision Agent:
           - Helps select educational programs
           - Provides program details and recommendations
           - Requires basic profile info to be filled
           
        4. Authorization Agent:
           - Handles registration and login
           - Used when user shows interest in applying
           
        5. Lead Qualification Agent:
           - Detailed user profiling
           - Only for authorized users
           - Collects comprehensive information
           
        6. Objection Detection Agent:
           - Identifies and addresses user concerns
           - Used when user expresses doubts or hesitation
           
        7. Common Questions Agent:
           - Handles general inquiries
           - Default fallback for unclear intents

        SELECTION GUIDELINES (NOT STRICT RULES):
        1. Language Detection has highest priority if language not set
        2. Triage is mandatory if basic info missing
        3. Authorization required before Lead Qualification
        4. Objection Detection should interrupt normal flow if concerns detected
        5. Program Decision requires basic profile data
        6. Common Questions is the safe fallback

        Return ONLY the agent name as a string, no explanation needed.
        """

        # Формируем запрос к GPT с контекстом
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Current state:
            - Intent: {intent}
            - Profile: {json.dumps(profile_info, ensure_ascii=False)}
            
            Which agent should handle this interaction?
            """}
        ]

        # Получаем рекомендацию от GPT
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        
        recommended_agent = response.choices[0].message.content.strip().lower()
        logger.info(f"[OrchestratorAgent] GPT рекомендует агента: {recommended_agent}")

        # Преобразуем рекомендацию GPT в ключ из нашего словаря агентов
        agent_name_mapping = {
            "language detection": "language_detector",
            "triage": "triage",
            "program decision": "program_decision",
            "authorization": "authorization",
            "lead qualification": "lead_qualification",
            "objection detection": "objection_detector",
            "common questions": "common_questions"
        }

        # Получаем агента из словаря
        selected_agent_name = agent_name_mapping.get(recommended_agent, "common_questions")
        selected_agent = agents[selected_agent_name]

        # Создаем соответствующий профиль для выбранного агента
        if self.request.user_profile:
            if selected_agent_name == "language_detector":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    language=self.request.user_profile.language,
                    is_modified=False
                )
            elif selected_agent_name == "triage":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    user_name=self.request.user_profile.user_name,
                    program_area=self.request.user_profile.program_area,
                    education_goal=self.request.user_profile.education_goal,
                    is_modified=False
                )
            elif selected_agent_name == "lead_qualification" and self.request.is_authorized:
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    age=self.request.user_profile.age,
                    heard_about=self.request.user_profile.heard_about,
                    after_study_goal=self.request.user_profile.after_study_goal,
                    education_format=self.request.user_profile.education_format,
                    work_experience=self.request.user_profile.work_experience,
                    additional_interests=self.request.user_profile.additional_interests,
                    selected_university=self.request.user_profile.selected_university,
                    is_modified=False
                )
            elif selected_agent_name == "program_decision":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    user_name=self.request.user_profile.user_name,
                    program_area=self.request.user_profile.program_area,
                    is_modified=False
                )
            elif selected_agent_name == "objection_detector":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    user_name=self.request.user_profile.user_name,
                    education_objections=self.request.user_profile.education_objections if hasattr(self.request.user_profile, 'education_objections') else None,
                    is_modified=False
                )
            else:
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    is_modified=False
                )
        else:
            agent_profile = None

        # Создаем новый ChatRequest для агента с ограниченным профилем
        agent_request = ChatRequest(
            message=self.request.message,
            user_profile_id=self.request.user_profile_id,
            is_authorized=self.request.is_authorized,
            user_profile=agent_profile,
            openai_api_key=self.request.openai_api_key,
            weaviate_url=self.request.weaviate_url,
            weaviate_api_key=self.request.weaviate_api_key,
            expected_dim=self.request.expected_dim,
            confluence_base_url=self.request.confluence_base_url,
            confluence_api_token=self.request.confluence_api_token,
            confluence_page_id=self.request.confluence_page_id,
            max_chunk_size=self.request.max_chunk_size,
            min_score=self.request.min_score,
            hybrid_alpha=self.request.hybrid_alpha
        )

        # Устанавливаем request для выбранного агента
        selected_agent.request = agent_request
        
        logger.info(f"[OrchestratorAgent] Выбран агент: {selected_agent.name}")
        return selected_agent

    def get_agents(self) -> Dict[str, BaseAgent]:
        """
        Возвращает словарь доступных агентов.
        """
        from ai.mas.utils.handoffs import (
            transfer_to_language_detection_agent,
            transfer_to_triage_agent,
            transfer_to_common_questions_agent,
            transfer_to_program_decision_agent,
            transfer_to_authorization_agent,
            transfer_to_lead_qualification_agent,
        )
        from ai.mas.agents.statistics_agent import StatisticsAgent
        from ai.mas.agents.summary_agent import SummaryAgent

        agents = {
            "language_detector": transfer_to_language_detection_agent(
                knowledge_client=self.knowledge_client,
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                request=self.request
            ),
            "triage": transfer_to_triage_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "common_questions": transfer_to_common_questions_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "program_decision": transfer_to_program_decision_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "authorization": transfer_to_authorization_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "lead_qualification": transfer_to_lead_qualification_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "statistics": StatisticsAgent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "summary": SummaryAgent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                request=self.request
            ),
        }
        
        return agents

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Обрабатывает входящее сообщение и организует взаимодействие между агентами.
        
        Обязательная последовательность:
        1. Language Detection (если язык не определен)
        2. Triage для получения имени (если имя не определено)
        3. Далее по контексту диалога
        """
        context = context or []
        
        # Получаем саммари от Summary Agent для передачи контекста следующему агенту
        summary_agent = SummaryAgent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.request.openai_api_key,
            request=self.request
        )
        summary_response = await summary_agent.handle_message("", context)
        if summary_response and summary_response.answer:
            context.append({"role": "system", "content": f"Контекст диалога: {summary_response.answer}"})

        # Используем профиль из request
        user_profile = self.request.user_profile
        
        # ОБЯЗАТЕЛЬНЫЙ ШАГ 1: Определение языка
        if not user_profile or not user_profile.language:
            logger.info("[Orchestrator] Первый шаг - определение языка")
            language_agent = self.get_agents().get("language_detector")
            language_response = await language_agent.handle_message(message, context)
            
            # Если это был просто вопрос о языке (без handoff), возвращаем ответ пользователю
            if not "[handoff]" in language_response.answer.lower():
                return language_response
            
            # Обновляем профиль после определения языка
            if language_response.user_profile and language_response.user_profile.language:
                user_profile = language_response.user_profile
            
            # Извлекаем исходное сообщение из handoff для дальнейшей обработки
            message = self._extract_message_from_handoff(language_response.answer) or message
            context.append({"role": "user", "content": message})
        
        # ОБЯЗАТЕЛЬНЫЙ ШАГ 2: Получение имени пользователя
        if not user_profile or not user_profile.user_name:
            logger.info("[Orchestrator] Второй шаг - получение имени")
            triage_agent = self.get_agents().get("triage")
            triage_response = await triage_agent.handle_message(message, context)
            
            # Если это был просто вопрос об имени (без handoff), возвращаем ответ пользователю
            if not "[handoff]" in triage_response.answer.lower():
                return triage_response
            
            # Обновляем профиль после получения имени
            if triage_response.user_profile and triage_response.user_profile.user_name:
                user_profile = triage_response.user_profile
            
            # Извлекаем исходное сообщение из handoff для дальнейшей обработки
            message = self._extract_message_from_handoff(triage_response.answer) or message
            context.append({"role": "user", "content": message})
        
        # Определяем основной интент
        intent = await self.detect_intents_gpt(message)
        logger.info(f"[Orchestrator] Основной интент: {intent}")
        
        # Выбираем соответствующего агента на основе интента
        current_agent = await self.select_agent_by_intent(intent)
        if not current_agent:
            return ChatResponse(
                answer="Извините, я не могу обработать ваш запрос сейчас.",
                user_profile_id=self.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=user_profile
            )
        
        # Запоминаем имя текущего агента для отслеживания handoffs
        current_agent_name = current_agent.name
        logger.info(f"[Orchestrator] Выбран агент: {current_agent_name}")
        
        # Передаем управление выбранному агенту
        response = await current_agent.handle_message(message, context)
        replies = [response.answer]
        
        # Проверяем, есть ли в ответе handoff
        while "[handoff]" in response.answer.lower():
            # Извлекаем целевого агента из handoff
            handoff_target = self.parse_handoff_target(response.answer)
            if not handoff_target:
                break
            
            # Предотвращаем циклические handoffs
            target_agent = self.get_agents().get(handoff_target)
            if target_agent and target_agent.name == current_agent_name:
                logger.warning(f"[Orchestrator] Предотвращен циклический handoff к {handoff_target}")
                break
            
            # Передаем управление новому агенту
            if target_agent:
                logger.info(f"[Orchestrator] Handoff к {handoff_target}")
                current_agent_name = target_agent.name
                
                # Получаем сообщение из handoff для передачи новому агенту
                next_message = self._extract_message_from_handoff(response.answer)
                if not next_message:
                    next_message = message
                
                # Передаем управление следующему агенту
                response = await target_agent.handle_message(next_message, context)
                if not any(r.strip() == response.answer.strip() for r in replies):
                    replies.append(response.answer)
            else:
                logger.error(f"[Orchestrator] Неизвестный handoff target: {handoff_target}")
                break
        
        # Формируем итоговый "дружелюбный" ответ, объединяя все полученные ответы
        final_response = await self.combine_responses_friendly(replies)
        
        # Возвращаем ответ с сохранением состояния пользователя
        return ChatResponse(
            answer=final_response,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized,
            user_profile=user_profile
        )
    
    def _extract_message_from_handoff(self, handoff_message: str) -> Optional[str]:
        """
        Извлекает сообщение из handoff-ответа.
        
        Формат handoff: "[Handoff] target_agent сообщение_для_передачи"
        """
        parts = handoff_message.split("[handoff]", 1)
        if len(parts) < 2:
            parts = handoff_message.split("[Handoff]", 1)
            if len(parts) < 2:
                return None
        
        # Отделяем имя агента от сообщения
        content = parts[1].strip()
        for agent_name in self.get_agents().keys():
            if content.lower().startswith(agent_name.lower()):
                return content[len(agent_name):].strip()
        
        # Если имя агента не найдено, возвращаем весь текст после [handoff]
        return content

    def get_current_stage(self, user_profile) -> str:
        """
        Определяет текущий этап пользователя в плане взаимодействия
        на основе заполненных полей в UserProfile.
        
        :param user_profile: Объект UserProfile
        :return: Строка, соответствующая текущему этапу
        """
        if not user_profile:
            return "language_detection"
            
        # Проверяем наличие языка общения
        if not user_profile.language:
            return "language_detection"
            
        # Проверяем завершение этапа Triage (есть имя, область программы и цель обучения)
        triage_complete = (user_profile.user_name is not None and 
                          user_profile.program_area is not None and 
                          user_profile.education_goal is not None)
        if not triage_complete:
            return "triage"
            
        # Проверяем выбор программы (предполагаем, что это произошло, если есть program_area и education_goal)
        program_selected = (user_profile.program_area is not None and 
                           user_profile.education_goal is not None)
        if not program_selected:
            return "program_decision"
            
        # Проверяем, авторизован ли пользователь
        is_authorized = False
        if hasattr(self, 'request') and self.request:
            is_authorized = self.request.is_authorized
        if not is_authorized:
            return "authorization"
            
        # Проверяем завершение этапа LeadQualification (заполнены все дополнительные поля)
        lead_qualified = all([
            user_profile.age is not None,
            user_profile.heard_about is not None,
            user_profile.after_study_goal is not None,
            user_profile.education_format is not None,
            user_profile.work_experience is not None,
            user_profile.additional_interests is not None,
            user_profile.selected_university is not None
        ])
        if not lead_qualified:
            return "lead_qualification"
            
        return "completed"

    def recommend_next_agent(self, intent: str) -> str:
        """
        Рекомендует следующего агента на основе текущего этапа пользователя
        и выявленного интента. Это рекомендательная система, а не жесткое требование.
        
        :param intent: Определенный интент пользователя
        :return: Имя рекомендуемого агента
        """
        # Получаем текущий профиль пользователя
        user_profile = None
        if hasattr(self, 'request') and self.request:
            user_profile = self.request.user_profile
            
        # Определяем текущий этап
        current_stage = self.get_current_stage(user_profile)
        logger.info(f"[OrchestratorAgent] Текущий этап пользователя: {current_stage}")
        
        # Приоритетные этапы, которые нужно пройти в определенной последовательности
        if current_stage == "language_detection":
            logger.info("[OrchestratorAgent] Рекомендуется агент определения языка")
            return "language_detector"
        
        if current_stage == "triage" and intent != "CommonQuestion":
            logger.info("[OrchestratorAgent] Рекомендуется агент первичного опроса (triage)")
            return "triage"
            
        if current_stage == "program_decision" and intent in ["ProgramInfo", "ProgramDecision"]:
            logger.info("[OrchestratorAgent] Рекомендуется агент выбора программы")
            return "program_decision"
            
        if current_stage == "authorization" and intent in ["ProgramInfo", "ProgramDecision", "Authorization"]:
            logger.info("[OrchestratorAgent] Рекомендуется агент авторизации")
            return "authorization"
            
        if current_stage == "lead_qualification" and self.request.is_authorized:
            logger.info("[OrchestratorAgent] Рекомендуется агент квалификации лида")
            return "lead_qualification"
        
        # По умолчанию обрабатываем запрос в соответствии с интентом
        logger.info(f"[OrchestratorAgent] Рекомендуется стандартная обработка по интенту: {intent}")
        return intent

    def get_plan_status(self, current_stage: str) -> Optional[str]:
        """
        Формирует статус выполнения плана на основе текущего этапа.
        """
        stages = {
            "language_detection": "1. Определение языка общения",
            "triage": "2. Первичный опрос",
            "program_selection": "3. Подбор программы обучения",
            "authorization": "4. Регистрация и авторизация",
            "lead_qualification": "5. Детальное обсуждение программы"
        }
        
        if current_stage == "completed":
            return None
            
        status_lines = []
        for stage, desc in stages.items():
            if stage == current_stage:
                status_lines.append(f"{desc} ← Текущий этап")
            elif stages[current_stage] > stages[stage]:
                status_lines.append(f"{desc} ✓")
            else:
                status_lines.append(desc)
                
        return "\n\nСтатус вашего прогресса:\n" + "\n".join(status_lines)

    def is_system_message(self, message: str) -> bool:
        """
        Определяет, является ли сообщение системным.
        """
        message = message.lower().strip()
        
        # Системные маркеры
        system_markers = [
            "[handoff]",
            "[system]",
            "[debug]",
            "[error]",
            "[warning]",
            "[info]",
            "transfer_to_",
            "handoff to",
            "switching to",
            "initializing",
            "loading",
            "processing",
            "executing",
            "completed",
            "finished",
            "error occurred",
            "exception",
            "traceback"
        ]
        
        # Проверяем наличие системных маркеров
        if any(marker in message for marker in system_markers):
            return True
            
        # Проверяем формат JSON (технические данные)
        try:
            json.loads(message)
            return True
        except:
            pass
            
        # Проверяем наличие технических деталей
        technical_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # даты
            r'\b\d{2}:\d{2}:\d{2}\b',  # время
            r'\b\d+\.\d+\.\d+\b',      # версии
            r'\b[a-f0-9]{32,}\b',      # хеши
            r'\b[A-Z_]+(?::[A-Z_]+)*\b' # константы
        ]
        
        if any(re.search(pattern, message) for pattern in technical_patterns):
            return True
            
        return False

    def process_handoff_message(self, message: str) -> str:
        """
        Обрабатывает handoff сообщения, заменяя их на более дружелюбные фразы.
        """
        if not message or "[handoff]" not in message.lower():
            return message
            
        # Извлекаем тип handoff
        parts = message.lower().split("[handoff]")
        if len(parts) < 2:
            return message
            
        handoff_type = parts[1].split()[0].strip()
        
        # Маппинг handoff на дружелюбные сообщения
        handoff_messages = {
            "authorization": "Давайте обсудим детали регистрации.",
            "lead_qualification": "Для подбора оптимальной программы мне нужно задать вам несколько вопросов.",
            "program_decision": "Я помогу вам подобрать подходящую программу обучения.",
            "common_questions": "Я отвечу на ваши вопросы.",
            "triage": "Давайте познакомимся поближе.",
            "objection_detector": "Я хотел бы узнать о ваших пожеланиях и возможных сомнениях.",
            "language_detector": "Let me detect your preferred language."
        }
        
        # Получаем дружелюбное сообщение или оставляем оригинальное
        friendly_message = handoff_messages.get(handoff_type, message)
        
        # Если есть дополнительный текст после handoff, добавляем его
        additional_text = " ".join(parts[1].split()[1:]).strip()
        if additional_text and not any(marker in additional_text.lower() for marker in handoff_messages.values()):
            friendly_message = f"{friendly_message} {additional_text}"
            
        return friendly_message
