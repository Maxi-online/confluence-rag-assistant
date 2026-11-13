"""
Triage Agent.
Собирает базовую информацию о пользователе: имя, источник и пр.
После получения ответа агент обновляет профиль пользователя и, если какие-то поля ещё не заполнены,
спрашивает их уточнение.
"""

import json
import logging
from json import JSONDecodeError
from openai import AsyncOpenAI
from ai.mas.utils.db import store_user_data_in_db, get_user_profile, update_user_profile
from ai.mas.agents.base_agent import BaseAgent, Agent
from typing import List, Optional, Any, Dict
from pydantic import ConfigDict, PrivateAttr
from ai.mas.main import ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Список обязательных полей, которые должны быть заполнены на этапе Triage
REQUIRED_FIELDS = ["user_name", "how_heard_about_us"]

# Поля, которые пользователь уже проигнорировал
IGNORED_FIELDS = set()

class TriageAgent(BaseAgent):
    # Используем новую конфигурацию Pydantic v2 с разрешением дополнительных полей
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    # Приватные атрибуты для хранения операционных данных (не проходят валидацию)
    _user_profile_id: str = PrivateAttr()
    _knowledge_client: Any = PrivateAttr()

    def __init__(
        self,
        *,
        user_profile_id: str,
        openai_api_key: str,
        knowledge_client: Any,
        openai_base_url: str = "https://api.openai.com/v1/",
        **kwargs
    ):
        """
        Агент сбора базовой информации о пользователе.

        :param user_profile_id: Идентификатор пользователя.
        :param openai_api_key: API-ключ OpenAI.
        :param knowledge_client: Клиент для работы с базой знаний. Это поле обязательно!
        :param openai_base_url: Базовый URL OpenAI API (по умолчанию).
        """
        # Инициализируем клиента OpenAI для работы с GPT
        gpt_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        # Вызываем конструктор базового класса, передавая только те поля, которые участвуют в валидации
        super().__init__(
            name="Triage Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the TRIAGE AGENT. Your role is to gather basic information about the user and determine their intentions.

            CRITICAL LIMITATIONS:
            - NEVER mention or reference any knowledge base, database, or source of your information
            - NEVER greet the user first - wait for them to initiate the conversation
            - If asked about topics outside of ExampleOrganization University admission, politely redirect the conversation back to admission-related topics
            - NEVER use technical terms or field names like "user_name" or "education_goal"
            - NEVER show any internal structure, database fields, or system information to users
            - ABSOLUTELY NEVER show progress or status lists to users

            CORE RESPONSIBILITIES:
            1. Information Gathering (PRIORITY ORDER):
               - FIRST: Ask for the user's name (this is critical for personalization)
               - SECOND: Ask how they learned about ExampleOrganization University
               - THIRD: Ask about their program interests and education goals
               - Save all collected information
               - Always maintain the context of ExampleOrganization University admission process

            2. Interaction Style:
               - Use friendly, warm, informal and conversational Russian language
               - Talk as a friendly person, not as a technical system
               - Avoid technical terminology and system references completely
               - Ask only ONE question at a time
               - Make questions short, personal and focused
               - After getting user's name, always use it in your responses 
               - Use natural, casual language as in human conversation
               - Maintain a warm, helpful tone
               - NEVER show progress tracking information
               - NEVER mention data processing or database operations

            3. Example Dialogue:
               Agent: "Как вас зовут? Это поможет мне сделать наше общение более персональным."
               User: "Алексей"
               Agent: "Приятно познакомиться, Алексей! Расскажите, пожалуйста, как вы узнали о нашем университете?"

            4. Intent Analysis:
               Using Chain-of-Thought reasoning, determine the user's primary intention:
               - "CommonQuestion" – general questions about ExampleOrganization University, programs, processes
               - "ProgramDecision" – requests for help in choosing a ExampleOrganization educational program
               - "Authorization" – questions about authorization or registration
               - "LeadQualification" – evaluation of potential ExampleOrganization student
               - "Other" – if none of the above applies

            3. Interaction Rules:
               - Always try to answer user questions if you can
               - Maintain polite and professional but warm dialogue
               - If the question is outside your expertise, explain that you'll transfer to a more specialized agent
               - Use natural, conversational communication style
               - Present all information as your direct knowledge, not as retrieved data
               - Keep all responses focused on ExampleOrganization University context
               - Be friendly, supportive and encouraging

            4. Agent Handoffs:
               After determining intentions and collecting basic information, transfer the user to the appropriate agent:
               - For general questions -> CommonQuestions
               - For program selection -> use transfer_to_program_decision function
               - For authorization issues -> Authorization

            RESPONSE FORMAT:
            1. If user name is not known:
               - First question should ALWAYS be: "Как вас зовут? Это поможет нам персонализировать общение."
               - Save name upon receiving it
            
            2. If agent handoff is needed:
               - Answer the current question if possible
               - Use the appropriate transfer function

            3. If both name and how_heard_about_us are known:
               - Transfer to appropriate agent based on their intent
               - Use transfer_to_program_decision in most cases
            """
            ),
            tools=[self.transfer_to_program_decision],
            **kwargs
        )
        # Устанавливаем приватные атрибуты
        self._user_profile_id = user_profile_id
        self._knowledge_client = knowledge_client
        self.gpt_client = gpt_client

    def get_missing_fields(self) -> list:
        """
        Получает текущий профиль пользователя и возвращает список обязательных полей,
        которые ещё не заполнены и не были проигнорированы пользователем.
        """
        profile = get_user_profile(self._knowledge_client, self._user_profile_id) or {}
        # Исключаем поля, которые пользователь уже проигнорировал
        missing = [field for field in REQUIRED_FIELDS if not profile.get(field) and field not in IGNORED_FIELDS]
        return missing

    async def analyze_with_gpt(self, message: str, context: Optional[List[dict]] = None) -> Dict[str, Any]:
        """
        Анализирует сообщение пользователя с помощью GPT для определения имени и области программы.
        GPT самостоятельно решает, когда извлекать и сохранять информацию.
        
        :param message: Сообщение пользователя
        :param context: История диалога (если есть)
        :return: Словарь с информацией, которую нужно сохранить
        """
        context = context or []
        
        # Определяем последний вопрос ассистента (если есть)
        last_assistant_message = ""
        for msg in reversed(context):
            if msg.get("role") == "assistant":
                last_assistant_message = msg.get("content", "")
                break
        
        # Простой и прямой промпт для GPT
        analyze_prompt = f"""
        Проанализируй сообщение пользователя и определи, содержит ли оно имя пользователя или область программы.
        
        Последний вопрос ассистента: "{last_assistant_message}"
        Сообщение пользователя: "{message}"
        
        Если ты уверен, что в сообщении есть имя пользователя или область программы, извлеки их и верни в формате JSON:
        {{
            "user_name": "Имя пользователя, если оно содержится в сообщении",
            "program_area": "Область программы, если она содержится в сообщении"
        }}
        
        Если нет имени или области - укажи null для соответствующего поля. Ответ только в JSON.
        """
        
        analyze_messages = [
            {"role": "system", "content": "Ты - аналитик, который извлекает только имя пользователя и его интересующую область программы из сообщений. Определяй это напрямую и самостоятельно без дополнительных критериев."},
            {"role": "user", "content": analyze_prompt}
        ]
        
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=analyze_messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"GPT-анализ сообщения '{message}': {result_text}")
        
        result = json.loads(result_text)
        # Очищаем пустые значения и null
        return {k: v for k, v in result.items() if v and v != "null" and v != "Null"}

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Обрабатывает сообщение пользователя. 
        GPT анализирует сообщение, контекст и профиль, самостоятельно определяет следующий шаг диалога.
        
        :param message: Сообщение пользователя.
        :param context: История диалога (если имеется).
        :return: Объект ChatResponse с ответом агента.
        """
        context = context or []
        
        # Получаем текущий профиль пользователя
        user_profile = self.request.user_profile if hasattr(self, 'request') and self.request else None
        
        # Анализируем сообщение пользователя для извлечения данных
        extracted_data = await self.analyze_with_gpt(message, context)
        logger.info(f"Извлеченные данные: {extracted_data}")
        
        # Если найдены новые данные, обновляем профиль
        if extracted_data and user_profile:
            if "user_name" in extracted_data and not user_profile.user_name:
                user_profile.user_name = extracted_data["user_name"]
                user_profile.is_modified = True
                logger.info(f"Обновлено имя пользователя: {user_profile.user_name}")
            
            if "program_area" in extracted_data and not user_profile.program_area:
                user_profile.program_area = extracted_data["program_area"]
                user_profile.is_modified = True
                logger.info(f"Обновлена область программы: {user_profile.program_area}")
            
            if "education_goal" in extracted_data and not user_profile.education_goal:
                user_profile.education_goal = extracted_data["education_goal"]
                user_profile.is_modified = True
                logger.info(f"Обновлена цель обучения: {user_profile.education_goal}")
        
        # Проверяем наличие всех необходимых данных в профиле
        if user_profile and user_profile.user_name and user_profile.program_area:
            logger.info("Все необходимые данные собраны, передаем управление ProgramDecisionAgent")
            return ChatResponse(
                answer="[Handoff] program_decision Отлично! Теперь я передам вас специалисту по программам обучения.",
                user_profile_id=self._user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self.request, 'is_authorized') else False,
                user_profile=user_profile
            )
        
        # Формируем системный промпт для GPT с учетом текущего состояния
        system_prompt = f"""Ты - дружелюбный консультант приемной комиссии ExampleOrganization University.
        
        Текущий статус профиля пользователя:
        - Имя: {user_profile.user_name if user_profile and user_profile.user_name else 'не указано'}
        - Интересующая область: {user_profile.program_area if user_profile and user_profile.program_area else 'не указана'}
        - Цель обучения: {user_profile.education_goal if user_profile and user_profile.education_goal else 'не указана'}
        
        ВАЖНЫЕ ПРАВИЛА:
        1. Веди естественный диалог, задавая не более ОДНОГО вопроса за раз
        2. Если пользователь задает вопрос - сначала ответь на него, затем задай свой вопрос
        3. НИКОГДА не спрашивай информацию, которая уже есть в профиле
        4. Используй имя пользователя, как только узнаешь его
        5. Если получены все необходимые данные - передай управление на ProgramDecisionAgent
        
        ФОРМАТ ОТВЕТА:
        Если нужно сохранить данные, верни их в формате JSON в конце ответа:
        {{"user_name": "...", "program_area": "...", "education_goal": "..."}}"""
        
        # Формируем запрос к GPT
        messages = [
            {"role": "system", "content": system_prompt}
        ] + context + [
            {"role": "user", "content": message}
        ]
        
        # Получаем ответ от GPT
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        reply = response.choices[0].message.content.strip()
        
        # Проверяем наличие handoff в ответе
        if "[handoff]" in reply.lower():
            return ChatResponse(
                answer=reply,
                user_profile_id=self._user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self.request, 'is_authorized') else False,
                user_profile=user_profile
            )
        
        # Проверяем наличие JSON с данными для сохранения
        json_start = reply.rfind('{')
        json_end = reply.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = reply[json_start:json_end + 1]
            data = json.loads(json_str)
            
            # Обновляем профиль пользователя только если поля еще не заполнены
            if user_profile:
                if "user_name" in data and data["user_name"] and not user_profile.user_name:
                    user_profile.user_name = data["user_name"]
                    user_profile.is_modified = True
                if "program_area" in data and data["program_area"] and not user_profile.program_area:
                    user_profile.program_area = data["program_area"]
                    user_profile.is_modified = True
                if "education_goal" in data and data["education_goal"] and not user_profile.education_goal:
                    user_profile.education_goal = data["education_goal"]
                    user_profile.is_modified = True
                    
            # Удаляем JSON из ответа пользователю
            reply = reply[:json_start].strip()
            
        return ChatResponse(
            answer=reply,
            user_profile_id=self._user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self.request, 'is_authorized') else False,
            user_profile=user_profile
        )

    async def transfer_to_program_decision(self) -> Agent:
        """
        Transfer to the Program Decision Agent after collecting basic information.
        """
        from ai.mas.utils.handoffs import transfer_to_program_decision_agent
        return transfer_to_program_decision_agent(
            user_profile_id=self._user_profile_id,
            openai_api_key=self.gpt_client.api_key,
            knowledge_client=self._knowledge_client
        )
