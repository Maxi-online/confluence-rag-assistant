#!/usr/bin/env python3
"""
Lead Qualification Agent.
Проводит детальный опрос для сбора информации, необходимой для передачи в отдел продаж.
"""

import logging
from openai import AsyncOpenAI
from pydantic import ConfigDict, PrivateAttr
from typing import List, Optional, Any, Dict
from ai.mas.utils.db import store_user_data_in_db
from ai.mas.utils.handoffs import transfer_back_to_orchestrator, send_to_sales_department, transfer_to_common_questions_agent
from ai.mas.agents.base_agent import BaseAgent, Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeadQualificationAgent(BaseAgent):
    # Конфигурация Pydantic: разрешаем дополнительные поля и произвольные типы
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    _gpt_client: AsyncOpenAI = PrivateAttr()
    _current_stage: int = PrivateAttr(default=0)
    _collected_data: Dict[str, str] = PrivateAttr(default_factory=dict)
    _skipped_questions: List[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        user_profile_id: str,
        openai_api_key: str,
        knowledge_client: Any,
        openai_base_url: str = "https://api.openai.com/v1/",
        request: Any = None,
        **kwargs
    ):
        """
        Агент для детального опроса, собирающий информацию для отдела продаж.

        :param user_profile_id: Идентификатор пользователя.
        :param openai_api_key: API-ключ OpenAI.
        :param knowledge_client: Клиент для работы с базой знаний (если потребуется расширение логики).
        :param openai_base_url: Базовый URL OpenAI API.
        :param request: Объект запроса с контекстом пользователя.
        """
        super().__init__(
            name="Lead Qualification Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the LEAD QUALIFICATION AGENT for ExampleOrganization University. Your role is to conduct a detailed interview while maintaining natural conversation flow.

            CRITICAL LIMITATIONS:
            - NEVER mention or reference any knowledge base, database, or source of your information
            - NEVER greet the user first - wait for them to initiate the conversation
            - If asked about topics outside of ExampleOrganization University context, politely redirect the conversation back to admission-related topics
            - Present all information as your direct knowledge, not as retrieved data

            CORE RESPONSIBILITIES:
            1. Primary Task - Application Form Completion:
               - Collect and verify all required information
               - Explain the purpose of each question naturally
               - Handle user questions while maintaining focus on form completion
               - Provide context for why each piece of information is important

            2. Secondary Task - Question Handling:
               - Answer user questions professionally and accurately
               - Keep responses focused on ExampleOrganization University context
               - Gracefully return to form completion after answering
               - Maintain natural conversation flow

            3. Interview Approach:
               - Ask only ONE question per response
               - Provide context for why the information is needed
               - If a question is ignored or declined, mark it as skipped and never ask again
               - Adapt questions based on previous answers
               - Make the interview feel like a natural conversation

            4. Question Sequence and Context:
               a) Basic Information (if not available):
                  - Name verification (explain it's for personalized communication)
                  - Information source confirmation (to improve our outreach)
               b) Educational Goals:
                  - Career objectives (to match with suitable programs)
                  - Professional development needs (for customized recommendations)
               c) Study Format:
                  - Schedule flexibility requirements (to suggest optimal format)
                  - Location and commuting considerations
               d) Professional Background:
                  - Relevant experience (for potential credits or advanced placement)
                  - Industry knowledge (to tailor program recommendations)
               e) Additional Preferences:
                  - Practical training interests (for program feature matching)
                  - Career support needs (to highlight relevant services)
               f) Program Selection:
                  - Specific interests (to provide detailed information)
                  - Faculty preferences (for specialized guidance)

            5. Response Guidelines:
               - Always explain why information is needed
               - Use natural conversation transitions
               - Acknowledge and answer user questions
               - Return to form completion smoothly
               - Show understanding of user's situation
               - Maintain professional yet friendly tone

            After completing the interview, use the transfer_to_common_questions function to transfer the user to the Common Questions Agent.

            RESPONSE FORMAT:
            1. If user asks a question:
               - Provide clear, accurate answer
               - Naturally transition back to form
               - Example: "That's a great question about our campus facilities. [Answer]. Speaking of your educational journey, I'd like to understand..."

            2. When asking form questions:
               - Provide context first
               - Ask question naturally
               - Example: "To help match you with the most suitable program options, I'd like to understand your career goals better. Could you share what motivates you to pursue further education at this time?"

            3. When handling skipped questions:
               - Never repeat skipped questions
               - Move to next topic naturally
               - Note: If critical information is missing, explain its importance once before marking as skipped

            Remember: 
            - Maintain natural conversation flow
            - Never mention data sources
            - Focus on ExampleOrganization University context
            - Ask only one question per response
            - Provide context for each question
            """
            ),
            tools=[self.transfer_to_common_questions]
        )
        self.user_profile_id = user_profile_id
        self.knowledge_client = knowledge_client
        # Используем object.__setattr__ для установки приватного атрибута без валидации Pydantic
        object.__setattr__(self, '_gpt_client', AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url))
        
        # Структура опроса с контекстом и обоснованием вопросов
        self.interview_stages = [
            {
                "id": "name_verification",
                "context": "Для того чтобы наше общение было более персонализированным,",
                "question": "не могли бы вы подтвердить, как к вам можно обращаться?",
                "skip_if": lambda data: bool(data.get("user_name")),
                "store_as": "user_name"
            },
            {
                "id": "age_verification",
                "context": "Для подбора оптимальной программы обучения,",
                "question": "не могли бы вы указать ваш возраст?",
                "store_as": "age"
            },
            {
                "id": "source_verification",
                "context": "Это поможет нам улучшить наши каналы коммуникации.",
                "question": "Как вы узнали о нашем университете?",
                "skip_if": lambda data: bool(data.get("heard_about")),
                "store_as": "heard_about"
            },
            {
                "id": "education_goal",
                "context": "Чтобы подобрать наиболее подходящую программу и формат обучения,",
                "question": "не могли бы вы поделиться своими целями в получении образования? Это может быть карьерный рост, смена профессии или развитие определенных навыков.",
                "store_as": "after_study_goal"
            },
            {
                "id": "study_format",
                "context": "Для формирования оптимального учебного графика,",
                "question": "какой формат обучения вам был бы наиболее удобен: очный, заочный, онлайн или смешанный?",
                "store_as": "education_format"
            },
            {
                "id": "work_experience",
                "context": "Это поможет нам учесть ваш профессиональный опыт при составлении индивидуальной программы,",
                "question": "расскажите, пожалуйста, о вашем опыте работы в интересующей вас области, если таковой имеется.",
                "store_as": "work_experience"
            },
            {
                "id": "additional_interests",
                "context": "Чтобы сделать ваше обучение максимально эффективным,",
                "question": "какие дополнительные возможности вас интересуют? Например, стажировки, практические занятия, инфраструктура кампуса?",
                "store_as": "additional_interests"
            },
            {
                "id": "program_selection",
                "context": "Для предоставления более детальной информации,",
                "question": "определились ли вы с конкретной программой или факультетом?",
                "store_as": "selected_university"
            }
        ]
        self.request = request

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> str:
        """
        Обрабатывает входящее сообщение от пользователя, следуя структурированному процессу опроса.
        """
        context = context or []
        # Проверка профиля - если есть запрос с профилем, проверяем поля на наличие данных
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            # Если все поля заполнены - не задаем вопросы
            if (self.request.user_profile.age is not None and
                self.request.user_profile.heard_about is not None and
                self.request.user_profile.after_study_goal is not None and
                self.request.user_profile.education_format is not None and
                self.request.user_profile.work_experience is not None and
                self.request.user_profile.additional_interests is not None and
                self.request.user_profile.selected_university is not None):
                return "Спасибо за предоставленную информацию! У меня уже есть все необходимые данные."

        # Оригинальная логика обработки сообщения
        # Проверяем, является ли сообщение вопросом
        messages_for_analysis = (
            [{"role": "system", "content": self.instructions}] +
            [{"role": "user", "content": message}]
        )
        
        response = await self._gpt_client.chat.completions.create(
            model=self.model,
            messages=messages_for_analysis
        )
        reply = response.choices[0].message.content.strip()
        
        # Если это вопрос (определяется по наличию знака "?" или ключевых слов)
        if "?" in message or any(keyword in message.lower() for keyword in ["что", "как", "где", "когда", "почему", "расскажите"]):
            # Отвечаем на вопрос и плавно переходим к следующему этапу опроса
            return reply

        # Сохраняем ответ на предыдущий вопрос, если это не первое сообщение
        if self._current_stage > 0:
            current_stage = self.interview_stages[self._current_stage - 1]
            # Проверяем, не проигнорирован ли вопрос
            if message.strip():  # Если ответ не пустой
                self._collected_data[current_stage["store_as"]] = message
                store_user_data_in_db(self.user_profile_id, {current_stage["store_as"]: message})
                logger.info(f"Сохранен ответ на вопрос {current_stage['id']} в БД: {message}")
                
                # Обновляем соответствующие поля в user_profile, если он доступен
                if hasattr(self, 'request') and self.request and self.request.user_profile:
                    store_as = current_stage["store_as"]
                    
                    # Сохраняем ответ в базу данных
                    store_user_data_in_db(self.user_profile_id, {store_as: message})
                    logger.info(f"Сохранен ответ на вопрос {current_stage['id']} в БД: {message}")
                    
                    # Специальная обработка для числовых полей
                    if store_as == "age":
                        self.request.user_profile.age = int(message.strip()) if message.strip().isdigit() else None
                    # Прямое сопоставление полей
                    elif store_as == "heard_about":
                        self.request.user_profile.heard_about = message
                    elif store_as == "after_study_goal":
                        self.request.user_profile.after_study_goal = message
                    elif store_as == "education_format":
                        self.request.user_profile.education_format = message
                    elif store_as == "work_experience":
                        self.request.user_profile.work_experience = message
                    elif store_as == "additional_interests":
                        self.request.user_profile.additional_interests = message
                    elif store_as == "selected_university":
                        self.request.user_profile.selected_university = message
                    
                    # Отмечаем, что профиль был изменен
                    self.request.user_profile.is_modified = True
                    logger.info(f"Обновлено поле {store_as} в профиле пользователя: {message}")
            else:
                # Если ответ пустой или игнорирующий, добавляем вопрос в список пропущенных
                self._skipped_questions.append(current_stage["id"])
                logger.info(f"Вопрос {current_stage['id']} пропущен")

        # Проверяем, завершен ли опрос
        if self._current_stage >= len(self.interview_stages):
            # Отправляем данные в отдел продаж, если есть контактная информация
            if "contact_info" in self._collected_data:
                send_to_sales_department(self._collected_data)
                logger.info("Данные переданы в отдел продаж")
                # Сбрасываем состояние для возможного нового диалога
                self._current_stage = 0
                self._collected_data = {}
                self._skipped_questions = []
                # Возвращаем управление с явным handoff, но без дополнительного текста
                return "[Handoff] common_questions"

        # Находим следующий вопрос для задания
        while self._current_stage < len(self.interview_stages):
            stage = self.interview_stages[self._current_stage]
            
            # Пропускаем вопрос, если он в списке пропущенных или должен быть пропущен по логике
            if (stage["id"] in self._skipped_questions or 
                ("skip_if" in stage and stage["skip_if"](self._collected_data))):
                self._current_stage += 1
                continue
            
            # Пропускаем вопрос, если соответствующее поле в профиле уже заполнено
            if hasattr(self, 'request') and self.request and self.request.user_profile:
                store_as = stage["store_as"]
                field_mapping = {
                    "preferred_format": "education_format",
                    "how_heard_about_us": "heard_about",
                    "selected_program": "selected_university"
                }
                
                skip_question = False
                if store_as == "age" and self.request.user_profile.age is not None:
                    skip_question = True
                elif store_as == "work_experience" and self.request.user_profile.work_experience is not None:
                    skip_question = True
                elif store_as == "additional_interests" and self.request.user_profile.additional_interests is not None:
                    skip_question = True
                elif store_as in field_mapping:
                    profile_field = field_mapping[store_as]
                    if getattr(self.request.user_profile, profile_field) is not None:
                        skip_question = True
                elif store_as == "after_study_goal" and self.request.user_profile.after_study_goal is not None:
                    skip_question = True
                    
                if skip_question:
                    self._current_stage += 1
                    continue
            
            # Формируем вопрос с контекстом
            question = f"{stage['context']} {stage['question']}"
            if "{source}" in question and "how_heard_about_us" in self._collected_data:
                question = question.format(source=self._collected_data["how_heard_about_us"])
            
            self._current_stage += 1
            return question

        # Если все вопросы заданы или пропущены, возвращаем управление с чистым handoff
        return "[Handoff] common_questions"

    async def transfer_to_common_questions(self) -> Agent:
        """
        Transfer control to the Common Questions Agent.
        Call this when lead qualification is complete or user asks general questions.
        """
        return transfer_to_common_questions_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self._gpt_client.api_key,
            knowledge_client=self.knowledge_client
        )
