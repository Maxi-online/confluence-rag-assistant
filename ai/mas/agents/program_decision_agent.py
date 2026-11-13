#!/usr/bin/env python3
"""
Program Decision Agent.
Выясняет, какую образовательную программу ищет пользователь и уточняет детали 
(сфера, формат, бюджет, цели и т.д.). Если пользователь готов оформить заявку, сообщает: 
'Transfer to Authorization Agent'.
"""

import json
import logging
from json import JSONDecodeError
from openai import AsyncOpenAI
from typing import List, Optional, Any, Dict
from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.utils.db import store_user_data_in_db, get_user_profile, update_user_profile
from ai.mas.utils.graph_base import WeaviateClient
from ai.mas.main import ChatResponse, UserProfile
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

# Определяем список обязательных полей для этапа выбора программы
REQUIRED_PROGRAM_FIELDS = [
    "user_name",
    "program_area"
]

class ProgramDecisionAgent(BaseAgent):
    user_profile_id: str
    openai_api_key: str
    knowledge_client: Any
    request: Any

    # Конфигурация для Pydantic v2: разрешаем дополнительные поля и произвольные типы
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(
        self,
        user_profile_id: str,
        openai_api_key: str,
        knowledge_client: WeaviateClient,
        openai_base_url: str = "https://api.openai.com/v1/",
        request: Any = None,
        **kwargs
    ):
        """
        Агент образовательного консультанта.

        :param user_profile_id: Идентификатор пользователя.
        :param openai_api_key: API-ключ OpenAI (передается явно).
        :param knowledge_client: Клиент для работы с базой знаний Weaviate.
        :param openai_base_url: Базовый URL OpenAI API (по умолчанию).
        :param request: Объект запроса с контекстом пользователя.
        """
        super().__init__(
            name="Program Decision Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the PROGRAM DECISION AGENT—a дружелюбный консультант по образовательным программам ExampleOrganization University.

            ОСНОВНАЯ ФУНКЦИЯ:
            - Помочь пользователю выбрать подходящую образовательную программу
            - Отвечать на вопросы о программах обучения естественным, разговорным языком
            - Задавать ПРОСТЫЕ, КРАТКИЕ уточняющие вопросы - по одному за раз

            ВАЖНО! СТИЛЬ ОБЩЕНИЯ:
            - Используйте дружелюбный, неформальный тон
            - Избегайте длинных, многосоставных вопросов
            - Задавайте ТОЛЬКО ОДИН вопрос за раз
            - Сначала отвечайте на вопрос пользователя, затем задавайте свой вопрос

            ПРИМЕРЫ ДИАЛОГА:
            
            Пользователь: "Какие направления обучения у вас есть?"
            Агент: "У нас широкий выбор направлений! Например, «Бизнес и управление» для тех, кто хочет стать руководителем или развивать свой стартап, «IT и программирование» для будущих разработчиков и аналитиков, «Дизайн и медиа» для креативных профессий, а также «Маркетинг», «Финансы», «Психология» и многие другие. Есть ли у вас интерес к какому-то из этих направлений, или рассказать подробнее?"

            Пользователь: "Меня интересует бизнес и управление"
            Агент: "Отлично! Выбор направления обучения (Бизнес и управление) записан. Почему именно бизнес и управление привлекли ваше внимание?"

            Пользователь: "Хочу стать руководителем"
            Агент: "Отличный выбор! В «Example Organization» вы сможете получить практические навыки управления через реальные кейсы от топ-менеджеров компаний. Наши программы MBA и Executive MBA уже помогли тысячам выпускников занять руководящие должности. Хотели бы узнать больше о наших программах?"

            ПРОЦЕСС ВЗАИМОДЕЙСТВИЯ:
            1. Сначала дайте ПОЛНЫЙ ответ на вопрос пользователя
            2. Затем задайте ОДИН простой уточняющий вопрос
            3. Если пользователь выбрал направление - подтвердите выбор одним предложением
            4. Если пользователь готов подать заявку - используйте transfer_to_authorization_agent

            ПРИМЕРЫ ХОРОШИХ УТОЧНЯЮЩИХ ВОПРОСОВ:
            - "Какое направление вас интересует больше всего?"
            - "Почему именно это направление вас привлекло?"
            - "Вы планируете учиться очно или дистанционно?"
            - "Какое у вас сейчас образование?"

            ПРИМЕРЫ ПЛОХИХ ВОПРОСОВ (СЛИШКОМ СЛОЖНЫЕ!):
            - "Какой у вас уровень образования, какую программу вы хотите изучать, когда хотите начать и какой у вас бюджет?"

            Если пользователь выражает готовность зарегистрироваться или подать заявку, используйте функцию transfer_to_authorization_agent.
            """
            ),
            tools=[self.transfer_to_authorization_agent],
            user_profile_id=user_profile_id,
            openai_api_key=openai_api_key,
            knowledge_client=knowledge_client,
            request=request
        )
        self.user_profile_id = user_profile_id
        self.openai_api_key = openai_api_key
        self.knowledge_client = knowledge_client
        self.gpt_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )

    async def transfer_to_authorization_agent(self) -> Agent:
        """
        Transfer to the Authorization Agent when the user is ready to apply.
        """
        from ai.mas.utils.handoffs import transfer_to_authorization_agent
        return transfer_to_authorization_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.openai_api_key,
            knowledge_client=self.knowledge_client
        )

    @staticmethod
    def format_context(context: List[Any]) -> List[dict]:
        """
        Гарантирует, что каждый элемент списка контекста является словарём с ключами 'role' и 'content'.
        Если элемент является строкой, преобразует его в сообщение от пользователя.
        """
        formatted = []
        for item in context:
            if isinstance(item, dict) and "role" in item and "content" in item:
                formatted.append(item)
            elif isinstance(item, str):
                formatted.append({"role": "user", "content": item})
        return formatted

    def get_missing_program_fields(self) -> List[str]:
        """
        Получает текущий профиль пользователя и возвращает список обязательных полей,
        связанных с выбором программы, которые ещё не заполнены.
        """
        missing = []
        # Сначала проверяем профиль пользователя из request (если он есть)
        request_profile = None
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            request_profile = self.request.user_profile
            logger.info(f"Проверка профиля из request: user_name={request_profile.user_name}, program_area={request_profile.program_area}")
        
        # Затем проверяем профиль из базы данных
        db_profile = get_user_profile(self.knowledge_client, self.user_profile_id) or {}
        
        # Проверяем каждое обязательное поле
        for field in REQUIRED_PROGRAM_FIELDS:
            # Поле считается заполненным, если оно есть хотя бы в одном из профилей
            field_value = None
            
            # Проверяем сначала в request_profile
            if request_profile and getattr(request_profile, field, None):
                field_value = getattr(request_profile, field)
                logger.info(f"Поле {field} найдено в request_profile: {field_value}")
            
            # Если не нашли в request_profile, проверяем в db_profile
            if not field_value and field in db_profile and db_profile.get(field):
                field_value = db_profile.get(field)
                logger.info(f"Поле {field} найдено в db_profile: {field_value}")
            
            # Если поле не найдено ни в одном из профилей, добавляем его в список отсутствующих
            if not field_value:
                logger.info(f"Поле {field} отсутствует во всех профилях")
                missing.append(field)
        
        return missing

    async def search_knowledge_base(self, query: str) -> str:
        """
        Выполняет поиск в базе знаний по заданному запросу.
        Использует search_documents_with_fallback для надежного поиска с автоматическим обновлением базы знаний.
        
        :param query: Запрос пользователя для поиска информации
        :return: Строка с результатами поиска
        """
        try:
            # Используем search_documents_with_fallback для надежного поиска
            # Этот метод автоматически обновит базу знаний через Confluence если нужно
            results = await self.knowledge_client.search_documents_with_fallback(query)
            
            if not results:
                logger.warning(f"Не найдено результатов для запроса: {query}")
                # Пробуем поиск с более общими ключевыми словами
                general_query = " ".join([word for word in query.split() if len(word) > 3])
                if general_query and general_query != query:
                    logger.info(f"Пробуем поиск с общими ключевыми словами: {general_query}")
                    results = await self.knowledge_client.search_documents_with_fallback(general_query)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске в базе знаний: {str(e)}")
            return ""

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Обрабатывает сообщение пользователя и помогает с подбором программы обучения.
        """
        # Получаем профиль пользователя
        user_profile = get_user_profile(self.knowledge_client, self.user_profile_id)
        
        # Создаем объект UserProfile
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            profile = self.request.user_profile
            logger.info(f"Program Decision Agent получил профиль из request: user_name={profile.user_name}, program_area={profile.program_area}")
        else:
            profile = UserProfile(
                user_profile_id=int(self.user_profile_id),
                language=user_profile.get('language') if user_profile else None,
                user_name=user_profile.get('user_name') if user_profile else None,
                program_area=user_profile.get('program_area') if user_profile else None,
                is_modified=False
            )
            logger.info(f"Program Decision Agent создал профиль из БД: user_name={profile.user_name}, program_area={profile.program_area}")

        # Если нет необходимых данных для подбора программы, передаем управление на Triage
        if not profile or not profile.user_name or not profile.program_area:
            logger.info("Отсутствуют необходимые данные для подбора программы")
            return ChatResponse(
                answer="[Handoff] triage Для подбора программы мне нужна дополнительная информация о вас.",
                user_profile_id=self.user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                user_profile=profile
            )

        # Ищем релевантную информацию о программах
        kb_result = await self.search_knowledge_base(message)
        
        # Формируем системный промпт для GPT
        system_prompt = f"""Ты - эксперт по образовательным программам ExampleOrganization University.
        
        Информация о пользователе:
        - Имя: {profile.user_name}
        - Интересующая область: {profile.program_area}
        - Цель обучения: {profile.education_goal if hasattr(profile, 'education_goal') else 'не указана'}
        - Формат обучения: {profile.education_format if hasattr(profile, 'education_format') else 'не указан'}
        
        ВАЖНЫЕ ПРАВИЛА:
        1. НИКОГДА не задавай вопросы - этим занимается Triage Agent
        2. Твоя задача - помочь с подбором программы, основываясь на имеющейся информации
        3. Всегда обращайся к пользователю по имени
        4. Рекомендуй программы с учетом указанной области интересов
        5. Если пользователь готов к регистрации - передай управление на AuthorizationAgent
        
        Информация о программах:
        {kb_result if kb_result else 'Информация о программах будет добавлена по запросу.'}"""
        
        # Формируем запрос к GPT
        messages = [
            {"role": "system", "content": system_prompt}
        ] + context + [
            {"role": "user", "content": message}
        ]
        
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "transfer_to_authorization_agent",
                        "description": "Transfer to the Authorization Agent when the user is ready to apply.",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                }
            ]
        )
        
        # Проверяем наличие tool calls
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "transfer_to_authorization_agent":
                return ChatResponse(
                    answer="[Handoff] authorization Отлично! Теперь я передам вас специалисту по регистрации.",
                    user_profile_id=self.user_profile_id,
                    is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                    user_profile=profile
                )
        
        reply = response.choices[0].message.content.strip()
        
        # Сохраняем ответ в базу данных
        store_user_data_in_db(self.user_profile_id, {"program_decision": reply})
        
        return ChatResponse(
            answer=reply,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
            user_profile=profile
        )
