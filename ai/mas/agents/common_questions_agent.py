#!/usr/bin/env python3
"""
Common Questions Agent.
Отвечает на общие вопросы, не связанные напрямую с выбором программы или авторизацией.
"""

import logging
import json
from openai import AsyncOpenAI
from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.utils.graph_base import WeaviateClient
from pydantic import ConfigDict
from typing import List, Optional, Any
from ai.mas.main import ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommonQuestionsAgent(BaseAgent):
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
        Агент FAQ. Отвечает кратко на вопросы о кампусе, общежитии, сроках обучения, расписании, стоимости модулей и т.д.
        Если в процессе ответа выясняется, что пользователь интересуется образовательной программой или оформлением заявки,
        агент передаёт сигнал для handoff.

        :param user_profile_id: Идентификатор пользователя.
        :param openai_api_key: API-ключ OpenAI.
        :param knowledge_client: Клиент для работы с базой знаний Weaviate.
        :param openai_base_url: Базовый URL OpenAI API.
        :param request: Объект запроса с контекстом пользователя.
        """
        super().__init__(
            name="Common Questions Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the COMMON QUESTIONS AGENT for ExampleOrganization University. Your role is to answer general questions about ExampleOrganization University admission and student life.

            CRITICAL LIMITATIONS:
            - NEVER mention or reference any knowledge base, database, or source of your information
            - NEVER greet the user first - wait for them to initiate the conversation
            - If asked about topics outside of ExampleOrganization University context, politely redirect the conversation back to university-related topics
            - Present all information as your direct knowledge, not as retrieved data

            CORE RESPONSIBILITIES:
            1. Answer Common Questions about:
               - Campus location and facilities
               - Student dormitories and accommodation
               - Academic year structure and duration
               - Scholarships and financial support (general information)
               - Student life and facilities
               - Basic admission requirements
               - General university policies

            2. Question Analysis:
               Using Chain-of-Thought reasoning, determine if the question:
               - Is within your scope (general university information)
               - Requires specific program details (transfer to Program Decision)
               - Needs authorization/registration help (transfer to Authorization)
               - Is too specific or complex for general answers

            3. Response Guidelines:
               - Provide clear, concise, and accurate information
               - Keep answers focused on ExampleOrganization University context
               - If question is too specific about a program, suggest transfer to Program Decision Agent
               - For complex financial or scholarship questions specific to programs, redirect to Program Decision Agent
               - Use professional but friendly tone
               - Avoid technical jargon unless necessary
               - ALWAYS conclude your response with a follow-up question to maintain conversation
               - Keep your answers detailed and informative, but conversational

            4. Interaction Flow:
               - Answer the current question completely
               - If user's questions move beyond general topics, initiate appropriate handoff
               - When all general questions are answered, prepare for handoff back to Orchestrator

            EXAMPLES FROM ACTUAL CONVERSATIONS:

            Example 1 - Employment guarantees:
            User: "Какие гарантии того, что я найду работу после окончания университета?"
            Agent: "В "Example Organization" мы гордимся высоким уровнем трудоустройства наших выпускников. Более 90% студентов находят работу по специальности уже в течение первого года после выпуска. Мы сотрудничаем с крупнейшими компаниями, такими как Яндекс, Mail.ru Group и другие. Наши программы включают обязательную практику и стажировки, которые помогают вам получить реальный опыт работы. Хотели бы узнать больше о наших партнерах по трудоустройству?"

            Example 2 - Education cost:
            User: "Какая стоимость обучения?"
            Agent: "Стоимость обучения зависит от выбранной программы и формата обучения. Однако мы предлагаем различные варианты оплаты, включая образовательные кредиты, гранты и рассрочки. Например, многие наши студенты пользуются программой "Студенческий кредит", которая позволяет оплачивать обучение постепенно без первоначального взноса. Хотели бы узнать подробнее о финансовых возможностях?"

            Example 3 - Program transfer:
            User: "Могу ли я перейти на другую программу, если изменятся мои интересы?"
            Agent: "Конечно! В "Example Organization" мы понимаем, что интересы могут меняться, и поэтому предлагаем возможность перехода на другие программы без потери времени и денег. Наши консультанты помогут вам выбрать наиболее подходящий вариант. Хотели бы обсудить этот вопрос с нашим менеджером?"

            Example 4 - Open days:
            User: "Как часто проводятся дни открытых дверей?"
            Agent: "Мы проводим дни открытых дверей ежемесячно, как в офлайн-формате, так и онлайн. Это отличная возможность лично пообщаться с преподавателями, узнать больше о программах и задать все интересующие вопросы. Ближайший день открытых дверей состоится [дата]. Хотели бы зарегистрироваться?"

            RESPONSE FORMAT:
            1. For General Questions:
               - Provide direct, informative answer
               - Include relevant details but stay concise
               - Maintain professional tone
               - Always end with a follow-up question

            2. For Questions Outside Scope:
               - Briefly acknowledge the question
               - Use transfer_to_program_decision function to transfer to Program Decision Agent
               - Example: "For detailed program information, I'll connect you with our Program Decision specialist"

            3. For Complex Questions:
               - Provide general information if possible
               - Explain why more specific assistance is needed
               - Initiate appropriate handoff

            Remember: Always present information as your direct knowledge, never reference any information sources or databases.
            """
            ),
            tools=[self.transfer_to_program_decision, self.transfer_to_authorization]
        )
        self.user_profile_id = user_profile_id
        self.knowledge_client = knowledge_client
        self.gpt_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        self.request = request

    async def transfer_to_program_decision(self) -> Agent:
        """
        Call this function when the user needs specific information about educational programs.
        """
        from ai.mas.utils.handoffs import transfer_to_program_decision_agent
        return transfer_to_program_decision_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.gpt_client.api_key,
            knowledge_client=self.knowledge_client
        )
        
    async def transfer_to_authorization(self) -> Agent:
        """
        Call this function when the user wants to register or needs authentication help.
        """
        from ai.mas.utils.handoffs import transfer_to_authorization_agent
        return transfer_to_authorization_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.gpt_client.api_key
        )

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Обрабатывает входящее сообщение: сначала ищет релевантную информацию в базе знаний,
        затем отправляет комбинированный запрос в OpenAI API и возвращает полученный ответ.
        Если в ответе содержатся сигналы handoff, возвращает их для переключения управления.

        :param message: Сообщение пользователя.
        :param context: История диалога (если имеется).
        :return: Ответ агента или сигнал handoff.
        """
        context = context or []

        # Проверяем, авторизован ли пользователь
        is_authorized = False
        if hasattr(self, 'request') and self.request and hasattr(self.request, 'is_authorized'):
            is_authorized = self.request.is_authorized
        
        # Сначала ищем в FAQ с учетом статуса авторизации
        faq_results = self.knowledge_client.search_faq(message, is_authorized=is_authorized, limit=2)
        
        # Если FAQ не дал результатов, ищем в базе знаний Weaviate
        if not faq_results:
            kb_result = self.knowledge_client.search_local_knowledge(message)
        else:
            kb_result = faq_results
        
        # Определяем язык пользователя
        user_language = "ru"  # по умолчанию русский
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            user_language = self.request.user_profile.language or "ru"
        
        # Адаптируем инструкции в зависимости от языка пользователя
        instructions = self.instructions
        if user_language == "ru":
            instructions += "\nIMPORTANT: ALWAYS respond in Russian language regardless of the language of the user's question."
        
        # Формируем полный запрос с учетом полученного контекста
        messages_payload = (
            [{"role": "system", "content": instructions}] +
            context +
            [{"role": "user", "content": message}]
        )
        
        # Добавляем результаты поиска, если они есть
        if kb_result:
            messages_payload.append({"role": "system", "content": f"Relevant information found in knowledge base: {kb_result}"})

        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages_payload,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "transfer_to_program_decision",
                        "description": "Call this function when the user needs specific information about educational programs.",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "transfer_to_authorization",
                        "description": "Call this function when the user wants to register or needs authentication help.",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                }
            ]
        )
        
        if response.choices[0].message.tool_calls:
            # Handle tool calls with proper description
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "transfer_to_program_decision":
                return ChatResponse(
                    answer="[Handoff] program_decision Для получения подробной информации о программах обучения, я передам вас нашему специалисту по выбору программ.",
                    user_profile_id=self.user_profile_id,
                    is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                    user_profile=self.request.user_profile if hasattr(self, 'request') else None
                )
            elif tool_call.function.name == "transfer_to_authorization":
                return ChatResponse(
                    answer="[Handoff] authorization Для регистрации или помощи с авторизацией, я передам вас нашему специалисту по этим вопросам.",
                    user_profile_id=self.user_profile_id,
                    is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                    user_profile=self.request.user_profile if hasattr(self, 'request') else None
                )
        else:
            # Return the standard response if no tool was called
            reply = response.choices[0].message.content.strip()
            # Убираем возможные JSON-артефакты из ответа
            if "tool_uses" in reply:
                reply = reply.split("{")[0].strip()
            logger.info(f"Common Questions Agent для пользователя {self.user_profile_id} ответил: {reply}")

        return ChatResponse(
            answer=reply,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
            user_profile=self.request.user_profile if hasattr(self, 'request') else None
        )
