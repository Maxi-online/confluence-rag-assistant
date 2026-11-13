#!/usr/bin/env python3
"""
Summary Agent - агент для создания контекстного саммари диалога.

Особенности:
- Создает краткое саммари последних N сообщений
- Выделяет ключевой контекст для других агентов
- Помогает агентам понимать предысторию диалога
- Обеспечивает связность и последовательность ответов
"""

from typing import List, Optional, Dict, Any
import logging

from ai.mas.agents.base_agent import BaseAgent, Agent
from openai import AsyncOpenAI
from pydantic import ConfigDict
from ai.mas.main import ChatResponse

logger = logging.getLogger(__name__)

class SummaryAgent(BaseAgent):
    """
    Агент для создания контекстного саммари диалога.
    Помогает поддерживать связность и последовательность беседы.
    """
    # Конфигурация для Pydantic v2
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    def __init__(
        self,
        user_profile_id: str,
        openai_api_key: str,
        request: Any = None,
        **kwargs
    ):
        super().__init__(
            name="Summary Agent",
            model="gpt-4o-mini",
            instructions="""Вы - специализированный агент по созданию контекстного саммари диалога.

            ОСНОВНЫЕ ЗАДАЧИ:
            1. Создание краткого саммари последних сообщений
            2. Выделение ключевого контекста для других агентов
            3. Обеспечение связности диалога
            4. Поддержка контекстной осведомленности агентов

            ПРАВИЛА САММАРИЗАЦИИ:
            1. Фокус на контексте:
               - Основная тема обсуждения
               - Ключевые вопросы пользователя
               - Важные детали из ответов системы
               - Нерешенные вопросы
               
            2. Стиль изложения:
               - Максимально кратко (1-2 предложения)
               - Только существенная информация
               - Акцент на текущем контексте
               - Без лишних деталей

            3. Особые указания:
               - Сохранять только релевантную информацию
               - Выделять ключевые намерения пользователя
               - Отмечать важные моменты для следующего агента
               - Обеспечивать плавный переход между агентами
            """,
            tools=[],
            user_profile_id=user_profile_id,
            openai_api_key=openai_api_key,
            **kwargs
        )
        self.user_profile_id = user_profile_id
        self.gpt_client = AsyncOpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1/")
        self.request = request

    async def create_context_summary(self, messages: List[dict], max_messages: int = 5) -> str:
        """
        Создает краткое саммари последних сообщений для контекста.
        
        :param messages: История сообщений
        :param max_messages: Максимальное количество последних сообщений для обработки
        :return: Краткое контекстное саммари
        """
        if not messages:
            return ""

        # Берем последние max_messages сообщений
        recent_messages = messages[-max_messages:]

        # Подготовка промпта для GPT
        system_prompt = """Создайте ОЧЕНЬ краткое (1-2 предложения) саммари последних сообщений диалога.
        Сфокусируйтесь на:
        - Основной теме обсуждения
        - Ключевых вопросах пользователя
        - Важных деталях из ответов системы
        
        Саммари должно помочь следующему агенту быстро понять контекст беседы."""

        # Подготовка контекста для GPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Создайте краткое контекстное саммари для последних сообщений:\n{str(recent_messages)}"}
        ]

        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content.strip()

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Создает контекстное саммари для других агентов.

        :param message: Текущее сообщение
        :param context: История диалога
        :return: ChatResponse с контекстным саммари
        """
        if not context:
            return ChatResponse(
                answer="",
                user_profile_id=self.user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self, 'request') and self.request else False,
                user_profile=self.request.user_profile if hasattr(self, 'request') and self.request else None
            )

        # Создаем контекстное саммари
        summary = await self.create_context_summary(context)
        return ChatResponse(
            answer=summary,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
            user_profile=self.request.user_profile if hasattr(self, 'request') else None
        ) 