#!/usr/bin/env python3
"""
Основной файл для запуска системы GPT-агентов.
Функция принимает запросы и на основе анализа передаёт управление нужному агенту.
"""

import sys
import asyncio
import json
import logging
import traceback
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ai.mas.utils.graph_base import WeaviateClient
from ai.mas.utils.config import (
    USER_ID,
    OPENAI_API_KEY,
    WEAVIATE_URL,
    WEAVIATE_API_KEY,
    EXPECTED_DIM,
    CONFLUENCE_BASE_URL,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_PAGE_ID,
    MAX_CHUNK_SIZE,
    MIN_SCORE,
    HYBRID_ALPHA,
    CLASS_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class UserProfile(BaseModel):
    """Модель профиля пользователя"""
    user_profile_id: int = Field(..., description="Уникальный идентификатор профиля пользователя")
    language: Optional[str] = Field(None, description="Язык общения с ботом")
    user_name: Optional[str] = Field(None, description="Имя, указанное им для обращения")
    program_area: Optional[str] = Field(None, description="Область программы")
    education_goal: Optional[str] = Field(None, description="Цель обучения")
    education_objections: Optional[str] = Field(None, description="Возражения к обучению")
    age: Optional[int] = Field(None, description="Возраст пользователя")
    heard_about: Optional[str] = Field(None, description="Откуда узнал о нас")
    after_study_goal: Optional[str] = Field(None, description="Цель после обучения")
    education_format: Optional[str] = Field(None, description="Формат обучения")
    work_experience: Optional[str] = Field(None, description="Опыт работы")
    additional_interests: Optional[str] = Field(None, description="Дополнительные интересы")
    selected_university: Optional[str] = Field(None, description="Выбранное уч. заведение")
    is_modified: bool = Field(False, description="Флаг модификации профиля")

class ChatRequest(BaseModel):
    """Модель запроса чата"""
    message: str = Field(..., description="Сообщение пользователя")
    user_profile_id: str = Field(default=USER_ID, description="Идентификатор пользователя")
    openai_api_key: str = Field(default=OPENAI_API_KEY, description="API-ключ OpenAI")
    weaviate_url: str = Field(default=WEAVIATE_URL, description="URL Weaviate сервера")
    weaviate_api_key: str = Field(default=WEAVIATE_API_KEY, description="API-ключ Weaviate")
    expected_dim: int = Field(default=EXPECTED_DIM, description="Ожидаемое количество измерений в векторизации")
    confluence_base_url: str = Field(default=CONFLUENCE_BASE_URL, description="Базовый URL для Confluence")
    confluence_api_token: str = Field(default=CONFLUENCE_API_TOKEN, description="API-токен Confluence")
    confluence_page_id: int = Field(default=CONFLUENCE_PAGE_ID, description="ID страницы Confluence")
    max_chunk_size: int = Field(default=MAX_CHUNK_SIZE, description="Максимальный размер чанка для разбивки текста")
    min_score: float = Field(default=MIN_SCORE, description="Минимальный порог релевантности")
    hybrid_alpha: float = Field(default=HYBRID_ALPHA, description="Параметр hybrid_alpha для поиска")
    is_authorized: bool = Field(default=False, description="Флаг авторизации пользователя")
    user_profile: Optional[UserProfile] = Field(default=None, description="Профиль пользователя")

class ChatResponse(BaseModel):
    """Модель ответа чата"""
    answer: str = Field(..., description="Ответ системы GPT-агентов")
    user_profile_id: str = Field(..., description="Идентификатор пользователя")
    is_authorized: bool = Field(..., description="Флаг авторизации пользователя")
    user_profile: Optional[UserProfile] = Field(default=None, description="Профиль пользователя")

async def process_message(request: ChatRequest) -> ChatResponse:
    """
    Обрабатывает входящее сообщение и возвращает ответ.
    
    Args:
        request (ChatRequest): Запрос с сообщением и конфигурацией
        
    Returns:
        ChatResponse: Ответ системы с сохранением всех полей запроса
    """
    # Импортируем OrchestratorAgent внутри функции, чтобы избежать циклической зависимости
    from ai.mas.agents.orchestrator_agent import OrchestratorAgent
    
    user_input = request.message.strip()
    if not user_input:
        return ChatResponse(
            answer="Сообщение не должно быть пустым",
            user_profile_id=request.user_profile_id,
            is_authorized=request.is_authorized,
            user_profile=request.user_profile
        )
    
    try:
        # Инициализируем клиента для работы с Weaviate
        with WeaviateClient(
            openai_api_key=request.openai_api_key,
            weaviate_url=request.weaviate_url,
            weaviate_api_key=request.weaviate_api_key,
            class_name=CLASS_NAME,
            expected_dim=request.expected_dim,
            confluence_base_url=request.confluence_base_url,
            confluence_api_token=request.confluence_api_token,
            confluence_page_id=request.confluence_page_id,
            max_chunk_size=request.max_chunk_size,
            min_score=request.min_score,
            hybrid_alpha=request.hybrid_alpha
        ) as knowledge_client:
            # Создаем экземпляр OrchestratorAgent, передавая весь объект request
            orchestrator = OrchestratorAgent(
                request=request,
                knowledge_client=knowledge_client
            )

            # Получаем ответ от оркестратора
            response = await orchestrator.handle_message(user_input)
            
            # Проверяем, что response является ChatResponse
            if not isinstance(response, ChatResponse):
                raise ValueError(f"Expected ChatResponse, got {type(response)}")
            
            # Логируем длину ответа
            logging.info(f"[Main] Successfully processed request. Answer length: {len(response.answer)}")
            
            # Возвращаем ответ в формате ChatResponse
            return response

    except Exception as e:
        logging.error(f"Error in process_message: {e}\n{traceback.format_exc()}")
        return ChatResponse(
            answer="Произошла ошибка при обработке запроса",
            user_profile_id=request.user_profile_id,
            is_authorized=request.is_authorized,
            user_profile=request.user_profile
        )
