import logging
from openai import AsyncOpenAI
from pydantic import ConfigDict
from typing import Any, List, Optional

from ai.mas.utils.db import update_user_profile, update_language_in_context, get_user_profile
from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.main import ChatResponse, UserProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDetectionAgent(BaseAgent):
    """
    Агент определения языка.
    
    Логика:
      1. При первом запуске (если в профиле отсутствует язык)
         – агент задаёт вопрос: "На каком языке вам удобно общаться?"
      2. При последующих вызовах агент не задаёт вопрос повторно и просто передаёт сообщение дальше.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(
        self,
        knowledge_client: Any,
        user_profile_id: str,
        openai_api_key: str,
        openai_base_url: str = "https://api.openai.com/v1/",
        request: Any = None,
        **kwargs
    ):
        super().__init__(
            name="Language Detection Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the LANGUAGE DETECTION AGENT. Your mission is to determine the language of the user's input quickly and accurately.

            Instructions:
            1. Analyze the input text using internal Chain‑of‑Thought reasoning.
            2. NEVER output the detected language to the user directly. Instead, save it internally.
            3. If multiple languages are detected, prioritize the main language.
            4. Handle the language detection silently in the background.
            5. Do not include any language detection details in your output to the user.
            6. After determining the language, use the transfer_to_orchestrator function to continue the conversation.
            7. Your response should be minimal and focused on continuing the conversation flow.

            Output: A simple message to handoff control back to the orchestrator without mentioning language detection.
            """
            ),
            tools=[self.transfer_to_orchestrator],
            **kwargs
        )
        self.user_profile_id = user_profile_id
        self.knowledge_client = knowledge_client
        self.gpt_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.request = request

    async def transfer_to_orchestrator(self, message: str = "") -> Agent:
        """
        Transfer control back to the Orchestrator with the user's original message.
        
        Args:
            message: Original user message to pass to the Orchestrator
        """
        from ai.mas.agents.orchestrator_agent import OrchestratorAgent
        logger.info(f"[LanguageDetectionAgent] Transferring to Orchestrator with message: {message}")
        return None

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Обрабатывает сообщение пользователя и определяет язык общения.
        
        Логика:
        1. Если язык уже определен, просто передает управление дальше.
        2. Если в сообщении содержится указание на язык, сохраняет его и передает управление дальше.
        3. Иначе, задает пользователю вопрос о предпочитаемом языке.
        """
        logger.info(f"\n{'='*80}\nЯЗЫКОВОЙ АГЕНТ НАЧАЛ ОБРАБОТКУ")
        logger.info(f"Входящее сообщение: {message}")
        
        # Проверяем профиль из request
        if not hasattr(self, 'request') or not self.request or not self.request.user_profile:
            logger.error("Ошибка: отсутствует request или профиль пользователя")
            return ChatResponse(
                answer="Извините, произошла ошибка при определении языка.",
                user_profile_id=self.user_profile_id,
                is_authorized=False,
                user_profile=None
            )

        logger.info("\nТЕКУЩЕЕ СОСТОЯНИЕ ПРОФИЛЯ:")
        logger.info(f"user_profile_id: {self.request.user_profile.user_profile_id}")
        logger.info(f"language: {self.request.user_profile.language}")
        logger.info(f"is_modified: {self.request.user_profile.is_modified}")

        # Если язык уже определен, передаем исходное сообщение дальше через handoff
        if self.request.user_profile.language is not None:
            logger.info(f"[LanguageDetectionAgent] Язык уже определен ({self.request.user_profile.language}). Передаю управление оркестратору.")
            return ChatResponse(
                answer=f"[Handoff] orchestrator {message}",
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )

        # Проверяем, является ли сообщение ответом на вопрос о языке
        if any(word.lower() in message.lower() for word in ["русский", "рус", "русск", "ру"]):
            language = "ru"
            logger.info(f"Определен язык: {language}")
            
            # Сохраняем язык в профиле
            logger.info("\nОБНОВЛЕНИЕ ПРОФИЛЯ:")
            logger.info(f"Старое значение языка: {self.request.user_profile.language}")
            logger.info(f"Старое значение is_modified: {self.request.user_profile.is_modified}")
            
            self.request.user_profile.language = language
            self.request.user_profile.is_modified = True
            
            logger.info(f"Новое значение языка: {self.request.user_profile.language}")
            logger.info(f"Новое значение is_modified: {self.request.user_profile.is_modified}")
            
            # Проверяем, есть ли имя пользователя в профиле
            if self.request.user_profile.user_name is None:
                # Если нет имени, передаем управление на TriageAgent
                logger.info(f"\nИМЯ ПОЛЬЗОВАТЕЛЯ НЕ ОПРЕДЕЛЕНО. ПЕРЕДАЧА УПРАВЛЕНИЯ TRIAGE AGENT")
                return ChatResponse(
                    answer=f"[Handoff] triage",
                    user_profile_id=self.request.user_profile_id,
                    is_authorized=self.request.is_authorized,
                    user_profile=self.request.user_profile
                )
            
            # Если имя есть, передаем управление оркестратору
            logger.info(f"\nПЕРЕДАЧА УПРАВЛЕНИЯ ОРКЕСТРАТОРУ")
            
            return ChatResponse(
                answer=f"[Handoff] orchestrator",
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )
        elif any(word.lower() in message.lower() for word in ["english", "eng", "английский", "англ"]):
            language = "en"
            logger.info(f"Определен язык: {language}")
            
            # Сохраняем язык в профиле
            logger.info("\nОБНОВЛЕНИЕ ПРОФИЛЯ:")
            logger.info(f"Старое значение языка: {self.request.user_profile.language}")
            logger.info(f"Старое значение is_modified: {self.request.user_profile.is_modified}")
            
            self.request.user_profile.language = language
            self.request.user_profile.is_modified = True
            
            logger.info(f"Новое значение языка: {self.request.user_profile.language}")
            logger.info(f"Новое значение is_modified: {self.request.user_profile.is_modified}")
            
            # Проверяем, есть ли имя пользователя в профиле
            if self.request.user_profile.user_name is None:
                # Если нет имени, передаем управление на TriageAgent
                logger.info(f"\nИМЯ ПОЛЬЗОВАТЕЛЯ НЕ ОПРЕДЕЛЕНО. ПЕРЕДАЧА УПРАВЛЕНИЯ TRIAGE AGENT")
                return ChatResponse(
                    answer=f"[Handoff] triage",
                    user_profile_id=self.request.user_profile_id,
                    is_authorized=self.request.is_authorized,
                    user_profile=self.request.user_profile
                )
            
            # Если имя есть, передаем управление оркестратору
            logger.info(f"\nПЕРЕДАЧА УПРАВЛЕНИЯ ОРКЕСТРАТОРУ")
            
            return ChatResponse(
                answer=f"[Handoff] orchestrator",
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )
        else:
            # Пытаемся автоматически определить язык сообщения если это первое общение
            if len(message) > 10:  # Если сообщение достаточно длинное для анализа
                # Простой анализ: если больше кириллицы, то русский, иначе английский
                cyrillic_count = sum(1 for char in message if 'а' <= char.lower() <= 'я')
                latin_count = sum(1 for char in message if 'a' <= char.lower() <= 'z')
                
                if cyrillic_count > latin_count:
                    self.request.user_profile.language = "ru"
                    self.request.user_profile.is_modified = True
                    logger.info(f"Автоматически определен язык: ru")
                    return ChatResponse(
                        answer=f"[Handoff] orchestrator {message}",
                        user_profile_id=self.request.user_profile_id,
                        is_authorized=self.request.is_authorized,
                        user_profile=self.request.user_profile
                    )
            
            # Задаем вопрос о предпочитаемом языке
            logger.info("[LanguageDetectionAgent] Задаю вопрос о языке")
            return ChatResponse(
                answer="На каком языке вам удобнее общаться?",
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )
