import time
import logging
from ai.mas.utils.db import store_user_data_in_db
from typing import Any, List, Optional
from pydantic import ConfigDict

from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.main import ChatResponse

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticsAgent(BaseAgent):
    """
    Агент сбора статистики. 
    Отслеживает взаимодействия пользователя с системой и сохраняет статистические данные.
    """
    # Конфигурация для Pydantic v2
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    def __init__(
        self,
        user_profile_id: str,
        openai_api_key: str = None,
        knowledge_client: Any = None,
        request: Any = None,
        **kwargs
    ) -> None:
        """
        Инициализация агента сбора статистики.

        :param user_profile_id: Идентификатор пользователя.
        :param openai_api_key: API-ключ OpenAI (не используется в данном агенте).
        :param knowledge_client: Клиент для работы с базой знаний (не используется в данном агенте).
        :param request: Объект запроса с контекстом пользователя.
        :param kwargs: Дополнительные параметры (игнорируются).
        """
        super().__init__(
            name="Statistics Agent",
            model="gpt-4o-mini",
            instructions="You are a Statistics Agent that collects usage data.",
            tools=[]
        )
        self.user_profile_id = user_profile_id
        self.stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "tool_calls": 0,
            "start_time": time.time(),
        }
        self.request = request

    def record_message(self, role: str, message: str) -> None:
        """
        Регистрирует сообщение в зависимости от роли отправителя.

        :param role: Роль отправителя ("user" или "assistant").
        :param message: Текст сообщения (для логирования).
        """
        if role == "user":
            self.stats["messages_received"] += 1
        elif role == "assistant":
            self.stats["messages_sent"] += 1
        logger.info(f"[StatisticsAgent] Recorded {role} message: {message}")

    def record_tool_call(self) -> None:
        """
        Регистрирует вызов инструмента.
        """
        self.stats["tool_calls"] += 1
        logger.info("[StatisticsAgent] Recorded a tool call.")

    def dump_stats(self) -> None:
        """
        Выводит текущую статистику и сохраняет её в базе данных.
        """
        elapsed = time.time() - self.stats["start_time"]
        logger.info(f"[StatisticsAgent] Stats: {self.stats}, elapsed: {elapsed:.2f} seconds")
        store_user_data_in_db(self.user_profile_id, {"statistics": self.stats})
        
    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Обрабатывает сообщение пользователя, собирая статистику.
        В данной реализации просто возвращает сообщение о записи статистики.
        
        :param message: Сообщение пользователя.
        :param context: История диалога (не используется).
        :return: ChatResponse с подтверждением записи статистики.
        """
        self.record_message("user", message)
        return ChatResponse(
            answer="Статистика успешно записана.",
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') and self.request else False,
            user_profile=self.request.user_profile if hasattr(self, 'request') and self.request else None
        )
