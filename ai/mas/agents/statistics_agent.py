import time
import logging
from ai.mas.utils.db import store_user_data_in_db
from typing import Any, List, Optional
from pydantic import ConfigDict

from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.main import ChatResponse

# Logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticsAgent(BaseAgent):
    """
    Agent in the collection of statistics. 
    Monitors user interaction with the system and stores statistics.
    """
    # Pydantic v2 configuration
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
        Initialization of the statistical collection agent.

        :param user_profile_id: User Identifier.
        :Param openai_api_key: API key OpenAI (not used in this agent)).
        :Param knowledge_client: Client to work with knowledge base (not used in this agent)).
        :Param request: subject of request with user context.
        :param kwargs: Additional parameters (to be ignored).
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
        Register the message depending on the sender &apos; s role.

        :param roll: Role of sender ("user" or "assistant").
        :param message: Message text (for logic).
        """
        if role == "user":
            self.stats["messages_received"] += 1
        elif role == "assistant":
            self.stats["messages_sent"] += 1
        logger.info(f"[StatisticsAgent] Recorded {role} message: {message}")

    def record_tool_call(self) -> None:
        """
        Registers a call to the tool.
        """
        self.stats["tool_calls"] += 1
        logger.info("[StatisticsAgent] Recorded a tool call.")

    def dump_stats(self) -> None:
        """
        Draws current statistics and keeps them in the database.
        """
        elapsed = time.time() - self.stats["start_time"]
        logger.info(f"[StatisticsAgent] Stats: {self.stats}, elapsed: {elapsed:.2f} seconds")
        store_user_data_in_db(self.user_profile_id, {"statistics": self.stats})
        
    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Processing the user message by collecting statistics.
        In this implementation, simply returns the statistical record message.
        
        :param message: User message.
        :param contact: History of the dialogue (not used).
        :Return: ChatResponse with validation of statistics.
        """
        self.record_message("user", message)
        return ChatResponse(
            answer='Statistics have been successfully recorded.',
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') and self.request else False,
            user_profile=self.request.user_profile if hasattr(self, 'request') and self.request else None
        )
