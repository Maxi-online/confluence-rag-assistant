from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import List, Any, Callable, Optional
from ai.mas.main import ChatResponse

class BaseAgent(BaseModel, ABC):
    """
    Basic class for agents.

    Provides common attributes for all agents: name, model, instructions and tools.
    The use of this class guarantees a uniform interface that facilitates implementation
    handoffs between agents in the "Routines and Handoffs" system".
    """
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful consultant in ExampleOrganization University."
    tools: List[Callable[..., Any]] = []

    @abstractmethod
    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Handle incoming message. Must be implemented by subclasses.
        
        Args:
            message: User message
            context: Optional conversation context
            
        Returns:
            ChatResponse: Agent response with user profile and state
        """
        raise NotImplementedError("Subclasses must implement handle_message")

    # New configuration for Pydantic v2:
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

# Define Agent type as BaseAgent
Agent = BaseAgent
