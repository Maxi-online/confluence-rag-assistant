#!/usr/bin/env python3
"""
Summary Agent is an agent for creating a context-specific dialogue samari.

Features:
- Creates short samari of the last N messages.
- Sets out a key context for other agents.
- Helps agents understand the background of the dialogue.
- Ensures coherence and consistency of responses
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
    Agent to create a context-specific samari dialogue.
    Helps maintain communication and consistency.
    """
    # Pydantic v2 configuration
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
            instructions="You're a specialized agent for creating a context-based samari dialogue.\n\n            Main challenges:\n            1. Creation of a short samari of recent messages\n            2. Selection of a key context for other agents\n            3. Ensuring the coherence of the dialogue\n            4. Support for contextual awareness of agents\n\n            RULES OF SEMARIZATION:\n            1. Focus in context:\n               - Main theme of the discussion\n               - Key user questions\n               - Important details from the system's responses\n               - Pending issues\n               \n            2. Presentation style:\n               - Maximum short (1-2 sentences)\n               - Only essential information\n               - Focus on the current context\n               - No more details.\n\n            3. Special instructions:\n               - Save only relevant information\n               - To highlight the user &apos; s key intentions\n               - To mark important moments for the next agent.\n               - Ensure a smooth transition between agents\n            ",
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
        Creates short samari of recent messages for context.
        
        :param messages: Message history
        :Param max_messages: Maximum number of last messages for processing
        :return: Short context samari
        """
        if not messages:
            return ""

        # Take the last max_messages messages
        recent_messages = messages[-max_messages:]

        # Preparation of a Prompt for GPT
        system_prompt = "Create a short (1-2 sentences) of the last messages of the dialogue.\n        Focus on:\n        - Main theme of the discussion\n        - Key user questions\n        - Important details from the system's responses\n        \n        Sammari should help the next agent quickly understand the context of the conversation."

        # Preparation of Context for GPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a short context samari for the last messages:\n{str(recent_messages)}"}
        ]

        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content.strip()

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Creates context samari for other agents.

        :param message: Current message
        :param contact: History of the dialogue
        :return: ChatResponse with context samari
        """
        if not context:
            return ChatResponse(
                answer="",
                user_profile_id=self.user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self, 'request') and self.request else False,
                user_profile=self.request.user_profile if hasattr(self, 'request') and self.request else None
            )

        # Create a context samari
        summary = await self.create_context_summary(context)
        return ChatResponse(
            answer=summary,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
            user_profile=self.request.user_profile if hasattr(self, 'request') else None
        ) 