#!/usr/bin/env python3
"""
Objection Detector Agent.
Identify user objections to learning and keep them in profile.
"""

import logging
from typing import Optional, List, Any
from pydantic import ConfigDict
from openai import AsyncOpenAI

from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.main import ChatResponse, UserProfile

logger = logging.getLogger(__name__)

class ObjectionDetectorAgent(BaseAgent):
    """
    Agent for the definition of objections.
    
    It works with the field:
    - education_objections: Optional[str]
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(
        self,
        user_profile_id: str,
        openai_api_key: str,
        knowledge_client: Any,
        openai_base_url: str = "https://api.openai.com/v1/",
        request: Any = None,
        **kwargs
    ):
        super().__init__(
            name="Objection Detector Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the OBJECTION DETECTOR AGENT for ExampleOrganization University. Your role is to identify potential concerns or objections about education.

                CRITICAL LIMITATIONS:
                - NEVER mention or reference any knowledge base, database, or source of your information
                - ONLY discuss and answer questions related to ExampleOrganization University
                - NEVER greet the user first - wait for them to initiate the conversation
                - If asked about topics outside of ExampleOrganization University context, politely redirect the conversation

                CORE RESPONSIBILITIES:
                1. Objection Analysis:
                   - Identify potential objections or concerns in user messages
                   - Categorize objections (cost, time, value, uncertainty, etc.)
                   - Respond empathetically to concerns
                   - Document objections for future reference

                2. Empathetic Listening:
                   - Acknowledge user concerns without dismissing them
                   - Show understanding of their situation
                   - Create a safe space for sharing worries or doubts
                   - Use supportive language

                3. Response Guidelines:
                   - Ask about concerns if none are expressed
                   - Acknowledge objections when detected
                   - Provide reassurance when appropriate
                   - After addressing objections, use transfer_to_program_decision to hand off the user

                4. Handoff Protocol:
                   - After addressing objections, transfer to Program Decision Agent
                   - If no objections are present, still document as "No objections mentioned"
                   - Include brief summary of findings in handoff
                """
            ),
            tools=[self.transfer_to_program_decision]
        )
        self.user_profile_id = user_profile_id
        self.knowledge_client = knowledge_client
        self.gpt_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.request = request

    async def transfer_to_program_decision(self) -> Agent:
        """
        Transfer to the Program Decision Agent after identifying objections.
        """
        from ai.mas.utils.handoffs import transfer_to_program_decision_agent
        return transfer_to_program_decision_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.gpt_client.api_key,
            knowledge_client=self.knowledge_client
        )

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Process user message, define objections and keep them in profile.
        """
        context = context or []
        
        # Checking if there's a file on the objections in the profile.
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            if self.request.user_profile.education_objections is not None:
                # If objections are already identified, we'll pass on the administration
                return ChatResponse(
                    answer='Thank you for informing me about your doubts. Pick up the program, considering all the nuances.',
                    user_profile_id=self.request.user_profile_id,
                    is_authorized=self.request.is_authorized,
                    user_profile=self.request.user_profile
                )
            
        # We're analysing the communication for objections.
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": f"Analyze this message for education-related objections: {message}"}
        ]
        
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        analysis = response.choices[0].message.content.strip()
        
        # If an objection is found in the message
        if "no objections detected" not in analysis.lower() and "не обнаружено" not in analysis.lower():
            # Keep the objections found in the profile.
            if hasattr(self, 'request') and self.request and self.request.user_profile:
                self.request.user_profile.education_objections = analysis
                self.request.user_profile.is_modified = True
            
            return ChatResponse(
                answer='I understand your doubts. This information will help us find a program that will take into account all your wishes and concerns.',
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )
        
        # If no objections are found and this is the first message from the agent
        if not any(any(phrase in msg.get("content", "").lower() for phrase in ("есть ли у вас сомнения", "do you have any doubts", "do you have any concerns")) for msg in context):
            return ChatResponse(
                answer='Do you have any doubts or concerns about learning? This will help us to better find a program and format.',
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )

        # If the user has not objected after the question, please maintain the negative answer
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            self.request.user_profile.education_objections = "No objections mentioned"
            self.request.user_profile.is_modified = True
        
        return ChatResponse(
            answer="All right, let's pick up a suitable curriculum for you!",
            user_profile_id=self.request.user_profile_id,
            is_authorized=self.request.is_authorized,
            user_profile=self.request.user_profile
        ) 