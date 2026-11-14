#!/usr/bin/env python3
"""
Program Decision Agent.
Finds out what educational program the user is looking for and details 
(scope, format, budget, objectives, etc.). If the user is ready to apply, report: 
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

# Establish a list of mandatory fields for the programme selection phase
REQUIRED_PROGRAM_FIELDS = [
    "user_name",
    "program_area"
]

class ProgramDecisionAgent(BaseAgent):
    user_profile_id: str
    openai_api_key: str
    knowledge_client: Any
    request: Any

    # Pydantic v2 configuration: Allow additional fields and arbitrary types
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
        Agent of an educational consultant.

        :param user_profile_id: User Identifier.
        :Param openai_api_key: API key OpenAI).
        :Para. knowledge_client: Client for the Weaviate knowledge base.
        :Param openai_base_url: Basic URL OpenAI API (by default)).
        :Param request: subject of request with user context.
        """
        super().__init__(
            name="Program Decision Agent",
            model="gpt-4o-mini",
            instructions=(
                'You are the PROGRAM DECISION AGENT—a friendly educational program consultant ExampleOrganization University.\n\n            BASIC FUNCTION:\n            - Help the user select a suitable educational program\n            - Answer questions about curricula in natural, conversational language\n            - To ask profuse, concise, clarifying questions - one at a time\n\n            IMPORTANT! COMMUNICATION TYPE:\n            - Use a friendly, informal tone.\n            - Avoid long, multiple questions.\n            - Ask only one question at a time.\n            - Answer the user\'s question first, then ask your question\n\n            EXAMPLES OF DIALOGUE:\n            \n            User: "What courses do you have?"\n            Agent: "We have a wide range of directions! For example, “Business and Management” for those who want to become a manager or develop their startup, “IT and programming” for future developers and analysts, “Design and Media” for creative professions, as well as “Marketing”, “Finances”, “Psychology” and many others. Do you have an interest in any of these areas, or do you want to elaborate on them?"\n\n            User: "I\'m interested in business and management."\n            Agent: "Great! The choice of course (business and management) is recorded. Why is business and management really bringing you to your attention?"\n\n            User: "Want to be a leader"\n            Agent: "Great choice! In Example Organization, you will be able to acquire practical management skills through real cases from top corporate managers. Our MBA and Exact MBA programs have already helped thousands of graduates take leadership positions. We\'d like to know more about our programs?"\n\n            THE CONTEXT PROCESS:\n            1. First, give a complete answer to the user\'s question\n            2. Then ask Odin a simple clarification question\n            3. If the user chose the direction, confirm the choice with one proposal\n            4. If the user is ready to apply, use transfer_to_authorization_agent\n\n            EXAMPLES OF GOOD GUIDELINES:\n            - "What direction do you want most?"\n            - "Why is this particular direction attracted to you?"\n            - "Do you plan to study alone or remotely?"\n            - "What kind of education do you have now?"\n\n            EXAMPLES OF FUTURE ISSUES (LESS ON!):\n            - "What level of education do you have, what kind of program do you want to study, when you want to start and what budget do you have?"\n\n            If a user is willing to register or apply, use transfer_to_authorization_agent.\n            '
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
        Guarantees that each element of the context list is a dictionary with 'role' and 'content' keys'.
        If the element is a string, convert it to a message from the user.
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
        Receives the current user profile and returns the list of mandatory fields,
        Programme selection that has not yet been completed.
        """
        missing = []
        # First check the user profile from the request (if available))
        request_profile = None
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            request_profile = self.request.user_profile
            logger.info(f"Request profile check: user_name={request_profile.user_name}, program_area={request_profile.program_area}")
        
        # Then check the profile from the database
        db_profile = get_user_profile(self.knowledge_client, self.user_profile_id) or {}
        
        # Checking every mandatory field
        for field in REQUIRED_PROGRAM_FIELDS:
            # The field is considered to be filled if it is in one of the profiles
            field_value = None
            
            # Check request_profile first
            if request_profile and getattr(request_profile, field, None):
                field_value = getattr(request_profile, field)
                logger.info(f"Field {field} Found in request_profile: {field_value}")
            
            # If not found in request_profile, check db_profile
            if not field_value and field in db_profile and db_profile.get(field):
                field_value = db_profile.get(field)
                logger.info(f"Field {field} Found in db_profile: {field_value}")
            
            # If the field is not found in any of the profiles, add it to the missing list
            if not field_value:
                logger.info(f"Field {field} missing in all profiles")
                missing.append(field)
        
        return missing

    async def search_knowledge_base(self, query: str) -> str:
        """
        Searches the knowledge base on request.
        Uses search_documents_with_fallback for a reliable search with automatic updating of the knowledge base.
        
        :param query: User &apos; s request for information
        :return: Search string
        """
        try:
            # Use search_documents_with_fallback for secure search
            # This method will automatically update the knowledge base through Confluence if necessary
            results = await self.knowledge_client.search_documents_with_fallback(query)
            
            if not results:
                logger.warning(f"No results were found for the request: {query}")
                # Try a search with more general key words.
                general_query = " ".join([word for word in query.split() if len(word) > 3])
                if general_query and general_query != query:
                    logger.info(f"We're going to try a search with the key words: {general_query}")
                    results = await self.knowledge_client.search_documents_with_fallback(general_query)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search of knowledge: {str(e)}")
            return ""

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Processing the user message and helping with the selection of the curriculum.
        """
        # Getting a user profile
        user_profile = get_user_profile(self.knowledge_client, self.user_profile_id)
        
        # Create an object called UserProfile
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            profile = self.request.user_profile
            logger.info(f"Program Decision Agent received a profile from request: user_name={profile.user_name}, program_area={profile.program_area}")
        else:
            profile = UserProfile(
                user_profile_id=int(self.user_profile_id),
                language=user_profile.get('language') if user_profile else None,
                user_name=user_profile.get('user_name') if user_profile else None,
                program_area=user_profile.get('program_area') if user_profile else None,
                is_modified=False
            )
            logger.info(f"Program Decision Agent created a profile from database: user_name={profile.user_name}, program_area={profile.program_area}")

        # If you don't have the necessary data to select the program, we'll transfer the control to Triage
        if not profile or not profile.user_name or not profile.program_area:
            logger.info('No necessary data for program selection')
            return ChatResponse(
                answer='[I need more information about you to select the program.',
                user_profile_id=self.user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                user_profile=profile
            )

        # Looking for up-to-date information on programmes
        kb_result = await self.search_knowledge_base(message)
        
        # Forming System Prompte for GPT
        system_prompt = f"""You're an expert in educational programs, ExampleOrganization University.
        
        User information:
        - Name: {profile.user_name}
        - Interesting area: {profile.program_area}
        - Purpose of the training: {profile.education_goal if hasattr(profile, 'education_goal') else 'Not specified'}
        - Training format: {profile.education_format if hasattr(profile, 'education_format') else 'Not specified'}
        
        MAIN REGULATIONS:
        1. Don't ever ask questions - that's what Triage Agency does
        2. Your job is to help with the selection of the program based on the available information
        3. Always contact a user by name
        4. Recommended programme, taking into account this area of interest
        5. If the user is ready to register, pass the control to AuthorizationAgent
        
        Information on programmes:
        {kb_result if kb_result else 'Information on programmes will be added upon request.'}"""
        
        # Create a GPT Request
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
        
        # Checking tool calls
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "transfer_to_authorization_agent":
                return ChatResponse(
                    answer="[Handoff] authoration. Great! Now I'm gonna hand you over to the registrar.",
                    user_profile_id=self.user_profile_id,
                    is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                    user_profile=profile
                )
        
        reply = response.choices[0].message.content.strip()
        
        # Keep the answer to the database.
        store_user_data_in_db(self.user_profile_id, {"program_decision": reply})
        
        return ChatResponse(
            answer=reply,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
            user_profile=profile
        )
