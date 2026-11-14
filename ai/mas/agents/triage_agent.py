"""
Triage Agent.
Collects basic user information: name, source, etc.
Once the response is received, the agent updates the user profile and, if any fields are not yet filled in,
Ask for clarification.
"""

import json
import logging
from json import JSONDecodeError
from openai import AsyncOpenAI
from ai.mas.utils.db import store_user_data_in_db, get_user_profile, update_user_profile
from ai.mas.agents.base_agent import BaseAgent, Agent
from typing import List, Optional, Any, Dict
from pydantic import ConfigDict, PrivateAttr
from ai.mas.main import ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of mandatory fields to be filled out during the Triage phase
REQUIRED_FIELDS = ["user_name", "how_heard_about_us"]

# Fields that the user has already ignored
IGNORED_FIELDS = set()

class TriageAgent(BaseAgent):
    # Use a new Pydantic v2 configuration with additional field resolution
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    
    # Private attributes for the storage of operational data (not validated))
    _user_profile_id: str = PrivateAttr()
    _knowledge_client: Any = PrivateAttr()

    def __init__(
        self,
        *,
        user_profile_id: str,
        openai_api_key: str,
        knowledge_client: Any,
        openai_base_url: str = "https://api.openai.com/v1/",
        **kwargs
    ):
        """
        Agent in collecting basic user information.

        :param user_profile_id: User Identifier.
        :param openai_api_key: API key OpenAI.
        :Param knowledge_client: Knowledge client. This field is mandatory!
        :Param openai_base_url: Basic URL OpenAI API (by default)).
        """
        # Initiate OpenAI client to work with GPT
        gpt_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        # Call a basic designer, giving only those fields that participate in the validation
        super().__init__(
            name="Triage Agent",
            model="gpt-4o-mini",
            instructions=(
                'You are the TRIAGE AGENT. Your role is to gather basic information about the user and determine their intentions.\n\n            CRITICAL LIMITATIONS:\n            - NEVER mention or reference any knowledge base, database, or source of your information\n            - NEVER greet the user first - wait for them to initiate the conversation\n            - If asked about topics outside of ExampleOrganization University admission, politely redirect the conversation back to admission-related topics\n            - NEVER use technical terms or field names like "user_name" or "education_goal"\n            - NEVER show any internal structure, database fields, or system information to users\n            - ABSOLUTELY NEVER show progress or status lists to users\n\n            CORE RESPONSIBILITIES:\n            1. Information Gathering (PRIORITY ORDER):\n               - FIRST: Ask for the user\'s name (this is critical for personalization)\n               - SECOND: Ask how they learned about ExampleOrganization University\n               - THIRD: Ask about their program interests and education goals\n               - Save all collected information\n               - Always maintain the context of ExampleOrganization University admission process\n\n            2. Interaction Style:\n               - Use friendly, warm, informal and conversational Russian language\n               - Talk as a friendly person, not as a technical system\n               - Avoid technical terminology and system references completely\n               - Ask only ONE question at a time\n               - Make questions short, personal and focused\n               - After getting user\'s name, always use it in your responses \n               - Use natural, casual language as in human conversation\n               - Maintain a warm, helpful tone\n               - NEVER show progress tracking information\n               - NEVER mention data processing or database operations\n\n            3. Example Dialogue:\n               Agent: "What\'s your name? It\'ll help me make our communication more personal."\n               User: "Alexei"\n               Agent: "Nice to meet you, Alexei! Please tell me how you learned about our university?"\n\n            4. Intent Analysis:\n               Using Chain-of-Thought reasoning, determine the user\'s primary intention:\n               - "CommonQuestion" – general questions about ExampleOrganization University, programs, processes\n               - "ProgramDecision" – requests for help in choosing a ExampleOrganization educational program\n               - "Authorization" – questions about authorization or registration\n               - "LeadQualification" – evaluation of potential ExampleOrganization student\n               - "Other" – if none of the above applies\n\n            3. Interaction Rules:\n               - Always try to answer user questions if you can\n               - Maintain polite and professional but warm dialogue\n               - If the question is outside your expertise, explain that you\'ll transfer to a more specialized agent\n               - Use natural, conversational communication style\n               - Present all information as your direct knowledge, not as retrieved data\n               - Keep all responses focused on ExampleOrganization University context\n               - Be friendly, supportive and encouraging\n\n            4. Agent Handoffs:\n               After determining intentions and collecting basic information, transfer the user to the appropriate agent:\n               - For general questions -> CommonQuestions\n               - For program selection -> use transfer_to_program_decision function\n               - For authorization issues -> Authorization\n\n            RESPONSE FORMAT:\n            1. If user name is not known:\n               - First quaistion should ALWAYS be: "What\'s your name? This will help us to personalize our communication."\n               - Save name upon receiving it\n            \n            2. If agent handoff is needed:\n               - Answer the current question if possible\n               - Use the appropriate transfer function\n\n            3. If both name and how_heard_about_us are known:\n               - Transfer to appropriate agent based on their intent\n               - Use transfer_to_program_decision in most cases\n            '
            ),
            tools=[self.transfer_to_program_decision],
            **kwargs
        )
        # Set up private attributes.
        self._user_profile_id = user_profile_id
        self._knowledge_client = knowledge_client
        self.gpt_client = gpt_client

    def get_missing_fields(self) -> list:
        """
        Receives the current user profile and returns the list of mandatory fields,
        which have not yet been filled and have not been ignored by the user.
        """
        profile = get_user_profile(self._knowledge_client, self._user_profile_id) or {}
        # Delete fields that the user has already ignored
        missing = [field for field in REQUIRED_FIELDS if not profile.get(field) and field not in IGNORED_FIELDS]
        return missing

    async def analyze_with_gpt(self, message: str, context: Optional[List[dict]] = None) -> Dict[str, Any]:
        """
        Analyses the user message using GPT to identify the name and program area.
        GPT decides when to extract and store information.
        
        :param message: User message
        :param contact: History of the dialogue (if any)
        :return: Vocabulary with information to preserve
        """
        context = context or []
        
        # Defining the assistant's last question (if any)
        last_assistant_message = ""
        for msg in reversed(context):
            if msg.get("role") == "assistant":
                last_assistant_message = msg.get("content", "")
                break
        
        # Simple and Direct Prompte for GPT
        analyze_prompt = f"""
        Analyze the user message and determine whether it contains a username or a program area.
        
        The assistant's last question:{last_assistant_message}\"
        User message: \"{message}\"
        
        If you are sure that the message contains a user name or a program area, extract them and be true in JSON format:
        {{
            \"user_name\": \"User name if contained in the message\",
            \"program_area\": \"Programme area, if contained in a message\"
        }}
        
        If no name or area is available, indicate null for the field. The answer is only to JSON.
        """
        
        analyze_messages = [
            {"role": "system", "content": 'You are an analyst who extract only the user’s name and the program area of interest from the messages. Determine this directly and independently without additional criteria.'},
            {"role": "user", "content": analyze_prompt}
        ]
        
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=analyze_messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"GPT message analysis '{message}': {result_text}")
        
        result = json.loads(result_text)
        # Clear empty values and null
        return {k: v for k, v in result.items() if v and v != "null" and v != "Null"}

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Processing the user message. 
        GPT analyses the message, context and profile, determines the next step of the dialogue.
        
        :param message: User message.
        :param contact: History of the dialogue (if any).
        The object of ChatResponse with the agent's response.
        """
        context = context or []
        
        # Get current user profile
        user_profile = self.request.user_profile if hasattr(self, 'request') and self.request else None
        
        # We analyze the user’s message to extract data
        extracted_data = await self.analyze_with_gpt(message, context)
        logger.info(f"The extracted data: {extracted_data}")
        
        # If new data are found, update profile
        if extracted_data and user_profile:
            if "user_name" in extracted_data and not user_profile.user_name:
                user_profile.user_name = extracted_data["user_name"]
                user_profile.is_modified = True
                logger.info(f"Up-to-date username: {user_profile.user_name}")
            
            if "program_area" in extracted_data and not user_profile.program_area:
                user_profile.program_area = extracted_data["program_area"]
                user_profile.is_modified = True
                logger.info(f"Programme area updated: {user_profile.program_area}")
            
            if "education_goal" in extracted_data and not user_profile.education_goal:
                user_profile.education_goal = extracted_data["education_goal"]
                user_profile.is_modified = True
                logger.info(f"The purpose of the training has been updated: {user_profile.education_goal}")
        
        # Check that all necessary data are available in the profile
        if user_profile and user_profile.user_name and user_profile.program_area:
            logger.info('All necessary data collected, transfer to ProgramDecisionAgent')
            return ChatResponse(
                answer="[Handoff] program_decision Excellent! Now I'm gonna hand you over to the training specialist.",
                user_profile_id=self._user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self.request, 'is_authorized') else False,
                user_profile=user_profile
            )
        
        # Form a system propt for GPT, taking into account current status
        system_prompt = f"""You're a friendly consultant to ExampleOrganization University.
        
        Current status of user profile:
        - Name: {user_profile.user_name if user_profile and user_profile.user_name else 'not specified'}
        - Interesting area: {user_profile.program_area if user_profile and user_profile.program_area else 'Not specified'}
        - Purpose of the training: {user_profile.education_goal if user_profile and user_profile.education_goal else 'Not specified'}
        
        MAIN REGULATIONS:
        1. Conduct a natural dialogue, asking no more than one question at a time
        2. If the user asks a question, answer it first, then ask your question
        3. Don't ever ask for information that's already in the profile
        4. Use the user name as soon as you know it
        5. If all the necessary data are available, transfer the management to ProgrammeDecisionAgent
        
        FORMAT ANSWER:
        If you need to save the data, return it to the JSON format at the end of the answer:
        {{"user_name": "...", "program_area": "...", "education_goal": "..."}}"""
        
        # Create a GPT Request
        messages = [
            {"role": "system", "content": system_prompt}
        ] + context + [
            {"role": "user", "content": message}
        ]
        
        # Get an answer from GPT.
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        reply = response.choices[0].message.content.strip()
        
        # Check for handoff in response
        if "[handoff]" in reply.lower():
            return ChatResponse(
                answer=reply,
                user_profile_id=self._user_profile_id,
                is_authorized=self.request.is_authorized if hasattr(self.request, 'is_authorized') else False,
                user_profile=user_profile
            )
        
        # Check the availability of JSON with data to save
        json_start = reply.rfind('{')
        json_end = reply.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = reply[json_start:json_end + 1]
            data = json.loads(json_str)
            
            # Update the user profile only if the fields are not yet complete
            if user_profile:
                if "user_name" in data and data["user_name"] and not user_profile.user_name:
                    user_profile.user_name = data["user_name"]
                    user_profile.is_modified = True
                if "program_area" in data and data["program_area"] and not user_profile.program_area:
                    user_profile.program_area = data["program_area"]
                    user_profile.is_modified = True
                if "education_goal" in data and data["education_goal"] and not user_profile.education_goal:
                    user_profile.education_goal = data["education_goal"]
                    user_profile.is_modified = True
                    
            # Remove JSON from user response
            reply = reply[:json_start].strip()
            
        return ChatResponse(
            answer=reply,
            user_profile_id=self._user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self.request, 'is_authorized') else False,
            user_profile=user_profile
        )

    async def transfer_to_program_decision(self) -> Agent:
        """
        Transfer to the Program Decision Agent after collecting basic information.
        """
        from ai.mas.utils.handoffs import transfer_to_program_decision_agent
        return transfer_to_program_decision_agent(
            user_profile_id=self._user_profile_id,
            openai_api_key=self.gpt_client.api_key,
            knowledge_client=self._knowledge_client
        )
