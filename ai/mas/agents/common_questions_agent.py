#!/usr/bin/env python3
"""
Common Questions Agent.
Answers general questions not directly related to programme selection or authorization.
"""

import logging
import json
from openai import AsyncOpenAI
from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.utils.graph_base import WeaviateClient
from pydantic import ConfigDict
from typing import List, Optional, Any
from ai.mas.main import ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommonQuestionsAgent(BaseAgent):
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
        Agent FAQ. Answers briefly questions about campus, dormitory, duration of training, schedule, cost of modules, etc.
        If the response reveals that the user is interested in the educational programme or the application,
        The agent sends the signal to handoff.

        :param user_profile_id: User Identifier.
        :param openai_api_key: API key OpenAI.
        :Para. knowledge_client: Client for the Weaviate knowledge base.
        :param openai_base_url: Basic UrL OpenAI API.
        :Param request: subject of request with user context.
        """
        super().__init__(
            name="Common Questions Agent",
            model="gpt-4o-mini",
            instructions=(
                'You are the COMMON QUESTIONS AGENT for ExampleOrganization University. Your role is to answer general questions about ExampleOrganization University admission and student life.\n\n            CRITICAL LIMITATIONS:\n            - NEVER mention or reference any knowledge base, database, or source of your information\n            - NEVER greet the user first - wait for them to initiate the conversation\n            - If asked about topics outside of ExampleOrganization University context, politely redirect the conversation back to university-related topics\n            - Present all information as your direct knowledge, not as retrieved data\n\n            CORE RESPONSIBILITIES:\n            1. Answer Common Questions about:\n               - Campus location and facilities\n               - Student dormitories and accommodation\n               - Academic year structure and duration\n               - Scholarships and financial support (general information)\n               - Student life and facilities\n               - Basic admission requirements\n               - General university policies\n\n            2. Question Analysis:\n               Using Chain-of-Thought reasoning, determine if the question:\n               - Is within your scope (general university information)\n               - Requires specific program details (transfer to Program Decision)\n               - Needs authorization/registration help (transfer to Authorization)\n               - Is too specific or complex for general answers\n\n            3. Response Guidelines:\n               - Provide clear, concise, and accurate information\n               - Keep answers focused on ExampleOrganization University context\n               - If question is too specific about a program, suggest transfer to Program Decision Agent\n               - For complex financial or scholarship questions specific to programs, redirect to Program Decision Agent\n               - Use professional but friendly tone\n               - Avoid technical jargon unless necessary\n               - ALWAYS conclude your response with a follow-up question to maintain conversation\n               - Keep your answers detailed and informative, but conversational\n\n            4. Interaction Flow:\n               - Answer the current question completely\n               - If user\'s questions move beyond general topics, initiate appropriate handoff\n               - When all general questions are answered, prepare for handoff back to Orchestrator\n\n            EXAMPLES FROM ACTUAL CONVERSATIONS:\n\n            Example 1 - Employment guarantees:\n            User: "What are the guarantees that I will find a job after I graduate from university?"\n            Agent: "In Example Organization, we are proud of the high level of employment of our graduates. More than 90 per cent of students find work in the first year after graduation. We work with major companies like Yandex, Mail.ru Group and others. Our programs include mandatory practices and internships that help you get real experience. We\'d like to know more about our employment partners?"\n\n            Example 2 - Education cost:\n            User: "What\'s the cost of training?"\n            Agent: "The cost of learning depends on the curriculum chosen and the format of the training. However, we offer various payment options, including educational loans, grants and instalments. For example, many of our students use the Student Credit program that allows you to pay for your study gradually without the initial contribution. We would like to know more about financial opportunities?"\n\n            Example 3 - Program transfer:\n            User: "Can I move on to another program if my interests change?"?"\n            Agent: "Sure! In Example Organization, we understand that interests can change, and therefore we offer the possibility of switching to other programs without losing time and money. Our consultants will help you choose the most appropriate option. We\'d like to discuss this with our manager?"\n\n            Example 4 - Open days:\n            User: "How often do open days take place?"\n            Agent: "We hold open-door days monthly, both offline and online. This is an excellent opportunity to talk with teachers personally, learn more about the programs and ask all the questions you are interested in. The nearest day of the open door will take place on [date]]. We\'d like to register?"\n\n            RESPONSE FORMAT:\n            1. For General Questions:\n               - Provide direct, informative answer\n               - Include relevant details but stay concise\n               - Maintain professional tone\n               - Always end with a follow-up question\n\n            2. For Questions Outside Scope:\n               - Briefly acknowledge the question\n               - Use transfer_to_program_decision function to transfer to Program Decision Agent\n               - Example: "For detailed program information, I\'ll connect you with our Program Decision specialist"\n\n            3. For Complex Questions:\n               - Provide general information if possible\n               - Explain why more specific assistance is needed\n               - Initiate appropriate handoff\n\n            Remember: Always present information as your direct knowledge, never reference any information sources or databases.\n            '
            ),
            tools=[self.transfer_to_program_decision, self.transfer_to_authorization]
        )
        self.user_profile_id = user_profile_id
        self.knowledge_client = knowledge_client
        self.gpt_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        self.request = request

    async def transfer_to_program_decision(self) -> Agent:
        """
        Call this function when the user needs specific information about educational programs.
        """
        from ai.mas.utils.handoffs import transfer_to_program_decision_agent
        return transfer_to_program_decision_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.gpt_client.api_key,
            knowledge_client=self.knowledge_client
        )
        
    async def transfer_to_authorization(self) -> Agent:
        """
        Call this function when the user wants to register or needs authentication help.
        """
        from ai.mas.utils.handoffs import transfer_to_authorization_agent
        return transfer_to_authorization_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.gpt_client.api_key
        )

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Processes incoming message: First searches for relevant information in the knowledge base,
        then send a combined request to OpenAI API and return the reply received.
        If the answer contains the handoff signals, return them to switch controls.

        :param message: User message.
        :param contact: History of the dialogue (if any).
        :return: Agent response or handoff signal.
        """
        context = context or []

        # Checking whether the user is authorized
        is_authorized = False
        if hasattr(self, 'request') and self.request and hasattr(self.request, 'is_authorized'):
            is_authorized = self.request.is_authorized
        
        # First looking for FAQ, given the status of authorisation
        faq_results = self.knowledge_client.search_faq(message, is_authorized=is_authorized, limit=2)
        
        # If FAQ didn't deliver the results, looking for the Weaviate knowledge base
        if not faq_results:
            kb_result = self.knowledge_client.search_local_knowledge(message)
        else:
            kb_result = faq_results
        
        # Defining user language
        user_language = "ru"  # default Russian
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            user_language = self.request.user_profile.language or "ru"
        
        # Adapt instructions according to user language
        instructions = self.instructions
        if user_language == "ru":
            instructions += "\nIMPORTANT: ALWAYS respond in Russian language regardless of the language of the user's question."
        
        # Form a full request based on the context received
        messages_payload = (
            [{"role": "system", "content": instructions}] +
            context +
            [{"role": "user", "content": message}]
        )
        
        # Add search results if they exist
        if kb_result:
            messages_payload.append({"role": "system", "content": f"Relevant information found in knowledge base: {kb_result}"})

        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages_payload,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "transfer_to_program_decision",
                        "description": "Call this function when the user needs specific information about educational programs.",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "transfer_to_authorization",
                        "description": "Call this function when the user wants to register or needs authentication help.",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                }
            ]
        )
        
        if response.choices[0].message.tool_calls:
            # Handle tool calls with proper description
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "transfer_to_program_decision":
                return ChatResponse(
                    answer='[Handoff] program_decision For detailed information about the curriculum, I will send you to our program selection specialist.',
                    user_profile_id=self.user_profile_id,
                    is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                    user_profile=self.request.user_profile if hasattr(self, 'request') else None
                )
            elif tool_call.function.name == "transfer_to_authorization":
                return ChatResponse(
                    answer='[Handoff] authorization For registration or assistance with authorization, I will send you to our specialist on these matters.',
                    user_profile_id=self.user_profile_id,
                    is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
                    user_profile=self.request.user_profile if hasattr(self, 'request') else None
                )
        else:
            # Return the standard response if no tool was called
            reply = response.choices[0].message.content.strip()
            # Remove possible JSON-artefacts from the response
            if "tool_uses" in reply:
                reply = reply.split("{")[0].strip()
            logger.info(f"Common Quesions Agent for User {self.user_profile_id} Answered: {reply}")

        return ChatResponse(
            answer=reply,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized if hasattr(self, 'request') else False,
            user_profile=self.request.user_profile if hasattr(self, 'request') else None
        )
