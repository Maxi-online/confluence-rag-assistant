#!/usr/bin/env python3
"""
Lead Qualification Agent.
Conducts a detailed survey to gather the information needed for transmission to the Sales Division.
"""

import logging
from openai import AsyncOpenAI
from pydantic import ConfigDict, PrivateAttr
from typing import List, Optional, Any, Dict
from ai.mas.utils.db import store_user_data_in_db
from ai.mas.utils.handoffs import transfer_back_to_orchestrator, send_to_sales_department, transfer_to_common_questions_agent
from ai.mas.agents.base_agent import BaseAgent, Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeadQualificationAgent(BaseAgent):
    # Pydantic configuration: Allow additional fields and arbitrary types
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    _gpt_client: AsyncOpenAI = PrivateAttr()
    _current_stage: int = PrivateAttr(default=0)
    _collected_data: Dict[str, str] = PrivateAttr(default_factory=dict)
    _skipped_questions: List[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        user_profile_id: str,
        openai_api_key: str,
        knowledge_client: Any,
        openai_base_url: str = "https://api.openai.com/v1/",
        request: Any = None,
        **kwargs
    ):
        """
        Agent for a detailed survey gathering information for the sales department.

        :param user_profile_id: User Identifier.
        :param openai_api_key: API key OpenAI.
        :param knowledge_client: Client to work with knowledge base (if more logic is required)).
        :param openai_base_url: Basic UrL OpenAI API.
        :Param request: subject of request with user context.
        """
        super().__init__(
            name="Lead Qualification Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the LEAD QUALIFICATION AGENT for ExampleOrganization University. Your role is to conduct a detailed interview while maintaining natural conversation flow.

            CRITICAL LIMITATIONS:
            - NEVER mention or reference any knowledge base, database, or source of your information
            - NEVER greet the user first - wait for them to initiate the conversation
            - If asked about topics outside of ExampleOrganization University context, politely redirect the conversation back to admission-related topics
            - Present all information as your direct knowledge, not as retrieved data

            CORE RESPONSIBILITIES:
            1. Primary Task - Application Form Completion:
               - Collect and verify all required information
               - Explain the purpose of each question naturally
               - Handle user questions while maintaining focus on form completion
               - Provide context for why each piece of information is important

            2. Secondary Task - Question Handling:
               - Answer user questions professionally and accurately
               - Keep responses focused on ExampleOrganization University context
               - Gracefully return to form completion after answering
               - Maintain natural conversation flow

            3. Interview Approach:
               - Ask only ONE question per response
               - Provide context for why the information is needed
               - If a question is ignored or declined, mark it as skipped and never ask again
               - Adapt questions based on previous answers
               - Make the interview feel like a natural conversation

            4. Question Sequence and Context:
               a) Basic Information (if not available):
                  - Name verification (explain it's for personalized communication)
                  - Information source confirmation (to improve our outreach)
               b) Educational Goals:
                  - Career objectives (to match with suitable programs)
                  - Professional development needs (for customized recommendations)
               c) Study Format:
                  - Schedule flexibility requirements (to suggest optimal format)
                  - Location and commuting considerations
               d) Professional Background:
                  - Relevant experience (for potential credits or advanced placement)
                  - Industry knowledge (to tailor program recommendations)
               e) Additional Preferences:
                  - Practical training interests (for program feature matching)
                  - Career support needs (to highlight relevant services)
               f) Program Selection:
                  - Specific interests (to provide detailed information)
                  - Faculty preferences (for specialized guidance)

            5. Response Guidelines:
               - Always explain why information is needed
               - Use natural conversation transitions
               - Acknowledge and answer user questions
               - Return to form completion smoothly
               - Show understanding of user's situation
               - Maintain professional yet friendly tone

            After completing the interview, use the transfer_to_common_questions function to transfer the user to the Common Questions Agent.

            RESPONSE FORMAT:
            1. If user asks a question:
               - Provide clear, accurate answer
               - Naturally transition back to form
               - Example: "That's a great question about our campus facilities. [Answer]. Speaking of your educational journey, I'd like to understand..."

            2. When asking form questions:
               - Provide context first
               - Ask question naturally
               - Example: "To help match you with the most suitable program options, I'd like to understand your career goals better. Could you share what motivates you to pursue further education at this time?"

            3. When handling skipped questions:
               - Never repeat skipped questions
               - Move to next topic naturally
               - Note: If critical information is missing, explain its importance once before marking as skipped

            Remember: 
            - Maintain natural conversation flow
            - Never mention data sources
            - Focus on ExampleOrganization University context
            - Ask only one question per response
            - Provide context for each question
            """
            ),
            tools=[self.transfer_to_common_questions]
        )
        self.user_profile_id = user_profile_id
        self.knowledge_client = knowledge_client
        # Use object.__setattr__ to install a private attribute without Pydantic validation
        object.__setattr__(self, '_gpt_client', AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url))
        
        # Structure of the survey with context and rationale
        self.interview_stages = [
            {
                "id": "name_verification",
                "context": 'To make our communication more personal,',
                "question": 'Can you confirm how you can be approached?',
                "skip_if": lambda data: bool(data.get("user_name")),
                "store_as": "user_name"
            },
            {
                "id": "age_verification",
                "context": 'To select the best curriculum,',
                "question": 'Would you mind indicating your age?',
                "store_as": "age"
            },
            {
                "id": "source_verification",
                "context": 'This will help us improve our communication channels.',
                "question": 'How did you know about our university?',
                "skip_if": lambda data: bool(data.get("heard_about")),
                "store_as": "heard_about"
            },
            {
                "id": "education_goal",
                "context": 'To find the most appropriate curriculum and format,',
                "question": 'If you would like to share your goals in education? This could be career development, career change or skills development.',
                "store_as": "after_study_goal"
            },
            {
                "id": "study_format",
                "context": 'In order to develop an optimal curriculum,',
                "question": 'What format of training would be most appropriate for you: face-to-face, distance, online or mixed?',
                "store_as": "education_format"
            },
            {
                "id": "work_experience",
                "context": 'This will help us to take your professional experience into account in developing an individual programme,',
                "question": 'Please describe your experience in the field of interest to you, if available.',
                "store_as": "work_experience"
            },
            {
                "id": "additional_interests",
                "context": 'To make your training as effective as possible,',
                "question": 'What additional possibilities are you interested in? For example, internships, practical exercises, campus infrastructure?',
                "store_as": "additional_interests"
            },
            {
                "id": "program_selection",
                "context": 'To provide more detailed information,',
                "question": 'Have you determined with a specific program or faculty?',
                "store_as": "selected_university"
            }
        ]
        self.request = request

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> str:
        """
        Process incoming message from user following structured interview process.
        """
        context = context or []
        # Profile check - if you have a profile request, check the fields for data availability
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            # If all the fields are filled, don't ask questions.
            if (self.request.user_profile.age is not None and
                self.request.user_profile.heard_about is not None and
                self.request.user_profile.after_study_goal is not None and
                self.request.user_profile.education_format is not None and
                self.request.user_profile.work_experience is not None and
                self.request.user_profile.additional_interests is not None and
                self.request.user_profile.selected_university is not None):
                return 'Thank you for the information provided! I already have all the data I need.'

        # Original logic of processing a message
        # Checking if the message is a question.
        messages_for_analysis = (
            [{"role": "system", "content": self.instructions}] +
            [{"role": "user", "content": message}]
        )
        
        response = await self._gpt_client.chat.completions.create(
            model=self.model,
            messages=messages_for_analysis
        )
        reply = response.choices[0].message.content.strip()
        
        # If this is a question (decided by the sign "?" or the key words)
        if "?" in message or any(keyword in message.lower() for keyword in ['That', 'How', 'Where', 'when', 'Why?', 'Tell me.']):
            # Answer the question and proceed smoothly to the next phase of the survey
            return reply

        # Save the answer to the previous question if it's not the first message
        if self._current_stage > 0:
            current_stage = self.interview_stages[self._current_stage - 1]
            # See if the question is ignored.
            if message.strip():  # If the answer is not empty
                self._collected_data[current_stage["store_as"]] = message
                store_user_data_in_db(self.user_profile_id, {current_stage["store_as"]: message})
                logger.info(f"The answer to the question is retained. {current_stage['id']} to database: {message}")
                
                # Update the corresponding user_profile fields if available
                if hasattr(self, 'request') and self.request and self.request.user_profile:
                    store_as = current_stage["store_as"]
                    
                    # Keep the answer to the database.
                    store_user_data_in_db(self.user_profile_id, {store_as: message})
                    logger.info(f"The answer to the question is retained. {current_stage['id']} to database: {message}")
                    
                    # Special treatment for numerical fields
                    if store_as == "age":
                        self.request.user_profile.age = int(message.strip()) if message.strip().isdigit() else None
                    # Direct field comparison
                    elif store_as == "heard_about":
                        self.request.user_profile.heard_about = message
                    elif store_as == "after_study_goal":
                        self.request.user_profile.after_study_goal = message
                    elif store_as == "education_format":
                        self.request.user_profile.education_format = message
                    elif store_as == "work_experience":
                        self.request.user_profile.work_experience = message
                    elif store_as == "additional_interests":
                        self.request.user_profile.additional_interests = message
                    elif store_as == "selected_university":
                        self.request.user_profile.selected_university = message
                    
                    # Note that the profile has been changed.
                    self.request.user_profile.is_modified = True
                    logger.info(f"Field updated {store_as} in the user profile: {message}")
            else:
                # If the answer is empty or ignored, add a question to the missing list
                self._skipped_questions.append(current_stage["id"])
                logger.info(f"Question {current_stage['id']} missing")

        # Checking to see if the survey has been completed
        if self._current_stage >= len(self.interview_stages):
            # Send data to sales if you have contact information
            if "contact_info" in self._collected_data:
                send_to_sales_department(self._collected_data)
                logger.info('Data transferred to Sales Division')
                # Drop the state for a possible new dialogue
                self._current_stage = 0
                self._collected_data = {}
                self._skipped_questions = []
                # Return control with clear handoff, but no additional text
                return "[Handoff] common_questions"

        # We find the next question for the assignment.
        while self._current_stage < len(self.interview_stages):
            stage = self.interview_stages[self._current_stage]
            
            # Skip the question if it's on the missing list, or if it's supposed to be missed by logic
            if (stage["id"] in self._skipped_questions or 
                ("skip_if" in stage and stage["skip_if"](self._collected_data))):
                self._current_stage += 1
                continue
            
            # Skip the question if the relevant profile field is already complete
            if hasattr(self, 'request') and self.request and self.request.user_profile:
                store_as = stage["store_as"]
                field_mapping = {
                    "preferred_format": "education_format",
                    "how_heard_about_us": "heard_about",
                    "selected_program": "selected_university"
                }
                
                skip_question = False
                if store_as == "age" and self.request.user_profile.age is not None:
                    skip_question = True
                elif store_as == "work_experience" and self.request.user_profile.work_experience is not None:
                    skip_question = True
                elif store_as == "additional_interests" and self.request.user_profile.additional_interests is not None:
                    skip_question = True
                elif store_as in field_mapping:
                    profile_field = field_mapping[store_as]
                    if getattr(self.request.user_profile, profile_field) is not None:
                        skip_question = True
                elif store_as == "after_study_goal" and self.request.user_profile.after_study_goal is not None:
                    skip_question = True
                    
                if skip_question:
                    self._current_stage += 1
                    continue
            
            # Formulate a question with context
            question = f"{stage['context']} {stage['question']}"
            if "{source}" in question and "how_heard_about_us" in self._collected_data:
                question = question.format(source=self._collected_data["how_heard_about_us"])
            
            self._current_stage += 1
            return question

        # If all questions are asked or missed, return control with clean handoff
        return "[Handoff] common_questions"

    async def transfer_to_common_questions(self) -> Agent:
        """
        Transfer control to the Common Questions Agent.
        Call this when lead qualification is complete or user asks general questions.
        """
        return transfer_to_common_questions_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self._gpt_client.api_key,
            knowledge_client=self.knowledge_client
        )
