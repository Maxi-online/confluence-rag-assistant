#!/usr/bin/env python3
"""
Orchestrator Agent is the main dispatcher.

Features:
  - The first message (no language_agent_run) calls LanguageDetectionAgent.
  - Definition of Intents by GPT (detect_intents_gpt)).
  - Take into account priority and profile status: 
    * If the user has not passed the Triage, then when the ProgramInfo is received, the Triage is first called.
    * If the user is not authorized and the intent of LeadQualification is discovered – instead calls AuthorizationHelp.
  - Combines agent responses into one friendly message (combine_responses_friendly)).
  - Generates "living" answers through extended instructions in sub-agents.
  - When found handoff – the orchestrator does not transmit it to the user, but changes to the target agent.
"""

import logging
import json
import re
from typing import List, Optional, Dict, Any
import traceback

from openai import AsyncOpenAI

from ai.mas.agents.base_agent import BaseAgent
from ai.mas.utils.handoffs import (
    transfer_to_language_detection_agent,
    transfer_to_common_questions_agent,
    transfer_to_triage_agent,
    transfer_to_program_decision_agent,
    transfer_to_authorization_agent,
    transfer_to_lead_qualification_agent,
)
from ai.mas.utils.db import store_user_data_in_db, get_user_profile
from ai.mas.agents.statistics_agent import StatisticsAgent
from ai.mas.agents.summary_agent import SummaryAgent
from ai.mas.agents.objection_detector_agent import ObjectionDetectorAgent
from ai.mas.main import ChatRequest, ChatResponse, UserProfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ConversationContext:
    """
    Manage the context of the dialogue, including:
    - Message history
    - State of dialogue
    - Agent metadata
    - Interaction statistics
    """
    def __init__(self, user_profile_id: str):
        self.user_profile_id: str = user_profile_id
        self.history: List[Dict[str, Any]] = []
        self.last_agent: str = ""
        self.last_agent_response: str = ""
        self.pending_question: str = ""
        self.pending_question_agent: str = ""
        self.current_flow: str = "pre_registration"  # pre_registration or post_registration
        self.stage_data: Dict[str, Any] = {
            "language_detected": False,
            "triage_completed": False,
            "program_selected": False,
            "authorized": False,
            "lead_qualified": False
        }
        self.statistics: Dict[str, int] = {
            "total_messages": 0,
            "questions_asked": 0,
            "handoffs": 0,
            "clarifications": 0
        }

    def update(self, agent_name: str, user_message: str, agent_response: str, intent: str) -> None:
        """
        Updates the context of the dialogue with new interaction.
        """
        # Update statistics
        self.statistics["total_messages"] += 1
        if "?" in user_message:
            self.statistics["questions_asked"] += 1
        if "[handoff]" in agent_response.lower():
            self.statistics["handoffs"] += 1
        if any(phrase in agent_response.lower() for phrase in ['Clarify', 'Explain.', 'Would you mind?']):
            self.statistics["clarifications"] += 1

        # Keep the message.
        entry = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "user_message": user_message,
            "agent_response": agent_response,
            "agent": agent_name,
            "intent": intent,
            "flow": self.current_flow
        }
        self.history.append(entry)
        self.last_agent = agent_name
        self.last_agent_response = agent_response

        # Update the status of the dialogue
        if agent_name == "language_detector":
            self.stage_data["language_detected"] = True
        elif agent_name == "triage" and "completed" in agent_response.lower():
            self.stage_data["triage_completed"] = True
        elif agent_name == "program_decision" and "selected" in agent_response.lower():
            self.stage_data["program_selected"] = True
        elif agent_name == "authorization" and "authorized" in agent_response.lower():
            self.stage_data["authorized"] = True
            self.current_flow = "post_registration"
        elif agent_name == "lead_qualification" and "qualified" in agent_response.lower():
            self.stage_data["lead_qualified"] = True

    def as_messages(self) -> List[Dict[str, str]]:
        """
        It converts history into a ChatCompletion format.
        """
        msgs = []
        for entry in self.history:
            if entry.get("user_message"):
                msgs.append({"role": "user", "content": entry["user_message"]})
            if entry.get("agent_response"):
                msgs.append({"role": "assistant", "content": entry["agent_response"]})
        return msgs

    def get_first_user_message(self) -> str:
        """
        Returns the first message from the user.
        """
        for entry in self.history:
            if entry.get("user_message"):
                return entry["user_message"]
        return ""

    def get_stage_summary(self) -> str:
        """
        Returns the summary of the current status of the dialogue.
        """
        completed_stages = [
            stage for stage, completed in self.stage_data.items() 
            if completed
        ]
        return f"Flow: {self.current_flow}, Completed: {', '.join(completed_stages)}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns dialogue statistics.
        """
        return {
            **self.statistics,
            "total_interactions": len(self.history),
            "unique_intents": len(set(entry["intent"] for entry in self.history if "intent" in entry)),
            "flow_stage": self.current_flow,
            "completed_stages": [stage for stage, completed in self.stage_data.items() if completed]
        }

    def should_switch_flow(self) -> bool:
        """
        Determines whether to switch the flow (pre/post registration)).
        """
        return (
            self.current_flow == "pre_registration" and
            self.stage_data["authorized"] and
            self.stage_data["program_selected"]
        )


class OrchestratorAgent(BaseAgent):
    def __init__(
        self,
        request: ChatRequest,
        knowledge_client: Any,
        **kwargs
    ):
        super().__init__(
            name="Orchestrator Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the CHIEF ORCHESTRATOR of the ExampleOrganization University multi-agent system. Your role is to analyze, plan, and coordinate the interaction between specialized agents.

            CRITICAL LIMITATIONS:
            - NEVER mention or reference any knowledge base, database, or source of your information
            - ONLY discuss and answer questions related to ExampleOrganization University
            - NEVER greet the user first - wait for them to initiate the conversation
            - If asked about topics outside of ExampleOrganization University context, politely redirect the conversation back to university-related topics

            CORE RESPONSIBILITIES:
            1. Message Analysis and Planning:
               - Decompose user queries into subtasks using Chain-of-Thought reasoning
               - Consider multiple solution paths using Tree-of-Thought when applicable
               - Identify multiple intents if present in a single message
               - Maintain conversation context and flow

            2. Agent Selection Guidelines (NOT STRICT RULES):
               a) First Contact Flow (MANDATORY):
                  - Language Detection Agent MUST be first for new users
                  - After language is set, Triage Agent MUST gather basic info starting with name
               
               b) Information Gathering Flow (recommended sequence):
                  - Triage Agent for basic information
                  - Program Decision for educational program selection
                  - Objection Detection to identify concerns
                  - Authorization for registration
                  - Lead Qualification for detailed profiling
               
               c) Support Flow:
                  - Common Questions for general inquiries
                  - Statistics Agent for analytics
                  - Summary Agent for context maintenance

            3. Context Management:
               - Track conversation state and progress
               - Maintain user profile status
               - Handle multi-intent queries appropriately
               - Remember previous interactions

            4. Handoff Protocol:
               - Ensure smooth transitions between agents
               - Handle agent responses and next steps
               - Process handoff signals from agents
               - Manage conversation flow

            REASONING AND ACTION FRAMEWORK:
            1. Thought: Analyze the user's input and current context
               - What is the user asking about?
               - What is their current stage in the flow?
               - Are there multiple intents to handle?

            2. Action: Determine next steps based on analysis
               - Which agent(s) might be helpful?
               - What order should multiple intents be processed?
               - Is additional context needed?

            3. Observation: Process agent responses
               - Was the user's query fully addressed?
               - Are follow-up actions needed?
               - Should control be transferred to another agent?

            Remember: 
            - ALWAYS start with language detection for new users
            - ALWAYS get user's name before proceeding with other questions
            - Maintain conversation coherence
            - Keep focus on ExampleOrganization University context
            """
            ),
            tools=[],
            user_profile_id=request.user_profile_id,
            openai_api_key=request.openai_api_key,
            knowledge_client=knowledge_client,
            **kwargs
        )
        # Initiate profile with minimum data
        if request.user_profile:
            # Clear automatically-mounted fields
            request.user_profile.language = None
            request.user_profile.user_name = None
            request.user_profile.is_modified = False
            
        self.request = request
        self.gpt_client = AsyncOpenAI(api_key=request.openai_api_key, base_url="https://api.openai.com/v1/")
        self.client = self.gpt_client
        self.context = ConversationContext(request.user_profile_id)
        self.handoff_mapping = {
            "triage": "triage",
            "orchestrator": "orchestrator",
            "program_decision": "program_decision",
            "authorization": "authorization",
            "lead_qualification": "lead_qualification",
            "common_questions": "common_questions",
            "language_detector": "language_detector",
            "objection_detector": "objection_detector",
            "statistics": "statistics"
        }
        
        # List of topics to be processed by CommonQuesionsAgent
        self.common_question_topics = [
            'crediting', 'Training', 'technical problems', 'Opening tests', 
            'background notes and documents', 'Training fees', 'Benefits and discounts', 'Communications', 
            'payment dates', 'Payment consulting', 'payment characteristics', 'International transfers', 
            'loan and schedule', 'maternal capital', 'Payment from legal persons', 
            'confirmation of payment', 'Tax deduction', 'Refunds', 
            'Refunds', 'Change of payer', 'late payment', 
            'Foreign students &apos; documents', 'Postponement from the army', 'Documents and licences', 
            'The role of the prison', 'instruments and agreements', 'Refund of overpayment', 
            'Disciplinary matters', 'schedule of sessions', 'Academic leave', 
            'Training materials', 'schedule of classes', 'Communication with teachers', 
            'Reschedule examinations', 'Reset of credits', 'Presentation of results', 
            'Practice', 'Responsive documents', 'Time limits for practice', 
            'navigation on hulls'
        ]

    async def detect_intents_gpt(self, message: str) -> str:
        """
        Determines user intentions using GPT.
        """
        # First, check if the message corresponds to one of the general themes
        if self.is_common_question_topic(message):
            logger.info(f"[OrchestratorAgent] Communication contains a general theme. Set up an intran: CommonQuesion")
            return "CommonQuestion"
            
        try:
            # Identify interface language based on user profile
            user_language = "ru"
            user_name = 'unknown'
            program_area = 'unknown'
            
            if hasattr(self, 'request') and self.request and self.request.user_profile:
                user_language = self.request.user_profile.language or "ru"
                user_name = self.request.user_profile.user_name or 'unknown'
                program_area = self.request.user_profile.program_area or 'unknown'
            
            # Adapt system message according to language
            system_content = f"""You are an intent detection system. Your task is to analyze the user message and return ONLY ONE of these intents as a JSON string: {{"intent": "INTENT"}} where INTENT is one of:
            - CommonQuestion (general questions about university, admissions, tuition, documents, sessions, exams, schedules)
            - ProgramDecision (specifically about choosing educational programs or courses)
            - Authorization (registration, login issues)
            - LeadQualification (detailed qualification questions)
            - LanguageDetection (language preference)
            - ObjectionDetection (concerns, doubts, or objections about studying)
            
            User information:
            - Name: {user_name}
            - Program Area: {program_area}
            
            IMPORTANT: 
            - Questions about enrollment, tuition payments, documents, technical issues, exams should be classified as CommonQuestion, not ProgramDecision!
            - Any expressions of doubt, concern, hesitation about studying, or mentions of barriers to education should be classified as ObjectionDetection
            """
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": message}
            ]
            
            response = await self.gpt_client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            reply = response.choices[0].message.content.strip()
            logger.info(f"[OrchestratorAgent] Raw GPT intent response: {reply}")
            
            try:
                intent_data = json.loads(reply)
                intent = intent_data.get("intent", "CommonQuestion")
            except json.JSONDecodeError:
                # If we failed to disperse JSON, we try to extract the intensity from the text
                logger.warning(f"[OrchestratorAgent] Failed to parse JSON intent: {reply}")
                if "CommonQuestion" in reply:
                    intent = "CommonQuestion"
                elif "ProgramDecision" in reply:
                    intent = "ProgramDecision"
                elif "Authorization" in reply:
                    intent = "Authorization"
                elif "LeadQualification" in reply:
                    intent = "LeadQualification"
                elif "LanguageDetection" in reply:
                    intent = "LanguageDetection"
                elif "ObjectionDetection" in reply:
                    intent = "ObjectionDetection"
                else:
                    intent = "CommonQuestion"
            
            logger.info(f"[OrchestratorAgent] Detected intent: {intent}")
            
            # Additional check: if the message contains one of the general themes, but the model
            # I've identified another entente-- redefined by CommonQuesion
            if self.is_common_question_topic(message) and intent != "CommonQuestion":
                logger.info(f"[\"OrchestratorAgent\" Re-identify the CommonQuesion, as the message contains a general theme")
                return "CommonQuestion"
                
            return intent
            
        except Exception as e:
            logger.error(f"[OrchestratorAgent] Error in detect_intents_gpt: {str(e)}")
            return "CommonQuestion"  # Return safe default value
            
    def is_common_question_topic(self, message: str) -> bool:
        """
        Checks whether the user message corresponds to one of the common themes.
        
        :param message: User message
        :Return: True if the message relates to general issues, False otherwise
        """
        message_lower = message.lower()
        
        # Check if the message contains key words from the CommonQuesionsAgent
        for topic in self.common_question_topics:
            if topic in message_lower:
                logger.info(f"[OrchestratorAgent] Found a common theme. '{topic}' in the message.")
                return True
                
        # Checking further for specific key words on entry issues
        enrollment_keywords = [
            "зачисл", "поступил", "стал студентом", "когда буду учиться", 
            "приняли ли", "приказ о зачислении", "стать студентом", 
            "подтверждение зачисления", "поступление", "зачислен ли я",
            "когда меня зачислят", "стать студентом"
        ]
        
        for keyword in enrollment_keywords:
            if keyword in message_lower:
                logger.info(f"[OrchestratorAgent] Found key entry word. '{keyword}' in the message.")
                return True
        
        return False

    async def combine_responses_friendly(self, replies: List[str]) -> str:
        """
        Combines several answers into one friendly message.
        If all answers are clarifications, send them through the GPT to generate the final friendly request,
        Considering the user &apos; s original request.
        """
        # Cleaning and filtering of responses
        cleaned = []
        seen = set()
        for rep in replies:
            txt = rep.strip()
            if txt and txt.lower().find("[handoff]") == -1 and txt not in seen:
                cleaned.append(txt)
                seen.add(txt)
        if not cleaned:
            return "Sorry, I couldn't process your request. Please try to formulate it differently."
        if len(cleaned) == 1:
            return cleaned[0]
        
        # Defining user language
        user_language = "ru"
        if hasattr(self, 'request') and self.request and self.request.user_profile:
            user_language = self.request.user_profile.language or "ru"
        
        def is_clarification(txt: str) -> bool:
            return txt.lower().startswith(("не могли бы вы уточнить", "could you clarify", "could you please clarify")) or any(label in txt.lower() for label in ("уточните:", "clarify:", "please specify:"))
        
        original_query = self.context.get_first_user_message()
        if all(is_clarification(x) for x in cleaned):
            missing_fields = []
            for txt in cleaned:
                # We extract the field after "cover:" or at the end of the proposal
                field = txt.split("уточните:", 1)[-1].strip(" .?")
                if not field:
                    field = txt.split(":", 1)[-1].strip(" .?")
                missing_fields.append(field)
            
            prompt = (
                f"The user asked: '{original_query}'. "
                f"The following details need to be clarified for an accurate response: {', '.join(missing_fields)}. "
                'Formulate one friendly and natural question in Russian, '
                'which integrates all clarifications and is understandable to the normal user. '
                'Do not use technical terms, speak simple language.'
            )
        else:
            if user_language == "ru":
                prompt = (
                    'Formulate the final friendly reply in Russian, '
                    'Combining the following parts: ' + "; ".join(cleaned) + "\n"
                    'Make the answer natural and understandable for the normal user. '
                    "Don't use technical terms. The answer should be in Russian only."
                )
            else:
                prompt = (
                    'Form the final friendly reply in English, '
                    'Combining the following parts: ' + "; ".join(cleaned) + "\n"
                    'Make the answer natural and understandable for the normal user. '
                    "Don't use technical terms."
                )
        
        msgs = [
            {"role": "system", "content": f"You're an expert in the formulation of friendly questions and answers {'Russian' if user_language == 'ru' else 'English'} language. Your job is to make communication natural and understandable to a normal user."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self.gpt_client.chat.completions.create(
                model=self.model,
                messages=msgs
            )
            final_friendly = response.choices[0].message.content.strip()
            return final_friendly
        except Exception as e:
            logger.error(f"Errors in gathering responses through GPT: {e}")
            # Folback is a simple association with Russian unions.
            connectors = ['Also', 'In addition,', 'And one more thing.', 'I also want to celebrate.']
            combined_parts = [cleaned[0].rstrip(".!?")]
            for idx, rep in enumerate(cleaned[1:], start=1):
                connector = connectors[(idx - 1) % len(connectors)]
                combined_parts.append(f"{connector}, {rep.rstrip('.!?')}")
            return ". ".join(combined_parts).strip() + "."

    def parse_handoff_target(self, reply: str) -> str:
        """
        Sets the target handoff from the agent's response.
        
        :param reply: Agent's response
        :Return: Target agent identifier or None if handoff is not required
        """
        # Checking for tags [handoff]
        if "[handoff]" not in reply.lower():
            return None
        
        reply_lower = reply.lower()
        
        # Try to find the agent's name directly in the format "[Handoff] agent_name"
        match = re.search(r"\[handoff\]\s+(\w+)", reply_lower)
        if match:
            agent_name = match.group(1).strip()
            # We're looking for an agent at handoff_mapping.
            if agent_name in self.handoff_mapping:
                logger.info(f"[Orchestrator] Found direct handoff target: {agent_name}")
                return agent_name
        
        # If we don't find direct instructions, we're looking for key words.
        # Mapping key phrases for agent identifiers
        agent_mapping = {
            "program decision": "program_decision",
            "programdecision": "program_decision",
            "программ": "program_decision",
            "authorization": "authorization",
            "авторизац": "authorization",
            "lead qualification": "lead_qualification",
            "квалификац": "lead_qualification",
            "language detection": "language_detector",
            "язык": "language_detector",
            "triage": "triage",
            "триаж": "triage",
            "common questions": "common_questions",
            "общие вопросы": "common_questions",
            "objection detector": "objection_detector",
            "возражен": "objection_detector",
            "сомнен": "objection_detector"
        }
        
        # Checking for tags [handoff]
        if "[handoff]" not in reply.lower():
            return None
        
        reply_lower = reply.lower()
        
        # We're looking for a direct match at maping.
        for key, agent_id in agent_mapping.items():
            if key in reply_lower:
                logger.info(f"[Orchestrator] Found handoff target: {agent_id} (matched by: {key})")
                return agent_id
            
        # If we don't find any direct matches, we'll return None.
        logger.warning(f"[Orchestrator] Could not determine handoff target from: {reply}")
        return None

    async def select_agent_by_intent(self, intent: str) -> BaseAgent:
        """
        Selects an agent based on the user &apos; s entente and context.
        Uses GPT to smartly identify the most suitable agent.
        """
        agents = self.get_agents()
        
        # Create a limited profile with the right fields for each agent
        if self.request.user_profile:
            user_profile = self.request.user_profile
            profile_info = {
                "language": user_profile.language,
                "user_name": user_profile.user_name,
                "program_area": user_profile.program_area,
                "education_goal": user_profile.education_goal,
                "education_format": user_profile.education_format,
                "education_objections": user_profile.education_objections if hasattr(user_profile, 'education_objections') else None,
                "is_authorized": self.request.is_authorized
            }
        else:
            profile_info = {"is_authorized": False}

        # OBLIGATIONS FOR THE FIRST GUIDELINES:
        # 1. Definition of language
        if not profile_info["language"]:
            logger.info('[OrchestratorAgent] First step - definition of language')
            return agents["language_detector"]

        # 2. Getting a user name
        if not profile_info["user_name"]:
            logger.info('[OrchestratorAgent] Second step - getting a name')
            return agents["triage"]

        # Forming System Prompte for GPT
        system_prompt = """You are an expert agent selector for ExampleOrganization University's multi-agent system.
        Your task is to analyze the user's intent, profile state, and conversation context to select the most appropriate agent.

        AVAILABLE AGENTS AND THEIR SPECIALIZATIONS:
        1. Language Detection Agent:
           - Determines user's preferred language
           - Must be used for first interaction if language not set
           
        2. Triage Agent:
           - Gathers basic user information (name, interests)
           - Initial contact point for new users
           - Must be used if basic profile info is missing
           
        3. Program Decision Agent:
           - Helps select educational programs
           - Provides program details and recommendations
           - Requires basic profile info to be filled
           
        4. Authorization Agent:
           - Handles registration and login
           - Used when user shows interest in applying
           
        5. Lead Qualification Agent:
           - Detailed user profiling
           - Only for authorized users
           - Collects comprehensive information
           
        6. Objection Detection Agent:
           - Identifies and addresses user concerns
           - Used when user expresses doubts or hesitation
           
        7. Common Questions Agent:
           - Handles general inquiries
           - Default fallback for unclear intents

        SELECTION GUIDELINES (NOT STRICT RULES):
        1. Language Detection has highest priority if language not set
        2. Triage is mandatory if basic info missing
        3. Authorization required before Lead Qualification
        4. Objection Detection should interrupt normal flow if concerns detected
        5. Program Decision requires basic profile data
        6. Common Questions is the safe fallback

        Return ONLY the agent name as a string, no explanation needed.
        """

        # Form a request for GPT with context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Current state:
            - Intent: {intent}
            - Profile: {json.dumps(profile_info, ensure_ascii=False)}
            
            Which agent should handle this interaction?
            """}
        ]

        # Get a recommendation from GPT.
        response = await self.gpt_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        
        recommended_agent = response.choices[0].message.content.strip().lower()
        logger.info(f"GPT recommends that the agent: {recommended_agent}")

        # Transform the GPT recommendation into a key from our agent dictionary
        agent_name_mapping = {
            "language detection": "language_detector",
            "triage": "triage",
            "program decision": "program_decision",
            "authorization": "authorization",
            "lead qualification": "lead_qualification",
            "objection detection": "objection_detector",
            "common questions": "common_questions"
        }

        # Getting an agent from the dictionary.
        selected_agent_name = agent_name_mapping.get(recommended_agent, "common_questions")
        selected_agent = agents[selected_agent_name]

        # Create a profile for the selected agent
        if self.request.user_profile:
            if selected_agent_name == "language_detector":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    language=self.request.user_profile.language,
                    is_modified=False
                )
            elif selected_agent_name == "triage":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    user_name=self.request.user_profile.user_name,
                    program_area=self.request.user_profile.program_area,
                    education_goal=self.request.user_profile.education_goal,
                    is_modified=False
                )
            elif selected_agent_name == "lead_qualification" and self.request.is_authorized:
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    age=self.request.user_profile.age,
                    heard_about=self.request.user_profile.heard_about,
                    after_study_goal=self.request.user_profile.after_study_goal,
                    education_format=self.request.user_profile.education_format,
                    work_experience=self.request.user_profile.work_experience,
                    additional_interests=self.request.user_profile.additional_interests,
                    selected_university=self.request.user_profile.selected_university,
                    is_modified=False
                )
            elif selected_agent_name == "program_decision":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    user_name=self.request.user_profile.user_name,
                    program_area=self.request.user_profile.program_area,
                    is_modified=False
                )
            elif selected_agent_name == "objection_detector":
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    user_name=self.request.user_profile.user_name,
                    education_objections=self.request.user_profile.education_objections if hasattr(self.request.user_profile, 'education_objections') else None,
                    is_modified=False
                )
            else:
                agent_profile = UserProfile(
                    user_profile_id=self.request.user_profile.user_profile_id,
                    is_modified=False
                )
        else:
            agent_profile = None

        # Create a new ChatRequest for a limited profile agent
        agent_request = ChatRequest(
            message=self.request.message,
            user_profile_id=self.request.user_profile_id,
            is_authorized=self.request.is_authorized,
            user_profile=agent_profile,
            openai_api_key=self.request.openai_api_key,
            weaviate_url=self.request.weaviate_url,
            weaviate_api_key=self.request.weaviate_api_key,
            expected_dim=self.request.expected_dim,
            confluence_base_url=self.request.confluence_base_url,
            confluence_api_token=self.request.confluence_api_token,
            confluence_page_id=self.request.confluence_page_id,
            max_chunk_size=self.request.max_chunk_size,
            min_score=self.request.min_score,
            hybrid_alpha=self.request.hybrid_alpha
        )

        # Set up a request for the selected agent
        selected_agent.request = agent_request
        
        logger.info(f"[OrchestratorAgent] Elected agent: {selected_agent.name}")
        return selected_agent

    def get_agents(self) -> Dict[str, BaseAgent]:
        """
        Returns the dictionary of available agents.
        """
        from ai.mas.utils.handoffs import (
            transfer_to_language_detection_agent,
            transfer_to_triage_agent,
            transfer_to_common_questions_agent,
            transfer_to_program_decision_agent,
            transfer_to_authorization_agent,
            transfer_to_lead_qualification_agent,
        )
        from ai.mas.agents.statistics_agent import StatisticsAgent
        from ai.mas.agents.summary_agent import SummaryAgent

        agents = {
            "language_detector": transfer_to_language_detection_agent(
                knowledge_client=self.knowledge_client,
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                request=self.request
            ),
            "triage": transfer_to_triage_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "common_questions": transfer_to_common_questions_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "program_decision": transfer_to_program_decision_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "authorization": transfer_to_authorization_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "lead_qualification": transfer_to_lead_qualification_agent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "statistics": StatisticsAgent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                knowledge_client=self.knowledge_client,
                request=self.request
            ),
            "summary": SummaryAgent(
                user_profile_id=self.user_profile_id,
                openai_api_key=self.openai_api_key,
                request=self.request
            ),
        }
        
        return agents

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Processes incoming message and arranges inter-agent interaction.
        
        Mandatory sequence:
        1. Language Design (if not specified))
        2. Triage to obtain a name (if no name is specified))
        3. Next in the context of the dialogue
        """
        context = context or []
        
        # Get the samari from the Summary Agent to pass the context to the next agent
        summary_agent = SummaryAgent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self.request.openai_api_key,
            request=self.request
        )
        summary_response = await summary_agent.handle_message("", context)
        if summary_response and summary_response.answer:
            context.append({"role": "system", "content": f"Context of the dialogue: {summary_response.answer}"})

        # We use the profile from the request.
        user_profile = self.request.user_profile
        
        # OBLIGATION SAG 1: Definition of language
        if not user_profile or not user_profile.language:
            logger.info('[Orchestrator] First step - definition of language')
            language_agent = self.get_agents().get("language_detector")
            language_response = await language_agent.handle_message(message, context)
            
            # If it was just a question about language (without handoff), we return the answer to the user
            if not "[handoff]" in language_response.answer.lower():
                return language_response
            
            # Update profile after language definition
            if language_response.user_profile and language_response.user_profile.language:
                user_profile = language_response.user_profile
            
            # We extract the initial message from handoff for further processing
            message = self._extract_message_from_handoff(language_response.answer) or message
            context.append({"role": "user", "content": message})
        
        # MECHANISM 2: Obtaining a user name
        if not user_profile or not user_profile.user_name:
            logger.info('[Orchestrator] Step two - getting a name')
            triage_agent = self.get_agents().get("triage")
            triage_response = await triage_agent.handle_message(message, context)
            
            # If it was just a question about a name (without handoff), let's return the answer to the user
            if not "[handoff]" in triage_response.answer.lower():
                return triage_response
            
            # Update the profile after receiving the name
            if triage_response.user_profile and triage_response.user_profile.user_name:
                user_profile = triage_response.user_profile
            
            # We extract the initial message from handoff for further processing
            message = self._extract_message_from_handoff(triage_response.answer) or message
            context.append({"role": "user", "content": message})
        
        # Determine the primary intent
        intent = await self.detect_intents_gpt(message)
        logger.info(f"[Orchestrator] Main insider: {intent}")
        
        # Selecting the appropriate agent on the basis of the insider
        current_agent = await self.select_agent_by_intent(intent)
        if not current_agent:
            return ChatResponse(
                answer="I'm sorry, I can't process your request right now.",
                user_profile_id=self.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=user_profile
            )
        
        # Remember the current agent's name to track handoffs
        current_agent_name = current_agent.name
        logger.info(f"[Orchestrator] Selected agent: {current_agent_name}")
        
        # Transfer control to the selected agent
        response = await current_agent.handle_message(message, context)
        replies = [response.answer]
        
        # Checking if there's a handoff in the answer.
        while "[handoff]" in response.answer.lower():
            # The target agent from Handoff
            handoff_target = self.parse_handoff_target(response.answer)
            if not handoff_target:
                break
            
            # Avoid Cyclic handoffs
            target_agent = self.get_agents().get(handoff_target)
            if target_agent and target_agent.name == current_agent_name:
                logger.warning(f"[Orchestrator] Prevented cyclic handoff to {handoff_target}")
                break
            
            # Transfer control to the new agent.
            if target_agent:
                logger.info(f"[Orchestrator] Handoff to {handoff_target}")
                current_agent_name = target_agent.name
                
                # Get a message from handoff for the new agent
                next_message = self._extract_message_from_handoff(response.answer)
                if not next_message:
                    next_message = message
                
                # Transfer control to the next agent.
                response = await target_agent.handle_message(next_message, context)
                if not any(r.strip() == response.answer.strip() for r in replies):
                    replies.append(response.answer)
            else:
                logger.error(f"Unknown handoff target: {handoff_target}")
                break
        
        # Form the final "friendly" answer, combining all the answers received
        final_response = await self.combine_responses_friendly(replies)
        
        # Return the answer while saving the user's state
        return ChatResponse(
            answer=final_response,
            user_profile_id=self.user_profile_id,
            is_authorized=self.request.is_authorized,
            user_profile=user_profile
        )
    
    def _extract_message_from_handoff(self, handoff_message: str) -> Optional[str]:
        """
        Extract a message from Handoff response.
        
        Formathandoff: "[Handoff] target_agent communication_for_Transfers"
        """
        parts = handoff_message.split("[handoff]", 1)
        if len(parts) < 2:
            parts = handoff_message.split("[Handoff]", 1)
            if len(parts) < 2:
                return None
        
        # We separate the agent's name from the message.
        content = parts[1].strip()
        for agent_name in self.get_agents().keys():
            if content.lower().startswith(agent_name.lower()):
                return content[len(agent_name):].strip()
        
        # If the agent's name is not found, return the entire text after [handoff]]
        return content

    def get_current_stage(self, user_profile) -> str:
        """
        Identify the current user phase in terms of interaction
        Based on the completed fields in UserProfile.
        
        :param user_profile: Object UserProfile
        :return: Current stage line
        """
        if not user_profile:
            return "language_detection"
            
        # Checking for a Language of Communication
        if not user_profile.language:
            return "language_detection"
            
        # Checking the completion of the Triage (name, program area and purpose of the training) phase)
        triage_complete = (user_profile.user_name is not None and 
                          user_profile.program_area is not None and 
                          user_profile.education_goal is not None)
        if not triage_complete:
            return "triage"
            
        # Check program selection (suspect that this happened if there is program_area and education_goal)
        program_selected = (user_profile.program_area is not None and 
                           user_profile.education_goal is not None)
        if not program_selected:
            return "program_decision"
            
        # Checking whether the user is authorized
        is_authorized = False
        if hasattr(self, 'request') and self.request:
            is_authorized = self.request.is_authorized
        if not is_authorized:
            return "authorization"
            
        # Checking the completion of the LeadQualification phase (all additional fields filled))
        lead_qualified = all([
            user_profile.age is not None,
            user_profile.heard_about is not None,
            user_profile.after_study_goal is not None,
            user_profile.education_format is not None,
            user_profile.work_experience is not None,
            user_profile.additional_interests is not None,
            user_profile.selected_university is not None
        ])
        if not lead_qualified:
            return "lead_qualification"
            
        return "completed"

    def recommend_next_agent(self, intent: str) -> str:
        """
        Recommends the next agent based on the current user phase
        and identified incenter. It's a recommendatory system, not a strict requirement.
        
        :paramint: A certain user interest
        :return: Name of agent recommended
        """
        # Get current user profile
        user_profile = None
        if hasattr(self, 'request') and self.request:
            user_profile = self.request.user_profile
            
        # Defining the current phase
        current_stage = self.get_current_stage(user_profile)
        logger.info(f"[OrchestratorAgent] Current user phase: {current_stage}")
        
        # Priority steps to be taken in a sequence
        if current_stage == "language_detection":
            logger.info('[OrchestratorAgent] Recommended language agent')
            return "language_detector"
        
        if current_stage == "triage" and intent != "CommonQuestion":
            logger.info('[OrchestratorAgent] Recommended initial interview agent (triage))')
            return "triage"
            
        if current_stage == "program_decision" and intent in ["ProgramInfo", "ProgramDecision"]:
            logger.info('[OrchestratorAgent] Recommended program selection agent')
            return "program_decision"
            
        if current_stage == "authorization" and intent in ["ProgramInfo", "ProgramDecision", "Authorization"]:
            logger.info('[OrchestratorAgent] Recommended authorisation agent')
            return "authorization"
            
        if current_stage == "lead_qualification" and self.request.is_authorized:
            logger.info('[OrchestratorAgent] Recommended Lide Qualification Agent')
            return "lead_qualification"
        
        # By default, processing a request in accordance with the insider
        logger.info(f"[OrchestratorAgent] Recommended standard enzyme treatment: {intent}")
        return intent

    def get_plan_status(self, current_stage: str) -> Optional[str]:
        """
        Formulates the status of implementation of the plan based on the current phase.
        """
        stages = {
            "language_detection": '1. Definition of communication language',
            "triage": '2. Initial survey',
            "program_selection": '3. Selection of the curriculum',
            "authorization": '4. Registration and authorization',
            "lead_qualification": '5. Detailed discussion of the programme'
        }
        
        if current_stage == "completed":
            return None
            
        status_lines = []
        for stage, desc in stages.items():
            if stage == current_stage:
                status_lines.append(f"{desc} ← Current phase")
            elif stages[current_stage] > stages[stage]:
                status_lines.append(f"{desc} ✓")
            else:
                status_lines.append(desc)
                
        return '\n\nStatus of your progress:\n' + "\n".join(status_lines)

    def is_system_message(self, message: str) -> bool:
        """
        Determines whether the message is systematic.
        """
        message = message.lower().strip()
        
        # System markers
        system_markers = [
            "[handoff]",
            "[system]",
            "[debug]",
            "[error]",
            "[warning]",
            "[info]",
            "transfer_to_",
            "handoff to",
            "switching to",
            "initializing",
            "loading",
            "processing",
            "executing",
            "completed",
            "finished",
            "error occurred",
            "exception",
            "traceback"
        ]
        
        # Checking for system markers.
        if any(marker in message for marker in system_markers):
            return True
            
        # Check JSON format (technical data)
        try:
            json.loads(message)
            return True
        except:
            pass
            
        # Checking for technical details
        technical_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # Date
            r'\b\d{2}:\d{2}:\d{2}\b',  # time
            r'\b\d+\.\d+\.\d+\b',      # versions
            r'\b[a-f0-9]{32,}\b',      # Hey, hey, hey, hey, hey, hey.
            r'\b[A-Z_]+(?::[A-Z_]+)*\b' # constants
        ]
        
        if any(re.search(pattern, message) for pattern in technical_patterns):
            return True
            
        return False

    def process_handoff_message(self, message: str) -> str:
        """
        Processing handoff messages, replacing them with more friendly phrases.
        """
        if not message or "[handoff]" not in message.lower():
            return message
            
        # Type of Handoff
        parts = message.lower().split("[handoff]")
        if len(parts) < 2:
            return message
            
        handoff_type = parts[1].split()[0].strip()
        
        # Mapping handoff on friendly messages
        handoff_messages = {
            "authorization": 'Discuss the details of the registration.',
            "lead_qualification": 'To select the optimal program, I need to ask you a few questions.',
            "program_decision": "I'll help you find a suitable curriculum.",
            "common_questions": "I'll answer your questions.",
            "triage": 'Get to know each other.',
            "objection_detector": 'I would like to know your wishes and possible doubts.',
            "language_detector": "Let me detect your preferred language."
        }
        
        # We get a friendly message or we leave the original message
        friendly_message = handoff_messages.get(handoff_type, message)
        
        # If there is an additional text after handoff, add it
        additional_text = " ".join(parts[1].split()[1:]).strip()
        if additional_text and not any(marker in additional_text.lower() for marker in handoff_messages.values()):
            friendly_message = f"{friendly_message} {additional_text}"
            
        return friendly_message
