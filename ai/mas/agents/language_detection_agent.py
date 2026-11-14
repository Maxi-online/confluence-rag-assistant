import logging
from openai import AsyncOpenAI
from pydantic import ConfigDict
from typing import Any, List, Optional

from ai.mas.utils.db import update_user_profile, update_language_in_context, get_user_profile
from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.main import ChatResponse, UserProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDetectionAgent(BaseAgent):
    """
    A language agent.
    
    Logic:
      1. On first launch (if no language is available in the profile))
         – The agent asks, "What language is it convenient for you to communicate in?"
      2. In subsequent calls, the agent does not ask the question again and simply sends the message further.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(
        self,
        knowledge_client: Any,
        user_profile_id: str,
        openai_api_key: str,
        openai_base_url: str = "https://api.openai.com/v1/",
        request: Any = None,
        **kwargs
    ):
        super().__init__(
            name="Language Detection Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the LANGUAGE DETECTION AGENT. Your mission is to determine the language of the user's input quickly and accurately.

            Instructions:
            1. Analyze the input text using internal Chain‑of‑Thought reasoning.
            2. NEVER output the detected language to the user directly. Instead, save it internally.
            3. If multiple languages are detected, prioritize the main language.
            4. Handle the language detection silently in the background.
            5. Do not include any language detection details in your output to the user.
            6. After determining the language, use the transfer_to_orchestrator function to continue the conversation.
            7. Your response should be minimal and focused on continuing the conversation flow.

            Output: A simple message to handoff control back to the orchestrator without mentioning language detection.
            """
            ),
            tools=[self.transfer_to_orchestrator],
            **kwargs
        )
        self.user_profile_id = user_profile_id
        self.knowledge_client = knowledge_client
        self.gpt_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.request = request

    async def transfer_to_orchestrator(self, message: str = "") -> Agent:
        """
        Transfer control back to the Orchestrator with the user's original message.
        
        Args:
            message: Original user message to pass to the Orchestrator
        """
        from ai.mas.agents.orchestrator_agent import OrchestratorAgent
        logger.info(f"[LanguageDetectionAgent] Transferring to Orchestrator with message: {message}")
        return None

    async def handle_message(self, message: str, context: Optional[List[dict]] = None) -> ChatResponse:
        """
        Process user message and define communication language.
        
        Logic:
        1. If the language is already defined, it just goes on with the administration.
        2. If the message contains a reference to the language, it retains it and transmits it further.
        3. Otherwise, it asks the user about the preferred language.
        """
        logger.info(f"\n{'='*80}\nENE AGENCY OPENING AGENCY")
        logger.info(f"Incoming message: {message}")
        
        # Check the profile from the request
        if not hasattr(self, 'request') or not self.request or not self.request.user_profile:
            logger.error('Error: No request or user profile')
            return ChatResponse(
                answer='Sorry, there was a mistake in defining the language.',
                user_profile_id=self.user_profile_id,
                is_authorized=False,
                user_profile=None
            )

        logger.info('\nCURRENT STATUS OF THE PROFILE:')
        logger.info(f"user_profile_id: {self.request.user_profile.user_profile_id}")
        logger.info(f"language: {self.request.user_profile.language}")
        logger.info(f"is_modified: {self.request.user_profile.is_modified}")

        # If the language is already defined, we'll forward the original message through handoff
        if self.request.user_profile.language is not None:
            logger.info(f"[LanguageDetectionAgent] Language already defined.{self.request.user_profile.language}). I'm handing over the orchestra control.")
            return ChatResponse(
                answer=f"[Handoff] orchestrator {message}",
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )

        # Check if the message is the answer to the question about the language
        if any(word.lower() in message.lower() for word in ['Russian', 'Russian', 'Russian', '(Signed) R.R.']):
            language = "ru"
            logger.info(f"Language defined: {language}")
            
            # Keep the language in the profile.
            logger.info('\nUPDATING PROFILE:')
            logger.info(f"Old meaning of language: {self.request.user_profile.language}")
            logger.info(f"Old value is_modified: {self.request.user_profile.is_modified}")
            
            self.request.user_profile.language = language
            self.request.user_profile.is_modified = True
            
            logger.info(f"New meaning of language: {self.request.user_profile.language}")
            logger.info(f"New value of is_modified: {self.request.user_profile.is_modified}")
            
            # Checking if there's a user name in the profile.
            if self.request.user_profile.user_name is None:
                # If there's no name, transfer control to TriageAgent.
                logger.info(f"\nIM OF THE USER NOT DEFINIATED. TRANSFER OF THE TRIAGE AGENT MANAGEMENT")
                return ChatResponse(
                    answer=f"[Handoff] triage",
                    user_profile_id=self.request.user_profile_id,
                    is_authorized=self.request.is_authorized,
                    user_profile=self.request.user_profile
                )
            
            # If there's a name, we'll hand over the orchestra control.
            logger.info(f"\n PERFORMANCE OF THE MANAGEMENT OF THE ORAXESTRATOR")
            
            return ChatResponse(
                answer=f"[Handoff] orchestrator",
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )
        elif any(word.lower() in message.lower() for word in ["english", "eng", 'English', 'English']):
            language = "en"
            logger.info(f"Language defined: {language}")
            
            # Keep the language in the profile.
            logger.info('\nUPDATING PROFILE:')
            logger.info(f"Old meaning of language: {self.request.user_profile.language}")
            logger.info(f"Old value is_modified: {self.request.user_profile.is_modified}")
            
            self.request.user_profile.language = language
            self.request.user_profile.is_modified = True
            
            logger.info(f"New meaning of language: {self.request.user_profile.language}")
            logger.info(f"New value of is_modified: {self.request.user_profile.is_modified}")
            
            # Checking if there's a user name in the profile.
            if self.request.user_profile.user_name is None:
                # If there's no name, transfer control to TriageAgent.
                logger.info(f"\nIM OF THE USER NOT DEFINIATED. TRANSFER OF THE TRIAGE AGENT MANAGEMENT")
                return ChatResponse(
                    answer=f"[Handoff] triage",
                    user_profile_id=self.request.user_profile_id,
                    is_authorized=self.request.is_authorized,
                    user_profile=self.request.user_profile
                )
            
            # If there's a name, we'll hand over the orchestra control.
            logger.info(f"\n PERFORMANCE OF THE MANAGEMENT OF THE ORAXESTRATOR")
            
            return ChatResponse(
                answer=f"[Handoff] orchestrator",
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )
        else:
            # Trying to automatically define the message language if this is the first communication
            if len(message) > 10:  # If the message is long enough for analysis
                # Simple analysis: if there is more Cyrillic, Russian, or English
                cyrillic_count = sum(1 for char in message if 'а' <= char.lower() <= 'я')
                latin_count = sum(1 for char in message if 'a' <= char.lower() <= 'z')
                
                if cyrillic_count > latin_count:
                    self.request.user_profile.language = "ru"
                    self.request.user_profile.is_modified = True
                    logger.info(f"Automatically defined language: ru")
                    return ChatResponse(
                        answer=f"[Handoff] orchestrator {message}",
                        user_profile_id=self.request.user_profile_id,
                        is_authorized=self.request.is_authorized,
                        user_profile=self.request.user_profile
                    )
            
            # Ask about the preferred language
            logger.info('[LanguageDetectionAgent]')
            return ChatResponse(
                answer='What language is more convenient for you to communicate?',
                user_profile_id=self.request.user_profile_id,
                is_authorized=self.request.is_authorized,
                user_profile=self.request.user_profile
            )
