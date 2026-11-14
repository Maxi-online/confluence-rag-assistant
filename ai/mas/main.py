#!/usr/bin/env python3
"""
The main file for launching the GPT-agent system.
The function accepts queries and transfers the control to the right agent on the basis of analysis.
"""

import sys
import asyncio
import json
import logging
import traceback
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ai.mas.utils.graph_base import WeaviateClient
from ai.mas.utils.config import (
    USER_ID,
    OPENAI_API_KEY,
    WEAVIATE_URL,
    WEAVIATE_API_KEY,
    EXPECTED_DIM,
    CONFLUENCE_BASE_URL,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_PAGE_ID,
    MAX_CHUNK_SIZE,
    MIN_SCORE,
    HYBRID_ALPHA,
    CLASS_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class UserProfile(BaseModel):
    'User profile model'
    user_profile_id: int = Field(..., description='Unique user profile identifier')
    language: Optional[str] = Field(None, description='Language of association with bot')
    user_name: Optional[str] = Field(None, description='Name given for circulation')
    program_area: Optional[str] = Field(None, description='Programme area')
    education_goal: Optional[str] = Field(None, description='Purpose of training')
    education_objections: Optional[str] = Field(None, description='Objections to learning')
    age: Optional[int] = Field(None, description='Age of user')
    heard_about: Optional[str] = Field(None, description='How did you know about us?')
    after_study_goal: Optional[str] = Field(None, description='Target after training')
    education_format: Optional[str] = Field(None, description='Training format')
    work_experience: Optional[str] = Field(None, description='Experience')
    additional_interests: Optional[str] = Field(None, description='Additional interests')
    selected_university: Optional[str] = Field(None, description='The chosen school.')
    is_modified: bool = Field(False, description='Profile modification flag')

class ChatRequest(BaseModel):
    'The chat request model'
    message: str = Field(..., description='User message')
    user_profile_id: str = Field(default=USER_ID, description='User Identifier')
    openai_api_key: str = Field(default=OPENAI_API_KEY, description='OpenAI key')
    weaviate_url: str = Field(default=WEAVIATE_URL, description='URL Weaviate server')
    weaviate_api_key: str = Field(default=WEAVIATE_API_KEY, description='API Key')
    expected_dim: int = Field(default=EXPECTED_DIM, description='Expected number of measurements in vectorization')
    confluence_base_url: str = Field(default=CONFLUENCE_BASE_URL, description='Core URL for Confluence')
    confluence_api_token: str = Field(default=CONFLUENCE_API_TOKEN, description='API-togen Confluence')
    confluence_page_id: int = Field(default=CONFLUENCE_PAGE_ID, description='Confluence ID page')
    max_chunk_size: int = Field(default=MAX_CHUNK_SIZE, description='Maximum size of the text breaker')
    min_score: float = Field(default=MIN_SCORE, description='Minimum threshold of relevance')
    hybrid_alpha: float = Field(default=HYBRID_ALPHA, description='hybrid_alpha Parameter for Search')
    is_authorized: bool = Field(default=False, description='User authorisation flag')
    user_profile: Optional[UserProfile] = Field(default=None, description='User Profile')

class ChatResponse(BaseModel):
    'The chat response model'
    answer: str = Field(..., description='GPT Agent System Response')
    user_profile_id: str = Field(..., description='User Identifier')
    is_authorized: bool = Field(..., description='User authorisation flag')
    user_profile: Optional[UserProfile] = Field(default=None, description='User Profile')

async def process_message(request: ChatRequest) -> ChatResponse:
    """
    Processing the incoming message and returning the answer.
    
    Args:
        request (ChatRequest): Message and configuration request
        
    Returns:
        ChatResponse: Response of the system while maintaining all fields of request
    """
    # Import OrchestratorAgent inside the function to avoid cyclical dependence
    from ai.mas.agents.orchestrator_agent import OrchestratorAgent
    
    user_input = request.message.strip()
    if not user_input:
        return ChatResponse(
            answer='The message should not be empty.',
            user_profile_id=request.user_profile_id,
            is_authorized=request.is_authorized,
            user_profile=request.user_profile
        )
    
    try:
        # Initiating client to work with Weaviate
        with WeaviateClient(
            openai_api_key=request.openai_api_key,
            weaviate_url=request.weaviate_url,
            weaviate_api_key=request.weaviate_api_key,
            class_name=CLASS_NAME,
            expected_dim=request.expected_dim,
            confluence_base_url=request.confluence_base_url,
            confluence_api_token=request.confluence_api_token,
            confluence_page_id=request.confluence_page_id,
            max_chunk_size=request.max_chunk_size,
            min_score=request.min_score,
            hybrid_alpha=request.hybrid_alpha
        ) as knowledge_client:
            # Create a copy of OrchestratorAgent by transmitting the entire object request
            orchestrator = OrchestratorAgent(
                request=request,
                knowledge_client=knowledge_client
            )

            # Get an answer from the orchestra.
            response = await orchestrator.handle_message(user_input)
            
            # Check that the response is ChatResponse
            if not isinstance(response, ChatResponse):
                raise ValueError(f"Expected ChatResponse, got {type(response)}")
            
            # Logize the length of the answer
            logging.info(f"[Main] Successfully processed request. Answer length: {len(response.answer)}")
            
            # Return the response in ChatResponse
            return response

    except Exception as e:
        logging.error(f"Error in process_message: {e}\n{traceback.format_exc()}")
        return ChatResponse(
            answer='There was an error in processing the request',
            user_profile_id=request.user_profile_id,
            is_authorized=request.is_authorized,
            user_profile=request.user_profile
        )
