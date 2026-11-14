#!/usr/bin/env python3
"""
Basic file for launching GPT agents without FastAPI.
The script is expected to enter the user message when it is launched, processing it through the Orchestrator Agent
And then he gives the final answer.
"""

import sys
import asyncio
import json
import logging

from ai.mas.agents.orchestrator_agent import OrchestratorAgent
from ai.mas.utils.config import (
    USER_ID,
    OPENAI_API_KEY,
    WEAVIATE_URL,
    WEAVIATE_API_KEY,
    CLASS_NAME,
    EXPECTED_DIM,
    CONFLUENCE_BASE_URL,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_PAGE_ID,
    HIERARCHY_FILE,
    MAX_CHUNK_SIZE,
    MIN_SCORE,
    HYBRID_ALPHA
)
from ai.mas.utils.graph_base import WeaviateClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Create a centralized client to access the Web with additional parameters for Conference
knowledge_client = WeaviateClient(
    openai_api_key=OPENAI_API_KEY,
    weaviate_url=WEAVIATE_URL,
    weaviate_api_key=WEAVIATE_API_KEY,
    class_name=CLASS_NAME,
    expected_dim=EXPECTED_DIM,
    confluence_base_url=CONFLUENCE_BASE_URL,
    confluence_api_token=CONFLUENCE_API_TOKEN,
    confluence_page_id=CONFLUENCE_PAGE_ID,
    hierarchy_file=HIERARCHY_FILE,
    max_chunk_size=MAX_CHUNK_SIZE,
    min_score=MIN_SCORE,
    hybrid_alpha=HYBRID_ALPHA
)

# We're setting up an Orchestrator.
orchestrator = OrchestratorAgent(
    user_id=USER_ID,
    openai_api_key=OPENAI_API_KEY,
    knowledge_client=knowledge_client
)

async def process_message(message: str) -> None:
    """
    Process user message via Orchestrator and output response.
    """
    answer = await orchestrator.handle_message(message)
    # If the answer is not a line, insert it to the line
    if not isinstance(answer, str):
        if isinstance(answer, dict) and "answer" in answer:
            answer = answer["answer"]
        else:
            try:
                answer = json.dumps(answer, ensure_ascii=False)
            except Exception:
                answer = str(answer)
    print('Answer:', answer)

def main():
    # If the message is transmitted as a command line argument, use it, otherwise request entry
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:]).strip()
    else:
        user_message = input('Enter message: ').strip()
    
    if not user_message:
        print("The message shouldn't be empty.")
        sys.exit(1)
    
    asyncio.run(process_message(user_message))

if __name__ == "__main__":
    main()
