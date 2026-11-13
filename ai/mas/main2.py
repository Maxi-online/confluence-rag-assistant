#!/usr/bin/env python3
"""
Основной файл для запуска системы GPT-агентов без FastAPI.
При запуске скрипт ожидает ввод сообщения пользователя, обрабатывает его через Orchestrator Agent
и выводит итоговый ответ.
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

# Создаём централизованный клиент для доступа к Weaviate с дополнительными параметрами для Confluence
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

# Создаём главного диспетчера (Orchestrator)
orchestrator = OrchestratorAgent(
    user_id=USER_ID,
    openai_api_key=OPENAI_API_KEY,
    knowledge_client=knowledge_client
)

async def process_message(message: str) -> None:
    """
    Обрабатывает сообщение пользователя через Orchestrator и выводит ответ.
    """
    answer = await orchestrator.handle_message(message)
    # Если ответ не строка, приводим его к строке
    if not isinstance(answer, str):
        if isinstance(answer, dict) and "answer" in answer:
            answer = answer["answer"]
        else:
            try:
                answer = json.dumps(answer, ensure_ascii=False)
            except Exception:
                answer = str(answer)
    print("Ответ:", answer)

def main():
    # Если сообщение передано как аргумент командной строки, используем его, иначе запрашиваем ввод
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:]).strip()
    else:
        user_message = input("Введите сообщение: ").strip()
    
    if not user_message:
        print("Сообщение не должно быть пустым.")
        sys.exit(1)
    
    asyncio.run(process_message(user_message))

if __name__ == "__main__":
    main()
