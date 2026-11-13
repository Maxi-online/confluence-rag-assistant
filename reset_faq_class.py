#!/usr/bin/env python3
"""
Скрипт для удаления и пересоздания класса FAQ в Weaviate с правильными настройками векторизатора.
"""

import os
import logging
import argparse
import weaviate
from weaviate import connect_to_weaviate_cloud, connect_to_local
from weaviate.auth import AuthApiKey

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Параметры по умолчанию
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CLASS_NAME = "FAQ"
EXPECTED_DIM = 3072

def reset_faq_class(force: bool = False):
    """
    Удаляет класс FAQ из Weaviate, если он существует, и создает его заново с правильными настройками.
    
    :param force: Принудительное удаление без подтверждения
    """
    try:
        # Подключаемся к Weaviate
        client = connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
        )
        
        # Проверяем существование класса
        class_exists = False
        try:
            collection = client.collections.get(CLASS_NAME)
            class_exists = True
            logger.info(f"Класс {CLASS_NAME} найден в Weaviate.")
        except Exception:
            logger.info(f"Класс {CLASS_NAME} не найден в Weaviate.")
        
        if class_exists:
            if not force:
                confirm = input(f"Вы уверены, что хотите удалить класс {CLASS_NAME}? (y/n): ")
                if confirm.lower() != 'y':
                    logger.info("Операция отменена пользователем.")
                    return
            
            # Удаляем класс
            try:
                client.collections.delete(CLASS_NAME)
                logger.info(f"Класс {CLASS_NAME} успешно удален из Weaviate.")
            except Exception as e:
                logger.error(f"Ошибка при удалении класса {CLASS_NAME}: {e}")
                return
        
        # Создаем класс FAQ с правильными настройками
        schema_obj = {
            "class": CLASS_NAME,
            "description": "База знаний FAQ с разделением по типу пользователя",
            "vectorizer": "text2vec-openai",  # Указываем векторизатор
            "vectorizerConfig": {
                "model": "text-embedding-3-large",
                "dimensions": EXPECTED_DIM,
                "type": "text"
            },
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "text-embedding-3-large",
                    "dimensions": EXPECTED_DIM,
                    "type": "text"
                }
            },
            "properties": [
                {"name": "question", "dataType": ["text"], "description": "Вопрос пользователя", "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "answer", "dataType": ["text"], "description": "Ответ на вопрос", "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "keywords", "dataType": ["text[]"], "description": "Ключевые слова для поиска", "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "topic", "dataType": ["text"], "description": "Тема запроса", "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "main_point", "dataType": ["text"], "description": "Основной момент", "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "user_type", "dataType": ["text"], "description": "Тип пользователя (Абитуриент/Студент)", "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "implementation_period", "dataType": ["text"], "description": "Срок реализации", "moduleConfig": {"text2vec-openai": {"skip": True}}}
            ]
        }
        
        # Создаем класс с новой схемой используя новый API
        client.collections.create_from_dict(schema_obj)
        logger.info(f"Класс {CLASS_NAME} успешно создан с правильными настройками векторизатора.")
        
        # Закрываем соединение
        client.close()
        
    except Exception as e:
        logger.error(f"Ошибка при работе с Weaviate: {e}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        try:
            client.close()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='Сброс класса FAQ в Weaviate')
    parser.add_argument('--force', action='store_true', help='Принудительное удаление без подтверждения')
    
    args = parser.parse_args()
    reset_faq_class(args.force)

if __name__ == "__main__":
    main() 