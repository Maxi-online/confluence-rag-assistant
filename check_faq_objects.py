#!/usr/bin/env python3
"""
Скрипт для проверки количества объектов в классе FAQ в Weaviate.
"""
import os

import logging
import weaviate
from weaviate import connect_to_weaviate_cloud, connect_to_local
from weaviate.auth import AuthApiKey

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Параметры подключения
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "FAQ"

def main():
    """
    Главная функция для проверки количества объектов.
    """
    try:
        # Подключаемся к Weaviate
        client = connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY)
        )
        
        logger.info(f"Подключение к Weaviate установлено: {WEAVIATE_URL}")
        
        try:
            # Проверяем существование класса путем попытки получить коллекцию
            collection = client.collections.get(CLASS_NAME)
            logger.info(f"Класс {CLASS_NAME} существует")
            
            # Получаем количество объектов
            total_count = collection.aggregate.over_all().total_count
            logger.info(f"Количество объектов в классе {CLASS_NAME}: {total_count}")
            
            # Получаем примеры объектов
            objects = collection.query.fetch_objects(limit=5)
            logger.info(f"Получено {len(objects)} объектов для проверки")
            
            if objects:
                for i, obj in enumerate(objects):
                    logger.info(f"\nОбъект #{i+1}:")
                    for key, value in obj.properties.items():
                        logger.info(f"  {key}: {value}")
            else:
                logger.info("Объекты отсутствуют")
        except Exception as class_error:
            logger.error(f"Класс {CLASS_NAME} не существует или не может быть получен: {class_error}")
        
        # Проверяем все классы
        try:
            all_collections = client.collections.list_all()
            logger.info(f"Все классы в Weaviate: {[c.name for c in all_collections]}")
        except Exception as list_error:
            logger.error(f"Ошибка при получении списка классов: {list_error}")
        
        # Закрываем соединение
        client.close()
        logger.info("Проверка завершена")
        
    except Exception as e:
        logger.error(f"Ошибка при проверке объектов: {e}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 