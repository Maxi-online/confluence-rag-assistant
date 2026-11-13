#!/usr/bin/env python3
"""
Скрипт для тестирования поиска в FAQ после загрузки данных в Weaviate.
Позволяет проверить разграничение доступа между абитуриентами и студентами.
"""

import os
import sys
import json
import logging
import argparse
import weaviate
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey
import weaviate.classes as wvc
from weaviate.collections.classes.filters import Filter


def _resolve_settings_env(value):
    """Resolve required environment references in already parsed configuration."""
    import os
    import re
    if isinstance(value, dict):
        return {key: _resolve_settings_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_settings_env(item) for item in value]
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([A-Z_][A-Z0-9_]*)(?::(int|float|bool))?\}", value)
    if match is None:
        return value
    name, kind = match.groups()
    content = os.environ.get(name)
    if content is None or not content.strip():
        raise RuntimeError(f"Required configuration variable is missing: {name}")
    if kind == "int":
        return int(content)
    if kind == "float":
        return float(content)
    if kind == "bool":
        if content.lower() not in {"true", "false", "1", "0"}:
            raise ValueError(f"Expected a boolean configuration variable: {name}")
        return content.lower() in {"true", "1"}
    return content



# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def search_faq(
    query: str,
    weaviate_client,
    is_authorized: bool = False,
    limit: int = 3,
    min_score: float = 0.5,
    class_name: str = "FAQ"
) -> str:
    """
    Поиск в FAQ с учетом авторизации пользователя.
    """
    try:
        # Получаем коллекцию
        collection = weaviate_client.collections.get(class_name)
        logger.info(f"Количество объектов в коллекции: {collection.aggregate.over_all().total_count}")
        
        # Создаем фильтр для типа пользователя, если пользователь не авторизован
        where_filter = None
        if not is_authorized:
            # Используем API v4 для создания фильтра
            where_filter = Filter.by_property("user_type").equal("Абитуриент")
        
        # Выполняем гибридный поиск с использованием нового API
        try:
            response = collection.query.hybrid(
                query=query,
                alpha=0.7,
                limit=limit,
                filters=where_filter,
                return_properties=["question", "answer", "topic", "user_type", "keywords"],
                return_metadata=wvc.query.MetadataQuery(score=True)
            )
        except Exception as e:
            error_msg = str(e)
            # Проверяем на наличие сообщения об ошибке от сервера OpenAI или Weaviate
            if "Технические работы" in error_msg or "invalid character '<'" in error_msg:
                logger.error("Сервис OpenAI API временно недоступен - технические работы на сервере.")
                return "СЕРВИС ВРЕМЕННО НЕДОСТУПЕН: На сервисе OpenAI API проводятся технические работы. Пожалуйста, попробуйте позже."
            # Пробрасываем остальные ошибки дальше
            raise
        
        # Обрабатываем результаты
        filtered_items = []
        for obj in response.objects:
            if obj.metadata.score >= min_score:
                filtered_items.append({
                    "question": obj.properties.get("question", ""),
                    "answer": obj.properties.get("answer", ""),
                    "topic": obj.properties.get("topic", ""),
                    "user_type": obj.properties.get("user_type", ""),
                    "keywords": obj.properties.get("keywords", []),
                    "_additional": {
                        "score": obj.metadata.score
                    }
                })
        
        logger.info(f"Найдено {len(filtered_items)} подходящих объектов")
        
        if not filtered_items:
            return ""
            
        # Формируем ответ
        response_parts = []
        for i, item in enumerate(filtered_items, 1):
            score = item["_additional"]["score"]
            keywords_str = ", ".join(item.get("keywords", []))
            
            response_parts.append(
                f"\nВопрос {i}: {item['question']}\n"
                f"Ответ: {item['answer']}\n"
                f"Тема: {item['topic']}\n"
                f"Тип пользователя: {item['user_type']}\n"
                f"Ключевые слова: {keywords_str}\n"
                f"Релевантность: {score:.2f}"
            )
            
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Ошибка при поиске в FAQ: {e}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        return ""

def main():
    parser = argparse.ArgumentParser(description='Тестирование поиска в FAQ')
    parser.add_argument('--query', type=str, required=True, help='Поисковый запрос')
    parser.add_argument('--authorized', action='store_true', help='Флаг авторизации (по умолчанию False - абитуриент)')
    parser.add_argument('--config', type=str, default='config.json', help='Путь к конфигурационному файлу')
    parser.add_argument('--weaviate-url', type=str, help='URL Weaviate сервера')
    parser.add_argument('--weaviate-api-key', type=str, help='API-ключ Weaviate')
    parser.add_argument('--openai-api-key', type=str, help='API-ключ OpenAI')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = _resolve_settings_env(json.load(f))
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
    
    # Приоритет: аргументы командной строки > конфигурационный файл > значения по умолчанию
    weaviate_url = args.weaviate_url or config.get("WEAVIATE_URL") or os.environ.get("WEAVIATE_URL")
    weaviate_api_key = args.weaviate_api_key or config.get("WEAVIATE_API_KEY") or os.environ.get("WEAVIATE_API_KEY")
    openai_api_key = args.openai_api_key or config.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if not all((weaviate_url, weaviate_api_key, openai_api_key)):
        parser.error("Configure WEAVIATE_URL, WEAVIATE_API_KEY and OPENAI_API_KEY through an approved local configuration or process environment")

    try:
        logger.info("Connecting to the explicitly configured Weaviate instance")
        
        # Подключаемся с использованием API версии 4
        client = connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(api_key=weaviate_api_key),
            headers={
                "X-OpenAI-Api-Key": openai_api_key,
                "X-OpenAI-BaseURL": "https://api.openai.com/v1/"
            }
        )
        
        # Получаем список коллекций
        collections = client.collections.list_all()
        logger.info(f"Тип collections: {type(collections)}")
        logger.info(f"Содержимое collections: {collections}")

        # В более новой версии API, collections может быть словарем
        try:
            # В версии 4 API collections возвращается как словарь с именами коллекций в качестве ключей
            if isinstance(collections, dict):
                collection_names = list(collections.keys())
            else:
                collection_names = [str(collections)]  # На случай, если формат совсем неожиданный
            
            logger.info(f"Подключение успешно. Доступные коллекции: {collection_names}")
        except Exception as e:
            logger.error(f"Ошибка при обработке списка коллекций: {e}")
            collection_names = []
        
        # Проверяем наличие коллекции FAQ
        if "FAQ" not in collection_names:
            logger.error(f"Коллекция FAQ не найдена в Weaviate!")
            print("Ошибка: Коллекция FAQ не найдена в базе данных. Сначала загрузите данные с помощью load_faq_to_weaviate.py")
            client.close()
            return 1
        
        logger.info(f"Коллекция FAQ найдена. Переходим к поиску.")
        logger.info(f"Поисковый запрос: '{args.query}'")
        logger.info(f"Тип пользователя: {'Студент (авторизован)' if args.authorized else 'Абитуриент (не авторизован)'}")
        
        # Выполняем поиск используя функцию search_faq
        results = search_faq(
            query=args.query,
            weaviate_client=client,
            is_authorized=args.authorized,
            limit=5,
            min_score=0.5,
            class_name="FAQ"
        )
        
        # Выводим результаты
        if results:
            print("\n=== РЕЗУЛЬТАТЫ ПОИСКА ===")
            print(results)
            print("\n=========================")
        else:
            print("\nПо вашему запросу ничего не найдено.")
        
        # Закрываем соединение
        client.close()
        
    except Exception as e:
        logger.error(f"Ошибка при подключении или выполнении поиска: {e}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        print(f"Произошла ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
