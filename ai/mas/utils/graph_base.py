"""
Модуль для подключения к Weaviate, создания схем и выполнения запросов с поддержкой multi-vector embeddings.
Если по запросу не найдено релевантной информации, инициируется дообогащение базы знаний через Confluence,
после чего поиск повторяется.
"""

import os
import logging
import asyncio
import json
from dataclasses import asdict
from typing import List, Optional, Any, Dict
import weaviate
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter  # Используем Filter согласно документации

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateClient:
    def __init__(
        self,
        openai_api_key: str,
        weaviate_url: str,
        weaviate_api_key: str,
        class_name: str,
        expected_dim: int,
        confluence_base_url: str,
        confluence_api_token: str,
        confluence_page_id: int,
        max_chunk_size: int,
        min_score: float = 0.7,
        hybrid_alpha: float = 0.7
    ):
        """
        Инициализирует клиента Weaviate с параметрами для дообогащения базы знаний через Confluence.
        """
        self.openai_api_key = openai_api_key
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.class_name = class_name
        self.expected_dim = expected_dim
        self.confluence_base_url = confluence_base_url
        self.confluence_api_token = confluence_api_token
        self.confluence_page_id = confluence_page_id
        self.max_chunk_size = max_chunk_size
        self.min_score = min_score
        self.hybrid_alpha = hybrid_alpha

        # Подключаемся к облачному Weaviate
        self.weaviate_client = connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=AuthApiKey(api_key=self.weaviate_api_key),
            headers={"X-OpenAI-Api-Key": self.openai_api_key},
            skip_init_checks=True
        )
        self.ensure_schema_exists()

    def close(self) -> None:
        """
        Закрывает соединение с Weaviate.
        """
        if hasattr(self, 'weaviate_client'):
            self.weaviate_client.close()
            logger.info("Соединение с Weaviate закрыто.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def ensure_schema_exists(self) -> None:
        """
        Проверяет, существует ли указанный класс (схема) в Weaviate. Если нет – создаёт его.
        """
        try:
            self.weaviate_client.collections.get(self.class_name)
            logger.info(f"Класс {self.class_name} уже существует.")
        except Exception:
            logger.info(f"Класс {self.class_name} не найден. Создаю схему...")
            schema_obj = {
                "class": self.class_name,
                "description": "База знаний для поиска контекста.",
                "vectorizer": "text2vec-openai",
                "vectorizerConfig": {
                    "name": "content_vector",
                    "sourceProperties": ["title", "content"],
                    "model": "text-embedding-3-large",
                    "dimensions": self.expected_dim
                },
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "level", "dataType": ["int"]},
                    {"name": "pageId", "dataType": ["text"]},
                    {"name": "lastUpdated", "dataType": ["date"]},
                    {"name": "parentSection", "dataType": [self.class_name]},
                    {"name": "childSections", "dataType": [self.class_name]}
                ]
            }
            self.weaviate_client.collections.create_from_dict(schema_obj)
            logger.info(f"Класс {self.class_name} успешно создан (dimensions={self.expected_dim}).")

    def search_documents(self, user_question: str) -> str:
        """
        Полностью реализует логику поиска и фильтрации документов в Weaviate.

        :param user_question: Запрос пользователя.
        :return: Отрывки документов, удовлетворяющих запросу, или пустая строка, если релевантные документы не найдены.
        """
        try:
            safe_query = user_question.replace('"', '\\"')
            query = (
                f'{{\n'
                f'  Get {{\n'
                f'    {self.class_name}(\n'
                f'      hybrid: {{\n'
                f'        query: "{safe_query}",\n'
                f'        alpha: {self.hybrid_alpha}\n'
                f'      }},\n'
                f'      limit: 10\n'
                f'    ) {{\n'
                f'      title\n'
                f'      content\n'
                f'      _additional {{\n'
                f'        score\n'
                f'      }}\n'
                f'    }}\n'
                f'  }}\n'
                f'}}'
            )
            
            # Выполняем запрос с использованием GraphQL
            try:
                result = self.weaviate_client.graphql_raw_query(query)
                logger.info(f"Documents search raw result: {result}")
            except Exception as e:
                logger.error(f"Ошибка при выполнении поиска в Weaviate: {e}")
                return ""
            
            # Извлекаем результаты из ответа с учетом различных возможных форматов
            try:
                # Правильная обработка результата _RawGQLReturn
                if hasattr(result, 'get') and callable(result.get):
                    # Если это _RawGQLReturn с методом get
                    docs = result.get('data', {}).get('Get', {}).get(self.class_name, [])
                elif hasattr(result, 'get') and hasattr(result, 'errors'):
                    # Проверяем наличие ошибок
                    if result.errors is not None and len(result.errors) > 0:
                        logger.error(f"GraphQL вернул ошибки: {result.errors}")
                        return ""
                    # Пробуем извлечь данные напрямую из атрибута
                    if hasattr(result, 'get') and isinstance(result.get, dict):
                        docs = result.get.get(self.class_name, [])
                    else:
                        # Для совместимости с предыдущими версиями API
                        result_dict = asdict(result) if hasattr(result, '__dict__') else result
                        docs = result_dict.get("data", {}).get("Get", {}).get(self.class_name, [])
                else:
                    # Обычная обработка словаря
                    result_dict = result
                    docs = result_dict.get("data", {}).get("Get", {}).get(self.class_name, [])
                
                # Проверяем, получили ли мы данные
                if not docs and hasattr(result, 'get') and isinstance(result.get, dict):
                    # Если docs пустой, но у нас есть данные в result.get
                    docs = result.get.get(self.class_name, [])
                
                logger.info(f"Found documents: {docs}")
                
                # Если документы все еще не найдены, попробуем другой способ извлечения
                if not docs and isinstance(result, object) and hasattr(result, 'get'):
                    # Прямой доступ к атрибуту как записано в логах
                    logger.info("Пробуем извлечь документы напрямую из атрибута")
                    try:
                        if hasattr(result, 'get') and result.get and self.class_name in result.get:
                            docs = result.get[self.class_name]
                            logger.info(f"Извлечено напрямую: {docs}")
                    except Exception as e:
                        logger.error(f"Ошибка при попытке прямого извлечения: {e}")
                
            except Exception as e:
                logger.error(f"Ошибка при извлечении результатов поиска: {e}")
                return ""
            
            # Фильтруем результаты по релевантности
            filtered_docs = []
            for doc in docs:
                try:
                    score = float(doc.get("_additional", {}).get("score", 0))
                    if score >= self.min_score:
                        filtered_docs.append(doc)
                except (TypeError, ValueError) as e:
                    logger.error(f"Ошибка обработки score для документа {doc}: {e}")
            
            if not filtered_docs:
                logger.info("Нет релевантных документов")
                return ""
            
            # Формируем ответ
            response_parts = []
            for i, doc in enumerate(filtered_docs, 1):
                response_parts.append(
                    f"\nОтрывок документа №{i}\n"
                    f"=====================\n"
                    f"Заголовок: {doc.get('title', '').strip()}\n"
                    f"Содержимое: {doc.get('content', '').strip()}\n"
                    f"(Score: {doc.get('_additional', {}).get('score', 0)})"
                )
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return ""

    async def search_documents_async(self, query: str) -> str:
        return await asyncio.to_thread(self.search_documents, query)

    def rerank_documents(self, user_question: str, docs: List[dict]) -> List[dict]:
        return sorted(docs, key=lambda d: float(d.get("_additional", {}).get("score", 0)), reverse=True)

    def autocut_documents(self, docs: List[dict]) -> List[dict]:
        return docs[:3]

    def search_local_knowledge(self, user_question: str) -> str:
        """
        Полностью реализует логику поиска и фильтрации документов в Weaviate.

        :param user_question: Запрос пользователя.
        :return: Отрывки документов, удовлетворяющих запросу, или пустая строка, если релевантные документы не найдены.
        """
        safe_query = user_question.replace('"', '\\"')
        query = (
            f'{{\n'
            f'  Get {{\n'
            f'    {self.class_name}(\n'
            f'      hybrid: {{\n'
            f'        query: "{safe_query}",\n'
            f'        alpha: {self.hybrid_alpha}\n'
            f'      }},\n'
            f'      limit: 10\n'
            f'    ) {{\n'
            f'      title\n'
            f'      content\n'
            f'      _additional {{\n'
            f'        score\n'
            f'      }}\n'
            f'    }}\n'
            f'  }}\n'
            f'}}'
        )
        try:
            raw_result = self.weaviate_client.graphql_raw_query(query)
            logging.info(f"Raw result: {raw_result}")
        except Exception as e:
            logging.error(f"Ошибка при выполнении поиска в Weaviate: {e}")
            return ""
        try:
            if isinstance(raw_result, dict):
                result_dict = raw_result
            else:
                result_dict = asdict(raw_result)
            docs = result_dict.get("data", {}).get("Get", {}).get(self.class_name, [])
            logging.info(f"Docs: {docs}")
        except Exception as e:
            logging.error(f"Ошибка при извлечении результатов: {e}")
            return ""
        filtered_docs = [
            doc for doc in docs if float(doc.get("_additional", {}).get("score", 0)) >= self.min_score
        ]
        if not filtered_docs:
            return ""
        reranked_docs = self.rerank_documents(user_question, filtered_docs)
        final_docs = self.autocut_documents(reranked_docs)
        question_tokens = set(user_question.lower().split())
        def is_relevant(doc: dict) -> bool:
            text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
            return any(token in text for token in question_tokens)
        if not any(is_relevant(doc) for doc in final_docs):
            logging.info("Нет релевантных документов, возвращаю пустой контекст.")
            return ""
        return "\n".join([
            f"\nОтрывок документа №{i+1}\n=====================\nЗаголовок: {doc.get('title', '').strip()}\nСодержимое: {doc.get('content', '').strip()}\n(Score: {doc.get('_additional', {}).get('score', 0)})"
            for i, doc in enumerate(final_docs)
        ])

    def search_faq(self, query: str, is_authorized: bool = False, limit: int = 3) -> str:
        """
        Поиск в FAQ с учетом авторизации пользователя.
        
        :param query: Запрос пользователя
        :param is_authorized: Флаг авторизации (True для студентов, False для абитуриентов)
        :param limit: Максимальное количество результатов
        :return: Строка с результатами поиска
        """
        try:
            # Подготовка запроса с учетом типа пользователя
            class_name = "FAQ"
            safe_query = query.replace('"', '\\"')
            
            # Формируем GraphQL-запрос с правильной фильтрацией
            # Для неавторизованных пользователей показываем только FAQ с типом "Абитуриент"
            if not is_authorized:
                # Правильный формат фильтрации в GraphQL для Weaviate
                graphql_query = f"""
                {{
                  Get {{
                    {class_name}(
                      hybrid: {{
                        query: "{safe_query}",
                        alpha: {self.hybrid_alpha}
                      }},
                      where: {{
                        path: ["user_type"],
                        operator: Equal,
                        valueString: "Абитуриент"
                      }},
                      limit: {limit}
                    ) {{
                      question
                      answer
                      topic
                      keywords
                      user_type
                      _additional {{
                        score
                      }}
                    }}
                  }}
                }}
                """
            else:
                # Для авторизованных пользователей не применяем фильтр по типу
                graphql_query = f"""
                {{
                  Get {{
                    {class_name}(
                      hybrid: {{
                        query: "{safe_query}",
                        alpha: {self.hybrid_alpha}
                      }},
                      limit: {limit}
                    ) {{
                      question
                      answer
                      topic
                      keywords
                      user_type
                      _additional {{
                        score
                      }}
                    }}
                  }}
                }}
                """
            
            # Выполняем запрос с использованием GraphQL
            try:
                result = self.weaviate_client.graphql_raw_query(graphql_query)
                logger.info(f"FAQ search raw result: {result}")
            except Exception as e:
                logger.error(f"Ошибка при выполнении поиска в FAQ: {e}")
                return ""
            
            # Извлекаем результаты из ответа с учетом различных возможных форматов ответа
            try:
                # Правильная обработка результата _RawGQLReturn
                if hasattr(result, 'get') and callable(result.get):
                    # Если это _RawGQLReturn с методом get
                    faq_items = result.get('data', {}).get('Get', {}).get(class_name, [])
                elif hasattr(result, 'get') and hasattr(result, 'errors'):
                    # Проверяем наличие ошибок
                    if result.errors is not None and len(result.errors) > 0:
                        logger.error(f"GraphQL вернул ошибки: {result.errors}")
                        return ""
                    # Пробуем извлечь данные напрямую из атрибута
                    if hasattr(result, 'get') and isinstance(result.get, dict):
                        faq_items = result.get.get(class_name, [])
                    else:
                        # Для совместимости с предыдущими версиями API
                        result_dict = asdict(result) if hasattr(result, '__dict__') else result
                        faq_items = result_dict.get("data", {}).get("Get", {}).get(class_name, [])
                else:
                    # Обычная обработка словаря
                    result_dict = result
                    faq_items = result_dict.get("data", {}).get("Get", {}).get(class_name, [])
                
                # Проверяем, получили ли мы данные
                if not faq_items and hasattr(result, 'get') and isinstance(result.get, dict):
                    # Если faq_items пустой, но у нас есть данные в result.get
                    faq_items = result.get.get(class_name, [])
                
                logger.info(f"FAQ Items: {faq_items}")
                
                # Если элементы все еще не найдены, попробуем другой способ извлечения
                if not faq_items and isinstance(result, object) and hasattr(result, 'get'):
                    # Прямой доступ к атрибуту как записано в логах
                    logger.info("Пробуем извлечь FAQ напрямую из атрибута")
                    try:
                        if hasattr(result, 'get') and result.get and class_name in result.get:
                            faq_items = result.get[class_name]
                            logger.info(f"Извлечено напрямую: {faq_items}")
                    except Exception as e:
                        logger.error(f"Ошибка при попытке прямого извлечения: {e}")
                
            except Exception as e:
                logger.error(f"Ошибка при извлечении результатов FAQ: {e}")
                return ""
            
            # Фильтруем результаты только по релевантности
            filtered_items = []
            for item in faq_items:
                try:
                    score = float(item.get("_additional", {}).get("score", 0))
                    if score >= self.min_score:
                        filtered_items.append(item)
                except (TypeError, ValueError) as e:
                    logger.error(f"Ошибка обработки score для item {item}: {e}")
            
            if not filtered_items:
                logger.info("Нет релевантных FAQ")
                return ""
            
            # Формируем ответ
            response_parts = []
            for i, item in enumerate(filtered_items, 1):
                response_parts.append(
                    f"\nОтвет на ваш вопрос №{i}:\n"
                    f"{item.get('answer', '')}\n"
                )
            
            return "\n".join(response_parts)
        
        except Exception as e:
            logger.error(f"Ошибка при поиске в FAQ: {e}")
            return ""

    async def search_faq_async(self, query: str, is_authorized: bool = False, limit: int = 3) -> str:
        """
        Асинхронный поиск в FAQ, использующий выполнение в отдельном потоке.
        
        :param query: Запрос пользователя
        :param is_authorized: Флаг авторизации (True для студентов, False для абитуриентов)
        :param limit: Максимальное количество результатов
        :return: Результат поиска в виде строки
        """
        return await asyncio.to_thread(self.search_faq, query, is_authorized, limit)

    async def update_base_from_confluence(self, user_question: str) -> None:
        logger.info("Запуск обновления базы знаний через Confluence.")
        from integrations.confluence_api.confluence_api_search import ConfluenceSearcher
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: ConfluenceSearcher(
                confluence_base_url=self.confluence_base_url,
                confluence_api_token=self.confluence_api_token,
                confluence_page_id=self.confluence_page_id,
                openai_api_key=self.openai_api_key,
                weaviate_api_key=self.weaviate_api_key,
                weaviate_url=self.weaviate_url,
                class_name=self.class_name,
                expected_dim=self.expected_dim,
                max_chunk_size=self.max_chunk_size
            ).run(user_question)
        )
        logger.info("Обновление базы знаний через Confluence завершено.")

    async def search_documents_with_fallback(self, query: str, limit: int = 5) -> str:
        result = await self.search_documents_async(query)
        if result:
            return result
        logger.info("Нет результатов в Weaviate. Обновляю базу знаний через Confluence.")
        await self.update_base_from_confluence(query)
        logger.info(f"Ожидание {retry_delay} секунд для обновления индексов...")
        await asyncio.sleep(retry_delay)
        result = await self.search_documents_async(query)
        return result

if __name__ == "__main__":
    # Пример запуска для ConfluenceWeaviateUploader (эталонного рабочего клиента)
    from integrations.confluence_api.confluence_api_search import ConfluenceWeaviateUploader
    uploader = ConfluenceWeaviateUploader(
        weaviate_url="http://localhost:8080",
        weaviate_api_key="YOUR_WEAVIATE_API_KEY",
        openai_api_key="YOUR_OPENAI_API_KEY",
        class_name="ConfluenceSection_v2",
        expected_dim=3072,
        max_chunk_size=500,
        confluence_base_url=os.environ["CONFLUENCE_BASE_URL"],
        confluence_api_token=os.environ["CONFLUENCE_API_TOKEN_ALT_2"],
        confluence_page_id=195635434
    )
    uploader.run()
