import os
import weaviate
import logging
import json
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)

def search_documents(query_text: str, host: str = "localhost", port: int = 8080):
    """
    Выполняет поиск по существующему классу ConfluenceSection_v2 в локальном экземпляре Weaviate.
    Эмбеддинги запроса вычисляются на клиенте с использованием OpenAIEmbeddings (через прокси),
    а затем выполняется поиск через near_vector.
    Возвращает JSON-строку с результатами, содержащими поля: title, content, pageId и _additional.score.
    
    Предполагается, что класс ConfluenceSection_v2 уже существует в Weaviate и использует векторы размерности 1536.
    """
    # Задаем API-ключ OpenAI (оставляем его в коде)
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        logging.error("OpenAI API ключ не задан.")
        return json.dumps([])

    # Устанавливаем переменные окружения для работы модуля text2vec-openai:
    # ВАЖНО: используйте переменную OPENAI_API_BASE_URL (без завершающего слэша)
    os.environ["OPENAI_APIKEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_BASE_URL"] = "https://api.example.org/v1"

    logging.info("Подключение к локальному экземпляру Weaviate...")
    try:
        client = weaviate.connect_to_local(
            host=host,
            port=port,
            skip_init_checks=True
        )
        logging.info("Соединение установлено успешно.")
    except Exception as e:
        logging.error(f"Ошибка подключения к Weaviate: {e}")
        return json.dumps([])

    # Вычисляем эмбеддинг запроса через OpenAIEmbeddings (модель text-embedding-ada-002, возвращающая 1536-мерный вектор)
    try:
        embeddings_client = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base="https://api.example.org/v1"
        )
        query_vector = embeddings_client.embed_query(query_text)
        logging.info(f"Эмбеддинг запроса успешно вычислен. Размерность: {len(query_vector)}")
    except Exception as e:
        logging.error(f"Ошибка вычисления эмбеддингов: {e}")
        client.close()
        return json.dumps([])

    # Выполняем поиск через near_vector с использованием вычисленного вектора
    try:
        # Получаем объект коллекции через актуальный вызов
        collection = client.collections.get("ConfluenceSection_v2")
        # Формируем и выполняем запрос поиска
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=5,
            return_properties=["title", "content", "pageId"]
        ).with_additional_fields(["score"])
    except Exception as e:
        logging.error(f"Ошибка выполнения запроса к Weaviate: {e}")
        client.close()
        return json.dumps([])

    results = []
    try:
        docs = response["data"]["Get"]["ConfluenceSection_v2"]
        for obj in docs:
            results.append({
                "title": obj.get("title"),
                "content": obj.get("content"),
                "pageId": obj.get("pageId"),
                "score": obj.get("_additional", {}).get("score")
            })
    except Exception as e:
        logging.error(f"Ошибка обработки ответа: {e}")
        client.close()
        return json.dumps([])

    logging.info(f"Найдено документов: {len(results)}")
    client.close()
    return json.dumps(results, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    query = "Какие есть курсы по беспилотным технологиям?"
    docs = search_documents(query)
    print(docs)
