import os
import sys
import asyncio
import json
import logging
import uuid

import httpx
import numpy as np
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
# Обновлённый импорт эмбеддингов для устранения предупреждения
from langchain_openai import OpenAIEmbeddings

from .weaviate_upload import ConfluenceWeaviateUploader
from .bm25_filter import BM25Filter

logging.basicConfig(level=logging.INFO)


class ConfluenceSearcher:
    """
    Класс для выполнения асинхронного поиска информации в Confluence с генерацией ключевых фраз,
    фильтрацией результатов по схожести, дополнительной BM25-проверкой и сохранением результатов в Weaviate.
    
    Перед сохранением в Weaviate к каждому найденному результату добавляется информация об иерархии
    (родительские и дочерние связи), чтобы можно было восстановить графовую структуру.
    """

    def __init__(self,
                 confluence_base_url: str,
                 confluence_api_token: str,
                 confluence_page_id: int,
                 openai_api_key: str,
                 weaviate_api_key: str,
                 weaviate_url: str,
                 class_name: str,
                 expected_dim: int,
                 max_chunk_size: int):
        """
        Инициализация параметров поиска. Все необходимые параметры передаются через конструктор.
        """
        logging.basicConfig(level=logging.INFO)
        self.confluence_base_url = confluence_base_url
        self.confluence_api_token = confluence_api_token
        self.confluence_page_id = confluence_page_id
        self.openai_api_key = openai_api_key
        self.weaviate_api_key = weaviate_api_key

        # Параметры для Weaviate
        self.weaviate_url = weaviate_url
        self.class_name = class_name
        self.expected_dim = expected_dim
        self.max_chunk_size = max_chunk_size

        # Инициализация OpenAI клиента и эмбеддингов
        self.client = AsyncOpenAI(
            api_key=self.openai_api_key,
            base_url="https://api.openai.com/v1/"
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=self.openai_api_key,
            base_url="https://api.openai.com/v1/"
        )
    
    async def upload_subset(self, pages_subset: list) -> None:
        """
        Обогащает и загружает в Weaviate только указанный набор страниц,
        который должен включать отфильтрованные документы и все необходимые родительские узлы.
        Все необходимые методы вызываются через класс ConfluenceWeaviateUploader.
        """
        # Вызов метода enrich_with_parent_pages из ConfluenceWeaviateUploader
        pages_subset = await ConfluenceWeaviateUploader.enrich_with_parent_pages(self, pages_subset)
        # Построение иерархии с использованием метода build_hierarchy
        hierarchy = ConfluenceWeaviateUploader.build_hierarchy(self, pages_subset)
        # Создание схемы в Weaviate через метод create_schema
        ConfluenceWeaviateUploader.create_schema(self)
        # Преобразование дерева в плоский список объектов с сохранением уровня вложенности
        flat_objects = ConfluenceWeaviateUploader.parse_hierarchy(hierarchy, parent_uuid=None)
        # Сортировка по возрастанию уровня: корневые узлы (level = 0) будут первыми
        flat_objects = sorted(flat_objects, key=lambda x: x.get("level", 0))
        # Загрузка объектов в Weaviate через метод save_search_results_to_weaviate
        await ConfluenceWeaviateUploader.save_search_results_to_weaviate(self, flat_objects)
        # Удаление устаревших объектов через метод delete_outdated_objects
        current_page_ids = {obj["pageId"] for obj in flat_objects if obj.get("pageId")}
        await ConfluenceWeaviateUploader.delete_outdated_objects(self, current_page_ids)
        self.client.close()

    async def generate_keywords_for_search(self, text: str) -> dict:
        """
        Асинхронная генерация ключевых фраз с использованием OpenAI API.
        """
        system_prompt = "You are an expert in generating keywords for searching a knowledge base."
        user_assist = (
            "When given a user's query: Расскажи о курсе беспилотных технологий, follow these steps:\n"
            "1) Rewrite the query into a simpler, more understandable question.\n"
            "2) Generate 3-5 of the most relevant keyword phrases (each 1-2 word long) that help locate the required information.\n"
            "3) Identify the most relevant phrase as 'primary' and list the remaining as 'secondary'.\n"
            "Output the result as a valid JSON object with two keys: 'primary' and 'secondary'."
        )
        assist = (
            '{"primary": "беспилотные технологии", "secondary": ["курс", "беспилотных", "технологий", "дрон", "аппараты"]}'
        )
        user_message = (
            f"When given a user's query: {text}, follow these steps:\n"
            "1) Rewrite the query into a simpler, more understandable question.\n"
            "2) Generate 3-5 of the most relevant keyword phrases (each 1-2 word long) and identify the most relevant phrase as 'primary', "
            "listing the rest as 'secondary'.\n"
            "Output the result as a valid JSON object with keys 'primary' and 'secondary'."
        )
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_assist},
                    {"role": "assistant", "content": assist},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            alt_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(alt_text)
                primary = result.get("primary", "").strip()
                secondary = result.get("secondary", [])
                if isinstance(secondary, str):
                    secondary = [kw.strip() for kw in secondary.split("OR") if kw.strip()]
            except Exception as parse_e:
                logging.error(f"Ошибка разбора JSON из ответа GPT: {parse_e}")
                raw_keywords = [kw.strip() for kw in alt_text.split("OR") if kw.strip()]
                primary = raw_keywords[0] if raw_keywords else text
                secondary = raw_keywords[1:] if len(raw_keywords) > 1 else []
            if "*" not in primary:
                primary += "*"
            secondary = [kw if "*" in kw else f"{kw}*" for kw in secondary]
            logging.info(f"Generated keywords: primary='{primary}', secondary={secondary}")
            return {"primary": primary, "secondary": secondary}
        except Exception as e:
            logging.error(f"Ошибка генерации ключевых слов: {e}")
            words = [f"{word.strip()}*" for word in text.split() if word.strip()]
            primary = words[0] if words else text
            secondary = words[1:4] if len(words) > 1 else []
            return {"primary": primary, "secondary": secondary}

    async def search_confluence_for_keyword(self, session: httpx.AsyncClient, keyword: str) -> list:
        """
        Асинхронный поиск в Confluence для одного ключевого слова.
        Использует единый HTTP-клиент для оптимизации запросов.
        """
        cql = (
            f'ancestor = {self.confluence_page_id} AND ('
            f'title ~ "{keyword}" OR text ~ "{keyword}" OR '
            f'(type = attachment AND text ~ "{keyword}") OR '
            f'(type = comment AND text ~ "{keyword}")'
            f')'
        )
        logging.info(f"CQL Query for keyword '{keyword}': {cql}")
        url = f"{self.confluence_base_url}/rest/api/content/search"
        headers = {"Authorization": f"Bearer {self.confluence_api_token}"}
        params = {
            "cql": cql,
            "expand": "body.storage,version",
            "includeArchivedSpaces": "false",
            "limit": "200",
        }
        try:
            response = await session.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Keyword '{keyword}' - totalSize: {data.get('totalSize')}")
            results = []
            for result in data.get("results", []):
                title = result.get("title", "")
                webui = result.get("_links", {}).get("webui", "")
                link = f"{self.confluence_base_url}{webui}" if webui else ""
                text_content = result.get("body", {}).get("storage", {}).get("value", "")
                last_updated = result.get("version", {}).get("when", "")
                results.append({
                    "title": title,
                    "link": link,
                    "text": text_content,
                    "last_updated": last_updated,
                    "ancestors": result.get("ancestors", []),
                    "children": []
                })
            return results
        except Exception as e:
            logging.error(f"Ошибка поиска в Confluence для ключевого слова '{keyword}': {e}")
            return []

    async def search_confluence_separately(self, session: httpx.AsyncClient, keywords: list) -> list:
        """
        Параллельный запуск поиска для каждого ключевого слова.
        """
        tasks = [self.search_confluence_for_keyword(session, kw) for kw in keywords]
        results_lists = await asyncio.gather(*tasks)
        aggregated_results = []
        seen = set()
        for results in results_lists:
            for res in results:
                uid = res.get("link")
                if uid and uid not in seen:
                    seen.add(uid)
                    aggregated_results.append(res)
        return aggregated_results

    async def filter_by_keyword_similarity_async(self,
                                                   primary_keyword: str,
                                                   secondary_keywords: list,
                                                   documents: list,
                                                   primary_threshold: float = 0.4,
                                                   secondary_threshold: float = 0.6) -> list:
        """
        Асинхронная фильтрация документов по схожести с ключевыми фразами.
        """
        if not documents:
            return []

        def clean_html(text: str) -> str:
            return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

        def get_combined_text(doc: dict) -> str:
            title = doc.get("title", "")
            body = doc.get("text", "")
            return f"{title} {clean_html(body)}"

        combined_texts = [get_combined_text(doc) for doc in documents]
        if not combined_texts:
            return []

        doc_embeddings = np.array(
            await asyncio.to_thread(self.embeddings.embed_documents, combined_texts)
        )
        primary_query = primary_keyword.rstrip("*").strip()
        secondary_queries = [kw.rstrip("*").strip() for kw in secondary_keywords]

        primary_emb = np.array(
            await asyncio.to_thread(self.embeddings.embed_query, primary_query)
        )
        secondary_embs_tasks = [asyncio.to_thread(self.embeddings.embed_query, q) for q in secondary_queries]
        secondary_embs = [np.array(await task) for task in secondary_embs_tasks]

        accepted_docs = []
        for i, doc in enumerate(documents):
            doc_emb = doc_embeddings[i]
            sim_primary = self.cosine_similarity(primary_emb, doc_emb)
            logging.info(f"Документ '{doc.get('title')}' - primary ('{primary_query}') similarity: {sim_primary:.3f}")
            secondary_similarities = []
            for j, sec_query in enumerate(secondary_queries):
                sim_sec = self.cosine_similarity(secondary_embs[j], doc_emb)
                secondary_similarities.append(sim_sec)
                logging.info(f"Документ '{doc.get('title')}' - secondary ('{sec_query}') similarity: {sim_sec:.3f}")
            max_secondary = max(secondary_similarities) if secondary_similarities else 0.0
            if sim_primary >= primary_threshold:
                doc_copy = doc.copy()
                doc_copy["sim"] = sim_primary
                accepted_docs.append(doc_copy)
            elif sim_primary < primary_threshold and max_secondary >= secondary_threshold:
                doc_copy = doc.copy()
                doc_copy["sim"] = max_secondary
                accepted_docs.append(doc_copy)
            else:
                logging.info(f"Документ '{doc.get('title')}' не принят: primary={sim_primary:.3f} (нужно >= {primary_threshold}) и max_secondary={max_secondary:.3f} (нужно >= {secondary_threshold}).")
        accepted_docs.sort(key=lambda x: x["sim"], reverse=True)
        logging.info(f"Принято {len(accepted_docs)} документов по критериям: primary >= {primary_threshold} или max_secondary >= {secondary_threshold}.")
        return accepted_docs

    async def bm25_filter_documents(self, query: str, documents: list) -> list:
        """
        Асинхронно фильтрует документы с помощью BM25Filter.
        """
        if not documents:
            logging.info("Список документов для BM25 фильтрации пуст. Пропускаем BM25 фильтрацию.")
            return []
        return await asyncio.to_thread(lambda: BM25Filter(documents).filter_documents(query))

    def cosine_similarity(self, vec1, vec2) -> float:
        """
        Расчёт косинусного сходства между двумя векторами.
        """
        if not np.any(vec1) or not np.any(vec2):
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    async def answer_query_async(self, user_question: str) -> str:
        """
        Основная логика поиска:
          1. Генерация ключевых фраз.
          2. Параллельный поиск в Confluence.
          3. Фильтрация результатов по схожести.
          4. Дополнительная BM25-проверка.
          5. Загрузка в Weaviate только отфильтрованных документов и их родительских узлов.
        """
        keywords_dict = await self.generate_keywords_for_search(user_question)
        primary_kw = keywords_dict.get("primary", "")
        secondary_kw = keywords_dict.get("secondary", [])
        logging.info(f"Сгенерированные ключевые фразы: primary='{primary_kw}', secondary={secondary_kw}")
        all_keywords = [primary_kw] + secondary_kw if primary_kw else secondary_kw

        async with httpx.AsyncClient() as session:
            confluence_results = await self.search_confluence_separately(session, all_keywords)
        logging.info(f"Общее количество результатов поиска до фильтрации: {len(confluence_results)}")

        filtered_results = await self.filter_by_keyword_similarity_async(
            primary_kw,
            secondary_kw,
            confluence_results,
            primary_threshold=0.4,
            secondary_threshold=0.4,
        )
        logging.info(f"Количество документов после BM25 фильтрации: {len(filtered_results)}")

        if filtered_results:
            # Формируем набор страниц для загрузки:
            # включаем все отфильтрованные документы и добавляем всех предков (ancestors)
            pages_to_upload = {}
            for page in filtered_results:
                # Если у документа отсутствует "id", генерируем fallback‑значение
                page_id = page.get("id")
                if not page_id:
                    page_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"confluence:{page.get('link', page.get('title', 'unknown'))}"))
                    page["id"] = page_id
                pages_to_upload[page_id] = page
                # Обрабатываем предков
                for ancestor in page.get("ancestors", []):
                    anc_id = ancestor.get("id")
                    if not anc_id:
                        anc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"confluence:{ancestor.get('link', ancestor.get('title', 'unknown'))}"))
                        ancestor["id"] = anc_id
                    if anc_id not in pages_to_upload:
                        pages_to_upload[anc_id] = ancestor
            pages_to_upload = list(pages_to_upload.values())
            # Создаем экземпляр загрузчика и вызываем загрузку только выбранного набора страниц
            uploader = ConfluenceWeaviateUploader(
                weaviate_url=self.weaviate_url,
                weaviate_api_key=self.weaviate_api_key,
                openai_api_key=self.openai_api_key,
                class_name=self.class_name,
                expected_dim=self.expected_dim,
                max_chunk_size=self.max_chunk_size,
                confluence_base_url=self.confluence_base_url,
                confluence_api_token=self.confluence_api_token,
                confluence_page_id=self.confluence_page_id
            )
            await uploader.upload_subset(pages_to_upload)
            confluence_text = "\n".join(page.get("text", "") for page in filtered_results)
            return confluence_text
        else:
            return "К сожалению, не найдено информации по вашему запросу."

    async def get_hierarchy_mapping(self) -> dict:
        """
        Асинхронно получает информацию об иерархии страниц из Confluence,
        используя механизм загрузчика (без временного файла).
        Формирует маппинг: ключ – ссылка страницы, значение – словарь с 'parent_uuid' и 'child_uuids'.
        """
        uploader = ConfluenceWeaviateUploader(
            weaviate_url=self.weaviate_url,
            weaviate_api_key=self.weaviate_api_key,
            openai_api_key=self.openai_api_key,
            class_name=self.class_name,
            expected_dim=self.expected_dim,
            max_chunk_size=self.max_chunk_size,
            confluence_base_url=self.confluence_base_url,
            confluence_api_token=self.confluence_api_token,
            confluence_page_id=self.confluence_page_id
        )
        pages = await uploader.fetch_confluence_pages()
        pages = await uploader.enrich_with_parent_pages(pages)
        hierarchy = uploader.build_hierarchy(pages)
        flat = ConfluenceWeaviateUploader.parse_hierarchy(hierarchy, parent_uuid=None)
        mapping = {}
        for obj in flat:
            mapping[obj.get("pageId")] = {
                "parent_uuid": obj.get("parent_uuid"),
                "child_uuids": obj.get("child_uuids", [])
            }
        return mapping

    def run(self, user_question: str) -> None:
        """
        Синхронный запуск асинхронного поиска по вопросу.
        """
        asyncio.run(self._run_async(user_question))

    async def _run_async(self, user_question: str) -> None:
        answer = await self.answer_query_async(user_question)
        print("\nОтвет AI:")
        print(answer)


if __name__ == "__main__":
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
    CLASS_NAME = os.getenv("CLASS_NAME", "ConfluenceSection_v2")
    EXPECTED_DIM = int(os.getenv("EXPECTED_DIM", "1536"))
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1536"))
    CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
    CONFLUENCE_PAGE_ID = os.getenv("CONFLUENCE_PAGE_ID", "195635434")
    searcher = ConfluenceSearcher(
        confluence_base_url=CONFLUENCE_BASE_URL,
        confluence_api_token=CONFLUENCE_API_TOKEN,
        confluence_page_id=CONFLUENCE_PAGE_ID,
        openai_api_key=OPENAI_API_KEY,
        weaviate_api_key=WEAVIATE_API_KEY,
        weaviate_url=WEAVIATE_URL,
        class_name=CLASS_NAME,
        expected_dim=EXPECTED_DIM,
        max_chunk_size=MAX_CHUNK_SIZE
    )
    searcher.run("Какие есть курсы по беспилотным технологиям?")
