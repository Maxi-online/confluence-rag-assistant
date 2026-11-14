from integrations.confluence_api.text_utils import format_date, static_clean_html
import os
import sys
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import weaviate
import httpx
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from openai import AsyncOpenAI
from dataclasses import asdict
from typing import List, Optional, Any, Dict

# Connecting to local version of Weaviate via gRPC
from weaviate import connect_to_local
from weaviate.exceptions import UnexpectedStatusCodeError
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter

import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





class ConfluenceWeaviateUploader:
    """
    Convention page download class on the Web with support for asynchronous work,
    Parallelation of operations and package processing of embeddings.
    
    The hierarchic structure is built directly from Confluence data – every node (even without content))
    retained and cross-links can be installed for nodes with the same headings.
    """

    def __init__(self,
                 weaviate_url: str,
                 weaviate_api_key: str,  # not used for local connection
                 openai_api_key: str,
                 class_name: str = "ConfluenceSection_v2",
                 expected_dim: int = 1536,
                 max_chunk_size: int = 1536,
                 confluence_base_url: str = None,
                 confluence_api_token: str = None,
                 confluence_page_id: int = None):
        """
        Initiating downloader, OpenAI and Weaviate clients.
        All parameters are transmitted through the designer's arguments.
        """
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s: %(message)s")

        self.weaviate_url = weaviate_url
        self.class_name = class_name
        self.expected_dim = expected_dim
        self.max_chunk_size = max_chunk_size

        self.confluence_base_url = confluence_base_url
        self.confluence_api_token = confluence_api_token
        self.confluence_page_id = confluence_page_id

        self.api_key = openai_api_key
        self.client_openai = AsyncOpenAI(api_key=self.api_key,
                                         base_url="https://api.openai.com/v1/")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key,
            base_url="https://api.openai.com/v1/"
        )

        self.client = connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,
            headers={"X-OpenAI-Api-Key": self.weaviate_api_key},
            skip_init_checks=True
        )

    # -------------------- Text-processing methods --------------------
    def clean_html(self, html: str) -> str:
        return static_clean_html(html)

    def split_text(self, text: str, max_chunk_size: int) -> list:
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=5)
        return splitter.split_text(text)

    def ensure_vector_dimension(self, vector: list, expected_dim: int) -> list:
        if len(vector) < expected_dim:
            vector = vector + [0.0] * (expected_dim - len(vector))
        elif len(vector) > expected_dim:
            vector = vector[:expected_dim]
        return vector

    def project_vector(self, vector: list, target_dim: int, projection_matrix: np.array = None) -> list:
        vector_np = np.array(vector)
        current_dim = vector_np.shape[0]
        if current_dim == target_dim:
            return vector
        if projection_matrix is None:
            np.random.seed(42)
            projection_matrix = np.random.randn(current_dim, target_dim)
        projected = np.dot(vector_np, projection_matrix)
        return projected.tolist()

    async def get_valid_embedding(self, text: str, max_chunk_size: int,
                                  expected_dim: int = None) -> list:
        if expected_dim is None:
            expected_dim = self.expected_dim
        vector = await asyncio.to_thread(self.embeddings.embed_query, text)
        if len(vector) != expected_dim:
            logging.info(f"Embedding dimension {len(vector)} != {expected_dim}. By random projection.")
            vector = await asyncio.to_thread(self.project_vector, vector, expected_dim)
        return vector

    async def get_valid_embeddings_batch(self, texts: list, expected_dim: int = None) -> list:
        if expected_dim is None:
            expected_dim = self.expected_dim
        vectors = await asyncio.to_thread(self.embeddings.embed_documents, texts)
        final_vectors = []
        for vec in vectors:
            if len(vec) != expected_dim:
                logging.info(f"Embedding dimension {len(vec)} != {expected_dim}. By random projection.")
                vec = self.project_vector(vec, expected_dim)
            final_vectors.append(vec)
        return final_vectors

    async def intelligent_chunk_text(self, text: str, max_chunk_size: int, model: str = "gpt-4o-mini") -> list:
        if len(text) <= max_chunk_size:
            return [text]
        system_message = (
            "Split the following text into logically connected parts, ensuring that each part does not exceed "
            f"{max_chunk_size} characters. For each part, generate a short heading that reflects the essence of the text. "
            "If a part is divided into several logically related blocks of equal length, append a label: ' - (Part 1)', ' - (Part 2)', etc. "
            "Output the result in JSON format — a list of objects with fields 'heading' and 'content'."
        )
        user_message = f"Battery text:\n{text}"
        try:
            response = await self.client_openai.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": user_message}],
                temperature=0
            )
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.strip("```")
                if result_text.lstrip().startswith("json"):
                    result_text = result_text.lstrip()[len("json"):].strip()
            start_idx = result_text.find('[')
            end_idx = result_text.rfind(']')
            json_str = (result_text[start_idx:end_idx + 1]
                        if start_idx != -1 and end_idx != -1 else result_text)
            chunks = json.loads(json_str)
            final_chunks = []
            for chunk in chunks:
                heading = chunk.get("heading", "").strip()
                content = chunk.get("content", "").strip()
                combined = f"{heading}\n{content}" if heading else content
                final_chunks.append(combined)
            return final_chunks
        except Exception as e:
            logging.error(f"The error of intellectual breakdown: {e}")
            return self.split_text(text, max_chunk_size)

    async def split_text_intelligent(self, text: str, max_chunk_size: int) -> list:
        if len(text) > max_chunk_size:
            return await self.intelligent_chunk_text(text, max_chunk_size)
        return [text]

    async def chunks_to_documents(self, text: str, metadata: dict, max_chunk_size: int) -> list:
        chunks = await self.split_text_intelligent(text, max_chunk_size)
        return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

    # -------------------- Methods of working with the Web --------------------
    def create_schema(self) -> None:
        if self.client.collections.exists(self.class_name):
            logging.info(f"Grade {self.class_name} It already exists. New information will be added to the existing collection.")
            return
        schema_obj = {
            "class": self.class_name,
            "description": 'Sections and pages of the knowledge base',
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
        self.client.collections.create_from_dict(schema_obj)
        logging.info(f"Grade {self.class_name} == sync, corrected by elderman == @elder_man{self.expected_dim}).")

    async def upsert_object(self, obj: dict, vector: list = None, retries: int = 3) -> None:
        properties = {
            "title": obj["title"],
            "content": obj["content"],
            "level": obj["level"],
            "pageId": obj["pageId"],
            "lastUpdated": format_date(obj.get("last_updated"))
        }
        logging.info(f"Send an object to Weaviate:\n{json.dumps(properties, ensure_ascii=False, indent=2)}")
        if obj["parent_uuid"]:
            properties["parentSection"] = [{
                "beacon": f"weaviate://{self.class_name}/{obj['parent_uuid']}"
            }]
        if obj.get("child_uuids"):
            properties["childSections"] = [{
                "beacon": f"weaviate://{self.class_name}/{child_uuid}"
            } for child_uuid in obj["child_uuids"]]
        collection = self.client.collections.get(self.class_name)
        for attempt in range(retries):
            try:
                if vector is not None:
                    await asyncio.to_thread(collection.data.insert, properties, uuid=obj["id"], vector=vector)
                else:
                    await asyncio.to_thread(collection.data.insert, properties, uuid=obj["id"])
                logging.info(f"The object has been inserted: {obj['title']} (ID: {obj['id']}).")
                return
            except UnexpectedStatusCodeError as e:
                if "503" in str(e):
                    logging.warning(f"503 errors when placing an object {obj['id']}, try {attempt + 1} from {retries}.")
                    await asyncio.sleep(2 ** attempt)
                elif "already exists" in str(e):
                    try:
                        if vector is not None:
                            await asyncio.to_thread(collection.data.replace, obj["id"], properties, vector=vector)
                        else:
                            await asyncio.to_thread(collection.data.replace, obj["id"], properties)
                        logging.info(f"New object: {obj['title']} (ID: {obj['id']}).")
                        return
                    except Exception as inner_e:
                        raise inner_e
                else:
                    raise e
        raise Exception(f"Unable to insert object {obj['id']} after {retries} Trying.")

    async def delete_outdated_objects(self, current_page_ids: set) -> None:
        try:
            query = (
                f"{{\n"
                f"  Get {{\n"
                f"    {self.class_name} {{\n"
                f"      pageId\n"
                f"      _additional {{\n"
                f"        id\n"
                f"      }}\n"
                f"    }}\n"
                f"  }}\n"
                f"}}"
            )
            result = self.client.graphql_raw_query(query)
            objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            delete_tasks = []
            for obj in objects:
                obj_page_id = obj.get("pageId")
                if obj_page_id not in current_page_ids:
                    obj_id = obj["_additional"]["id"]
                    logging.info(f"Remove object with ID: {obj_id} (pageId: {obj_page_id})")
                    delete_tasks.append(asyncio.to_thread(self.client.data.delete, uuid=obj_id))
            if delete_tasks:
                await asyncio.gather(*delete_tasks)
                logging.info('The disposal of obsolete sites has been completed.')
        except Exception as e:
            logging.error(f"Error in removing an old object: {e}")

    async def save_search_results_to_weaviate(self, data: list) -> None:
        """
        Uploads the results of a search for Confluence in the Web with the construction of a hierarchy.
        The objects are sorted according to the increasing level (level) for a sequenced insertion.
        """
        # We create a scheme if it's not there yet.
        self.create_schema()

        # Transform the tree into a flat list of objects while maintaining the level of input
        flat_objects = self.parse_hierarchy(data, parent_uuid=None)
        # Sorting objects up to level (corn knots will be the first))
        flat_objects = sorted(flat_objects, key=lambda x: x.get("level", 0))

        # Documents are generated for each object (final_docs)
        for item in flat_objects:
            raw_text = item.get("content", "")
            cleaned_text = self.clean_html(raw_text)
            # If the purified text is empty, insert a title or tableholder
            if not cleaned_text.strip():
                cleaned_text = item.get("title", 'No contents')
            # If pageId is empty, generate a tableholder based on the title
            page_link = item.get("pageId", "").strip() or f"confluence:{item.get('title', 'unknown')}"
            meta = {
                "title": item.get("title", ""),
                "link": page_link,
                "last_updated": item.get("last_updated", format_date())
            }
            docs = await self.chunks_to_documents(cleaned_text, meta, self.max_chunk_size)
            final_docs = []
            for doc in docs:
                if len(doc.page_content) > 2048:
                    sub_chunks = await self.intelligent_chunk_text(doc.page_content, 2048)
                    for sub in sub_chunks:
                        final_docs.append(Document(page_content=sub, metadata=doc.metadata))
                else:
                    final_docs.append(doc)
            item["final_docs"] = final_docs
            # Update pageId in object
            item["pageId"] = page_link

        # We collect all documents for the calculation of embeddings
        all_docs = []
        for item in flat_objects:
            for doc in item.get("final_docs", []):
                if doc.page_content:
                    all_docs.append(doc.page_content)
        vectors = await self.get_valid_embeddings_batch(all_docs, expected_dim=self.expected_dim) if all_docs else []

        # Upload objects to the Web in sequence (upgraded))
        for item in flat_objects:
            for doc in item.get("final_docs", []):
                vector = vectors.pop(0) if doc.page_content and vectors else None
                obj_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"confluence:{item.get('pageId', '')}:{item.get('title', '')}"))
                obj = {
                    "id": obj_id,
                    "title": item.get("title", ""),
                    "content": doc.page_content,
                    "level": item.get("level", 0),
                    "pageId": item.get("pageId", ""),
                    "lastUpdated": item.get("last_updated", format_date()),
                    "parent_uuid": item.get("parent_uuid", None),
                    "child_uuids": item.get("child_uuids", [])
                }
                await self.upsert_object(obj, vector=vector)
        logging.info('All results are successfully downloaded from the Web.')

    @staticmethod
    def parse_hierarchy(data: list, parent_uuid=None) -> list:
        objects = []
        for item in data:
            level = item.get("level", 0)
            title = item["title"]
            page_id = item.get("id", "")
            last_updated = item.get("last_updated", format_date())
            content = item.get("content", "")
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"confluence:{page_id}:{title}"))
            obj = {
                "id": obj_uuid,
                "title": title,
                "content": static_clean_html(content) if content else "",
                "level": level,
                "pageId": page_id,
                "last_updated": last_updated,
                "parent_uuid": parent_uuid,
                "child_uuids": []
            }
            children = item.get("children", [])
            if children:
                child_objs = ConfluenceWeaviateUploader.parse_hierarchy(children, parent_uuid=obj_uuid)
                obj["child_uuids"] = [child["id"] for child in child_objs]
                objects.append(obj)
                objects.extend(child_objs)
            else:
                objects.append(obj)
        return objects

    async def fetch_single_page(self, page_id: str) -> dict:
        url = f"{self.confluence_base_url}/rest/api/content/{page_id}"
        params = {"expand": "body.storage,ancestors,version"}
        headers = {"Authorization": f"Bearer {self.confluence_api_token}"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            page = {
                "id": data.get("id", ""),
                "title": data.get("title", ""),
                "last_updated": data.get("version", {}).get("when", format_date()),
                "content": data.get("body", {}).get("storage", {}).get("value", ""),
                "depth": len(data.get("ancestors", [])),
                "link": f"{self.confluence_base_url}{data.get('_links', {}).get('webui', '')}",
                "ancestors": data.get("ancestors", []),
                "children": []
            }
            return page

    async def enrich_with_parent_pages(self, pages: list) -> list:
        pages_by_id = {p["id"]: p for p in pages if p.get("id")}
        for page in list(pages):
            ancestors = page.get("ancestors", [])
            if ancestors:
                for ancestor in ancestors:
                    anc_id = ancestor.get("id")
                    if not anc_id:
                        anc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"confluence:{ancestor.get('link', ancestor.get('title', 'unknown'))}"))
                        ancestor["id"] = anc_id
                    if anc_id not in pages_by_id:
                        try:
                            fetched = await self.fetch_single_page(anc_id)
                            if fetched:
                                fetched.setdefault("children", [])
                                pages.append(fetched)
                                pages_by_id[anc_id] = fetched
                        except Exception as e:
                            logging.error(f"Error in receiving the parent page with ID {anc_id}: {e}")
        return pages

    def build_hierarchy(self, pages: list) -> list:
        pages_by_id = {p["id"]: p for p in pages if p.get("id")}
        for page in pages:
            page.setdefault("children", [])
            ancestors = page.get("ancestors", [])
            if ancestors:
                parent_info = ancestors[-1]  # Direct parent
                page["parent_id"] = parent_info.get("id")
                page["parent_title"] = parent_info.get("title")
            else:
                page["parent_id"] = None
                page["parent_title"] = None
        root_pages = []
        for page in pages:
            if page["parent_id"] and page["parent_id"] in pages_by_id:
                parent = pages_by_id[page["parent_id"]]
                parent.setdefault("children", []).append(page)
            else:
                root_pages.append(page)
        def set_level(nodes, current_level):
            for node in nodes:
                node["level"] = current_level
                logging.info(f"Host {node.get('title')} Got a level. {current_level}")
                if node.get("children"):
                    set_level(node["children"], current_level + 1)
        set_level(root_pages, 0)
        return root_pages

    async def fetch_confluence_pages(self) -> list:
        pages = []
        try:
            url_root = f"{self.confluence_base_url}/rest/api/content/{self.confluence_page_id}"
            params_root = {"expand": "body.storage,ancestors,version"}
            headers = {"Authorization": f"Bearer {self.confluence_api_token}"}
            async with httpx.AsyncClient() as client:
                response_root = await client.get(url_root, params=params_root, headers=headers)
                response_root.raise_for_status()
                root_page = response_root.json()
                pages.append({
                    "id": root_page.get("id", ""),
                    "title": root_page.get("title", ""),
                    "last_updated": root_page.get("version", {}).get("when", format_date()),
                    "content": root_page.get("body", {}).get("storage", {}).get("value", ""),
                    "depth": len(root_page.get("ancestors", [])),
                    "link": f"{self.confluence_base_url}{root_page.get('_links', {}).get('webui', '')}",
                    "ancestors": root_page.get("ancestors", []),
                    "children": []
                })
                logging.info(f"Got a root page: {root_page.get('title', '')} (ID: {root_page.get('id', '')})")
        except Exception as e:
            error_msg = f"Error when receiving a root page with ID {self.confluence_page_id}: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)
        start = 0
        limit = 50
        async with httpx.AsyncClient() as client:
            while True:
                cql = f"ancestor = {self.confluence_page_id}"
                url = f"{self.confluence_base_url}/rest/api/content/search"
                headers = {"Authorization": f"Bearer {self.confluence_api_token}"}
                params = {
                    "cql": cql,
                    "limit": limit,
                    "start": start,
                    "expand": "body.storage,ancestors,version"
                }
                try:
                    response = await client.get(url, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                    total = data.get("totalSize")
                    logging.info(f"totalSize = {total}")
                    if total == 0:
                        logging.info('TotalSize is 0, skipping the request for the sub-pages.')
                        break
                    results = data.get("results", [])
                    if not results:
                        logging.info('No additional results, complete request for subsidiaries.')
                        break
                    for result in results:
                        title = result.get("title", "")
                        page_id = result.get("id", "")
                        webui = result.get("_links", {}).get("webui", "")
                        link = f"{self.confluence_base_url}{webui}" if webui else ""
                        text_content = result.get("body", {}).get("storage", {}).get("value", "")
                        last_updated = result.get("version", {}).get("when", format_date())
                        ancestors = result.get("ancestors", [])
                        depth = len(ancestors)
                        pages.append({
                            "id": page_id,
                            "title": title,
                            "last_updated": last_updated,
                            "content": text_content,
                            "depth": depth,
                            "link": link,
                            "ancestors": ancestors,
                            "children": []
                        })
                    start += limit
                except Exception as e:
                    logging.error(f"Failure to request pages from Confluence: {e}")
                    break
        logging.info(f"Total pages collected: {len(pages)}")
        return pages

    async def upload_subset(self, pages_subset: list) -> None:
        """
        Enrichs and downloads only the specified page set on the Web,
        which should include filtered documents and all necessary parent nodes.
        """
        # Enriching the pages with parent knots
        pages_subset = await self.enrich_with_parent_pages(pages_subset)
        # Building a hierarchy
        hierarchy = self.build_hierarchy(pages_subset)
        # Create a scheme in the Weaviate (if not already created)
        self.create_schema()
        # Transform the tree into a flat list of objects while maintaining the level of input
        flat_objects = self.parse_hierarchy(hierarchy, parent_uuid=None)
        # Grading level (corn knots first)
        flat_objects = sorted(flat_objects, key=lambda x: x.get("level", 0))
        # Upload objects to Weaviate
        await self.save_search_results_to_weaviate(flat_objects)
        # Remove obsolete objects if they exist
        current_page_ids = {obj["pageId"] for obj in flat_objects if obj.get("pageId")}
        await self.delete_outdated_objects(current_page_ids)
        self.client.close()

    async def main_async(self) -> None:
        logging.info('We get the pages from Confluence...')
        pages = await self.fetch_confluence_pages()
        if not pages:
            logging.error('The list of pages from Confluence is empty!')
            sys.exit(1)
        logging.info('Enriching data with parent knots...')
        pages = await self.enrich_with_parent_pages(pages)
        logging.info('Building a hierarchy...')
        objects = self.build_hierarchy(pages)
        if not objects:
            logging.error('The built hierarchy is empty!')
            sys.exit(1)
        self.create_schema()
        await self.save_search_results_to_weaviate(self.parse_hierarchy(objects, parent_uuid=None))
        current_page_ids = {obj["pageId"] for obj in self.parse_hierarchy(objects, parent_uuid=None) if obj.get("pageId")}
        await self.delete_outdated_objects(current_page_ids)
        logging.info('All objects have been successfully downloaded and updated in the Weaviate.')
        self.client.close()

    def run(self) -> None:
        asyncio.run(self.main_async())


class WeaviateClient:
    def __init__(
        self,
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
        Initiate the Web client with parameters for improving the knowledge base through Confluence.

        :param weaviate_url: URL Weaviate server.
        :Paragraph weaviate_api_key: API Key for Access to the Web.
        :Paragraph class_name: Title of the document storage class(s).
        :Param expected_dim: Expected number of measurements of vector representation.
        :param confluence_base_url: URL Confluence server.
        :param confluence_api_token: API Key for Access to Conference.
        :Param confluence_page_id: Confulsion Page Identifier for Search.
        :param max_chunk_size: Maximum size of the cup when text is broken.
        :param min_score: Minimum relevance threshold (score)).
        :param hybrid_alpha: Hybrid search parameter (alpha).
        """
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

        self.weaviate_client = connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,
            headers={"X-OpenAI-Api-Key": self.weaviate_api_key},
            skip_init_checks=True
        )
        self.ensure_schema_exists()

    def ensure_schema_exists(self) -> None:
        """
        Checks whether there is a specified class(s) in Weaviate. If not, it creates it.
        """
        try:
            self.weaviate_client.collections.get(self.class_name)
            logger.info(f"Grade {self.class_name} It already exists.")
        except Exception:
            logger.info(f"Grade {self.class_name} Not found. I'm making a scheme...")
            schema_obj = {
                "class": self.class_name,
                "description": 'A knowledge base to find context.',
                "vectorizer": "text2vec-openai",
                "vectorizerConfig": {
                    "name": "content_vector",
                    "sourceProperties": ["title", "content"],
                    "model": "text-embedding-3-large",
                    "dimensions": self.expected_dim
                },
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]}
                ]
            }
            self.weaviate_client.collections.create_from_dict(schema_obj)
            logger.info(f"Grade {self.class_name} == sync, corrected by elderman == @elder_man{self.expected_dim}).")

    def search_documents(self, query: str, limit: int = 5) -> str:
        """
        Searches for documents from the Web on request.

        :param query: User request.
        :param limit: Maximum number of documents returned.
        :Return: Line with documents found or empty line if nothing is found.
        """
        safe_query = query.replace('"', '\\"')
        graphql_query = (
            f"{{\n"
            f"  Get {{\n"
            f"    {self.class_name}(\n"
            f"      hybrid: {{ query: \"{safe_query}\", alpha: {self.hybrid_alpha} }},\n"
            f"      limit: {limit}\n"
            f"    ) {{\n"
            f"      title\n"
            f"      content\n"
            f"      _additional {{ score }}\n"
            f"    }}\n"
            f"  }}\n"
            f"}}"
        )
        try:
            result = self.weaviate_client.graphql_raw_query(graphql_query)
            docs = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            if not docs:
                return ""
            answer = "\n".join(
                [
                    f"Heading: {doc.get('title', '').strip()}\n Content: {doc.get('content', '').strip()}\n(Score: {doc.get('_additional', {}).get('score', 0)})"
                    for doc in docs
                ]
            )
            return answer
        except Exception as e:
            logger.error(f"Search errors in Weaviate: {e}")
            return ""

    async def search_documents_async(self, query: str, limit: int = 5) -> str:
        """
        Asynchronous search for documents using separate stream performance.

        :param query: User request.
        :param limit: Maximum number of documents.
        :return: Line search result.
        """
        return await asyncio.to_thread(self.search_documents, query, limit)

    def rerank_documents(self, user_question: str, docs: List[dict]) -> List[dict]:
        """
        Ran the found documents on loss of relevance (score)).

        :param user_question: User &apos; s request.
        :param docs: List of documents found.
        :return: A sorted list of documents.
        """
        return sorted(docs, key=lambda d: float(d.get("_additional", {}).get("score", 0)), reverse=True)

    def autocut_documents(self, docs: List[dict]) -> List[dict]:
        """
        Cuts off the list of documents, leaving only the top N (e.g. the first three)).

        :param docs: List of documents.
        Shortened list of documents.
        """
        return docs[:3]

    def search_local_knowledge(self, user_question: str) -> str:
        """
        Fully implements the search and filtering logic of documents in the Web.

        :param user_question: User &apos; s request.
        :Return: Discreet documents that satisfy the request or empty string if relevant documents are not found.
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
            logger.info(f"Raw result: {raw_result}")
        except Exception as e:
            logger.error(f"Error when searching for Weaviate: {e}")
            return ""
        try:
            if isinstance(raw_result, dict):
                result_dict = raw_result
            else:
                result_dict = asdict(raw_result)
            docs = result_dict.get("data", {}).get("Get", {}).get(self.class_name, [])
            logger.info(f"Docs: {docs}")
        except Exception as e:
            logger.error(f"Error in obtaining results: {e}")
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
            logger.info("No relevant documents, I'm returning the empty context.")
            return ""
        return "\n".join([
            f"\n Excess of document No.{i+1}\n=====================\n Heading: {doc.get('title', '').strip()}\n Content: {doc.get('content', '').strip()}\n(Score: {doc.get('_additional', {}).get('score', 0)})"
            for i, doc in enumerate(final_docs)
        ])

    async def update_base_from_confluence(self, user_question: str) -> None:
        """
        Updates the Weaviate knowledge base by supplementing it with Confluence data.
        Call ConferenceSearcher from integration and wait for the process to be completed.

        :Paragraph user_question: User &apos; s Request for Refinement of Conference Search.
        """
        logger.info('Start updating the knowledge base via Confluence.')
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
        logger.info('The update of the knowledge base through Confluence has been completed.')

    async def search_documents_with_fallback(self, query: str, limit: int = 5, retry_delay: int = 5) -> str:
        """
        Asynchronous looking for documents at Weaviate. If local search doesn't return results,
        Updates the knowledge base through Confluence, waits for the time and repeats the search.

        :param query: User request.
        :param limit: Maximum number of documents.
        :param retry_delay: Waiting time (in seconds) after updating the base.
        :return: Context found or empty line found.
        """
        result = await self.search_documents_async(query, limit)
        if result:
            return result
        logger.info("No results in Weaviate. I'm updating the knowledge base through Confluence.")
        await self.update_base_from_confluence(query)
        logger.info(f"Waiting {retry_delay} seconds to update the indices...")
        await asyncio.sleep(retry_delay)
        result = await self.search_documents_async(query, limit)
        return result

if __name__ == "__main__":
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    CLASS_NAME = "ConfluenceSection_v2"
    EXPECTED_DIM = int(os.getenv("EXPECTED_DIM", "3072"))
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "3072"))
    CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
    CONFLUENCE_PAGE_ID = os.environ["CONFLUENCE_PAGE_ID"]
    uploader = ConfluenceWeaviateUploader(
        weaviate_url=WEAVIATE_URL,
        weaviate_api_key=WEAVIATE_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        class_name=CLASS_NAME,
        expected_dim=EXPECTED_DIM,
        max_chunk_size=MAX_CHUNK_SIZE,
        confluence_base_url=CONFLUENCE_BASE_URL,
        confluence_api_token=CONFLUENCE_API_TOKEN,
        confluence_page_id=CONFLUENCE_PAGE_ID
    )
    uploader.run()
