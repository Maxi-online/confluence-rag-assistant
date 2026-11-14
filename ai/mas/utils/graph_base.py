"""
A module for connecting to the Web, setting up schemes and responding to queries supported by multi-vector embeddings.
If relevant information is not found upon request, further enrichment of the knowledge base through Confluence will be initiated,
And then the search is repeated.
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
from weaviate.classes.query import Filter  # Using Filter as documented

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
        Initiate the Web client with parameters for improving the knowledge base through Confluence.
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

        # Connect to the cloud Weaviate.
        self.weaviate_client = connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=AuthApiKey(api_key=self.weaviate_api_key),
            headers={"X-OpenAI-Api-Key": self.openai_api_key},
            skip_init_checks=True
        )
        self.ensure_schema_exists()

    def close(self) -> None:
        """
        Closes the connection to Weaviate.
        """
        if hasattr(self, 'weaviate_client'):
            self.weaviate_client.close()
            logger.info('The connection to the Weaviate is closed.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
                    {"name": "content", "dataType": ["text"]},
                    {"name": "level", "dataType": ["int"]},
                    {"name": "pageId", "dataType": ["text"]},
                    {"name": "lastUpdated", "dataType": ["date"]},
                    {"name": "parentSection", "dataType": [self.class_name]},
                    {"name": "childSections", "dataType": [self.class_name]}
                ]
            }
            self.weaviate_client.collections.create_from_dict(schema_obj)
            logger.info(f"Grade {self.class_name} == sync, corrected by elderman == @elder_man{self.expected_dim}).")

    def search_documents(self, user_question: str) -> str:
        """
        Fully implements the search and filtering logic of documents in the Web.

        :param user_question: User &apos; s request.
        :Return: Discreet documents that satisfy the request or empty string if relevant documents are not found.
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
            
            # Run a request using GraphQL
            try:
                result = self.weaviate_client.graphql_raw_query(query)
                logger.info(f"Documents search raw result: {result}")
            except Exception as e:
                logger.error(f"Error when searching for Weaviate: {e}")
                return ""
            
            # We extract results from the answer taking into account the various possible formats
            try:
                # Correct result processing _RawGQLReturn
                if hasattr(result, 'get') and callable(result.get):
                    # If this is _RawGQLreturn with a get method
                    docs = result.get('data', {}).get('Get', {}).get(self.class_name, [])
                elif hasattr(result, 'get') and hasattr(result, 'errors'):
                    # Check the existence of errors
                    if result.errors is not None and len(result.errors) > 0:
                        logger.error(f"GraphQL returns errors: {result.errors}")
                        return ""
                    # We try to extract the data directly from the attribut
                    if hasattr(result, 'get') and isinstance(result.get, dict):
                        docs = result.get.get(self.class_name, [])
                    else:
                        # For compatibility with previous versions of API
                        result_dict = asdict(result) if hasattr(result, '__dict__') else result
                        docs = result_dict.get("data", {}).get("Get", {}).get(self.class_name, [])
                else:
                    # Normal processing of dictionary
                    result_dict = result
                    docs = result_dict.get("data", {}).get("Get", {}).get(self.class_name, [])
                
                # Checking if we've got the data.
                if not docs and hasattr(result, 'get') and isinstance(result.get, dict):
                    # If the docs are empty, but we have the data in result.get.
                    docs = result.get.get(self.class_name, [])
                
                logger.info(f"Found documents: {docs}")
                
                # If the documents are not yet found, we will try another way of extract
                if not docs and isinstance(result, object) and hasattr(result, 'get'):
                    # Direct access to attribute as written in logs
                    logger.info('We try to extract documents directly from the attributes')
                    try:
                        if hasattr(result, 'get') and result.get and self.class_name in result.get:
                            docs = result.get[self.class_name]
                            logger.info(f"drawn directly: {docs}")
                    except Exception as e:
                        logger.error(f"Errors in trying to get straight out: {e}")
                
            except Exception as e:
                logger.error(f"Error in Search Results: {e}")
                return ""
            
            # Filter the relevance results
            filtered_docs = []
            for doc in docs:
                try:
                    score = float(doc.get("_additional", {}).get("score", 0))
                    if score >= self.min_score:
                        filtered_docs.append(doc)
                except (TypeError, ValueError) as e:
                    logger.error(f"score error processing for the document {doc}: {e}")
            
            if not filtered_docs:
                logger.info('No relevant documents')
                return ""
            
            # Form the answer.
            response_parts = []
            for i, doc in enumerate(filtered_docs, 1):
                response_parts.append(
                    f"\n Excess of document No.{i}\n"
                    f"=====================\n"
                    f"Heading: {doc.get('title', '').strip()}\n"
                    f"Content: {doc.get('content', '').strip()}\n"
                    f"(Score: {doc.get('_additional', {}).get('score', 0)})"
                )
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in search of documents: {e}")
            return ""

    async def search_documents_async(self, query: str) -> str:
        return await asyncio.to_thread(self.search_documents, query)

    def rerank_documents(self, user_question: str, docs: List[dict]) -> List[dict]:
        return sorted(docs, key=lambda d: float(d.get("_additional", {}).get("score", 0)), reverse=True)

    def autocut_documents(self, docs: List[dict]) -> List[dict]:
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
            logging.info(f"Raw result: {raw_result}")
        except Exception as e:
            logging.error(f"Error when searching for Weaviate: {e}")
            return ""
        try:
            if isinstance(raw_result, dict):
                result_dict = raw_result
            else:
                result_dict = asdict(raw_result)
            docs = result_dict.get("data", {}).get("Get", {}).get(self.class_name, [])
            logging.info(f"Docs: {docs}")
        except Exception as e:
            logging.error(f"Error in obtaining results: {e}")
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
            logging.info("No relevant documents, I'm returning the empty context.")
            return ""
        return "\n".join([
            f"\n Excess of document No.{i+1}\n=====================\n Heading: {doc.get('title', '').strip()}\n Content: {doc.get('content', '').strip()}\n(Score: {doc.get('_additional', {}).get('score', 0)})"
            for i, doc in enumerate(final_docs)
        ])

    def search_faq(self, query: str, is_authorized: bool = False, limit: int = 3) -> str:
        """
        Search FAQ based on the user's authorization.
        
        :param query: User request
        :Param is_authorized: Authorisation flag (True for students, False for applicants))
        :param limit: Maximum number of results
        :return: Search string
        """
        try:
            # Preparation of a user-type request
            class_name = "FAQ"
            safe_query = query.replace('"', '\\"')
            
            # Form a GraphQL request with correct filtering
            # For unauthorised users, only the FAQ type "Alumnier" is shown"
            if not is_authorized:
                # The correct filtering format in GraphQL for Weaviate
                graphql_query = f"""
                {{
                  Get {{
                    {class_name}(
                      hybrid: {{
                        query: "{safe_query}",
                        alpha: {self.hybrid_alpha}
                      }},
                      where: {{
                        path: [\"user_type\"],
                        operator: Equal,
                        valueString: \"Alumnier\"
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
                # No filter type used for authorized users
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
            
            # Run a request using GraphQL
            try:
                result = self.weaviate_client.graphql_raw_query(graphql_query)
                logger.info(f"FAQ search raw result: {result}")
            except Exception as e:
                logger.error(f"Errors in search in FAQ: {e}")
                return ""
            
            # We extract results from the answer taking into account the various possible formats of response
            try:
                # Correct result processing _RawGQLReturn
                if hasattr(result, 'get') and callable(result.get):
                    # If this is _RawGQLreturn with a get method
                    faq_items = result.get('data', {}).get('Get', {}).get(class_name, [])
                elif hasattr(result, 'get') and hasattr(result, 'errors'):
                    # Check the existence of errors
                    if result.errors is not None and len(result.errors) > 0:
                        logger.error(f"GraphQL returns errors: {result.errors}")
                        return ""
                    # We try to extract the data directly from the attribut
                    if hasattr(result, 'get') and isinstance(result.get, dict):
                        faq_items = result.get.get(class_name, [])
                    else:
                        # For compatibility with previous versions of API
                        result_dict = asdict(result) if hasattr(result, '__dict__') else result
                        faq_items = result_dict.get("data", {}).get("Get", {}).get(class_name, [])
                else:
                    # Normal processing of dictionary
                    result_dict = result
                    faq_items = result_dict.get("data", {}).get("Get", {}).get(class_name, [])
                
                # Checking if we've got the data.
                if not faq_items and hasattr(result, 'get') and isinstance(result.get, dict):
                    # If faq_items is empty, but we have the data in result.get
                    faq_items = result.get.get(class_name, [])
                
                logger.info(f"FAQ Items: {faq_items}")
                
                # If the elements are not yet found, try another way of extract
                if not faq_items and isinstance(result, object) and hasattr(result, 'get'):
                    # Direct access to attribute as written in logs
                    logger.info('We try to extract FAQ directly from the attribut')
                    try:
                        if hasattr(result, 'get') and result.get and class_name in result.get:
                            faq_items = result.get[class_name]
                            logger.info(f"drawn directly: {faq_items}")
                    except Exception as e:
                        logger.error(f"Errors in trying to get straight out: {e}")
                
            except Exception as e:
                logger.error(f"Errors in finding FAQ results: {e}")
                return ""
            
            # Filter the results on relevance only
            filtered_items = []
            for item in faq_items:
                try:
                    score = float(item.get("_additional", {}).get("score", 0))
                    if score >= self.min_score:
                        filtered_items.append(item)
                except (TypeError, ValueError) as e:
                    logger.error(f"score processing error for item {item}: {e}")
            
            if not filtered_items:
                logger.info('No relevant FAQ')
                return ""
            
            # Form the answer.
            response_parts = []
            for i, item in enumerate(filtered_items, 1):
                response_parts.append(
                    f"\n Response to Your Question No.{i}:\n"
                    f"{item.get('answer', '')}\n"
                )
            
            return "\n".join(response_parts)
        
        except Exception as e:
            logger.error(f"Errors in search in FAQ: {e}")
            return ""

    async def search_faq_async(self, query: str, is_authorized: bool = False, limit: int = 3) -> str:
        """
        Asynchronous FAQ search using separate stream performance.
        
        :param query: User request
        :Param is_authorized: Authorisation flag (True for students, False for applicants))
        :param limit: Maximum number of results
        :return: Line search result
        """
        return await asyncio.to_thread(self.search_faq, query, is_authorized, limit)

    async def update_base_from_confluence(self, user_question: str) -> None:
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

    async def search_documents_with_fallback(self, query: str, limit: int = 5) -> str:
        result = await self.search_documents_async(query)
        if result:
            return result
        logger.info("No results in Weaviate. I'm updating the knowledge base through Confluence.")
        await self.update_base_from_confluence(query)
        logger.info(f"Waiting {retry_delay} seconds to update the indices...")
        await asyncio.sleep(retry_delay)
        result = await self.search_documents_async(query)
        return result

if __name__ == "__main__":
    # Example of a start-up for ConfulenceWeaviateUploader)
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
