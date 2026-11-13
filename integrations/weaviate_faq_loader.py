#!/usr/bin/env python3
"""
Load approved FAQ records into a Weaviate collection.
Collection visibility distinguishes two user types:
- Applicant (is_authorized=False): applicant-visible records only
- Student (is_authorized=True): applicant and student records
"""

import os
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
import weaviate
from weaviate import connect_to_weaviate_cloud, connect_to_local
from weaviate.auth import AuthApiKey
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from integrations.csv_to_faq import load_faq_from_csv

# Logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class FAQLoader:
    def __init__(
        self,
        weaviate_url: str,
        weaviate_api_key: str,
        openai_api_key: str,
        csv_file_path: str,
        class_name: str = "FAQ",
        expected_dim: int = 3072,
        openai_base_url: str = "https://api.openai.com/v1/",
        header_row: int = 0,
        data_start_row: Optional[int] = None,
        column_mapping: Dict[str, str] = None
    ):
        """Configure provider clients and an explicitly selected FAQ input.

        header_row and data_start_row are zero-based input row indices.
        column_mapping optionally maps canonical fields to private input headers.
        No filename or customer-specific worksheet layout is inferred.
        """
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.openai_api_key = openai_api_key
        self.csv_file_path = csv_file_path
        self.class_name = class_name
        self.expected_dim = expected_dim
        self.openai_base_url = openai_base_url
        self.header_row = header_row
        self.data_start_row = data_start_row
        self.column_mapping = column_mapping

        # Initialize the Weaviate client
        if "localhost" in weaviate_url or "127.0.0.1" in weaviate_url:
            self.client = connect_to_local(
                headers={"X-OpenAI-Api-Key": self.openai_api_key}
            )
        else:
            self.client = connect_to_weaviate_cloud(
                cluster_url=self.weaviate_url,
                auth_credentials=AuthApiKey(api_key=self.weaviate_api_key),
                headers={"X-OpenAI-Api-Key": self.openai_api_key}
            )

        # Customer Initialization OpenAI for Embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.openai_api_key,
            base_url=self.openai_base_url
        )

    def create_faq_schema(self) -> None:
        """
        Creates a scheme for the FAQ class in the Web if it does not exist.
        """
        try:
            # Check to see if there's a class like this.
            self.client.collections.get(self.class_name)
            logger.info(f"Grade {self.class_name} It already exists.")
        except Exception:
            logger.info(f"Grade {self.class_name} Not found. I'm making a scheme...")
            
            # Determine class diagram FAQ
            schema_obj = {
                "class": self.class_name,
                "description": 'FAQ knowledge base with user type split',
                "vectorizer": "text2vec-openai",
                "vectorizerConfig": {
                    "name": "faq_vector",
                    "sourceProperties": ["question", "answer", "keywords"],
                    "model": "text-embedding-3-large",
                    "dimensions": self.expected_dim
                },
                "properties": [
                    {"name": "question", "dataType": ["text"], "description": 'User Question'},
                    {"name": "answer", "dataType": ["text"], "description": 'Answer to the question'},
                    {"name": "keywords", "dataType": ["text[]"], "description": 'Keywords for Search'},
                    {"name": "topic", "dataType": ["text"], "description": "Тема запроса"},
                    {"name": "main_point", "dataType": ["text"], "description": 'Key point'},
                    {"name": "user_type", "dataType": ["text"], "description": 'Type of user (academic/student)'},
                    {"name": "implementation_period", "dataType": ["text"], "description": "Срок реализации"}
                ]
            }
            
            # Creating a Class
            self.client.collections.create_from_dict(schema_obj)
            logger.info(f"Grade {self.class_name} == sync, corrected by elderman == @elder_man{self.expected_dim}).")

    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """Read the explicitly selected file with the shared schema-aware loader."""
        return load_faq_from_csv(
            self.csv_file_path,
            header_row=self.header_row,
            data_start_row=self.data_start_row,
            column_mapping=self.column_mapping,
        )

    async def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for text using OpenAI.
        
        :param text: text for embedding
        :Return: Vector of Embedding
        """
        logger.info(f"Create an embedding for text (length: {len(text)})...")
        try:
            embedding = await asyncio.to_thread(self.embeddings.embed_query, text)
            logger.info(f"Embedded successfully created, size: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error in creating an embedding: {e}")
            import traceback
            logger.error(f"Trace: {traceback.format_exc()}")
            # In case of error, we return the zero vector of the required size
            return [0.0] * self.expected_dim

    async def upload_faq_to_weaviate(self) -> None:
        """
        Uploads FAQ data into Weaviate.
        """
        # Make a scheme if it doesn't exist yet.
        self.create_faq_schema()
        
        # Load data from CSV
        records = self.load_data_from_csv()
        if not records:
            logger.error("No FAQ records are available for ingestion.")
            return
        
        # Get a collection.
        collection = self.client.collections.get(self.class_name)
        
        # Upload every record into Weaviate
        for idx, record in enumerate(records):
            try:
                # Create a unique entry identifier by adding an index
                # It guarantees that even the same records will get different IDs
                record_id = str(uuid.uuid4())  # Use random UUID
                
                # We get embeddings for the question and answer together
                combined_text_for_embedding = f"{record['question']} {' '.join(record['keywords'])} {record['answer']}"
                vector = await self.get_embeddings(combined_text_for_embedding)
                
                # Trying to add a new entry.
                try:
                    logger.info(f"Add a new entry: {record['question'][:30]}...")
                    collection.data.insert(properties=record, uuid=record_id, vector=vector)
                except Exception as e:
                    logger.error(f"Failed to insert the FAQ record into Weaviate: {e}")
                
            except Exception as e:
                logger.error(f"Failed to insert the FAQ record into Weaviate: {e}")
        
        logger.info(f"FAQ upload in Weaviate completed. Loaded {len(records)} Records.")

    async def run(self) -> None:
        """
        Starts the FAQ upload process into Weaviate.
        """
        try:
            await self.upload_faq_to_weaviate()
        finally:
            # Close the connection to the Weaviate
            if hasattr(self, 'client'):
                self.client.close()
                logger.info('The connection to the Weaviate is closed.')

def search_faq(
    query: str,
    weaviate_client: Any,
    is_authorized: bool = False,
    limit: int = 3,
    min_score: float = 0.5,
    class_name: str = "FAQ"
) -> str:
    """
    Search FAQ based on the user's authorization.
    
    :param query: User request
    :param weaviate_client:Weaviate client
    :Param is_authorized: Authorisation flag (True for students, False for applicants))
    :param limit: Maximum number of results
    :param min_score: Minimum threshold of relevance
    :param class_name: Name of class FAQ in Weaviate
    :return: Search string
    """
    try:
        # Preparation of a user-type request
        safe_query = query.replace('"', '\\"')
        
        # For unauthorised users (academics) - only content for applicants
        # For authorized (students) all content
        user_type_filter = '' if is_authorized else 'And: [{ path: ["user_type"], operator: Equal, valueString: "Apprentice" }]'
        
        graphql_query = f"""
        {{
          Get {{
            {class_name}(
              hybrid: {{
                query: "{safe_query}"
                alpha: 0.7
              }}
              where: {{
                {user_type_filter}
              }}
              limit: {limit}
            ) {{
              question
              answer
              topic
              user_type
              _additional {{
                score
              }}
            }}
          }}
        }}
        """
        
        result = weaviate_client.graphql_raw_query(graphql_query)
        
        # Getting results.
        faq_items = result.get("data", {}).get("Get", {}).get(class_name, [])
        
        # Filter by relevance
        filtered_items = [
            item for item in faq_items 
            if float(item.get("_additional", {}).get("score", 0)) >= min_score
        ]
        
        if not filtered_items:
            return ""
        
        # Form the answer.
        response_parts = []
        for i, item in enumerate(filtered_items, 1):
            score = item.get("_additional", {}).get("score", 0)
            response_parts.append(
                f"\n Question {i}: {item.get('question', '')}\n"
                f"Answer: {item.get('answer', '')}\n"
                f"Theme: {item.get('topic', '')}\n"
                f"User type: {item.get('user_type', '')}\n"
                f"Relevance: {score:.2f}"
            )
        
        return "\n".join(response_parts)
    
    except Exception as e:
        logger.error(f"Errors in search in FAQ: {e}")
        return ""

def main(argv=None):
    """Run connected ingestion for an explicitly selected input file."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Ingest an explicitly selected FAQ file into Weaviate")
    parser.add_argument("--csv", required=True, help="Approved CSV/Excel input path")
    parser.add_argument("--header-row", type=int, default=0)
    parser.add_argument("--data-start-row", type=int)
    parser.add_argument("--column-map", help="Private JSON mapping of canonical fields to input headers")
    args = parser.parse_args(argv)
    mapping = json.loads(Path(args.column_map).read_text(encoding="utf-8")) if args.column_map else None
    loader = FAQLoader(
        weaviate_url=os.environ["WEAVIATE_URL"],
        weaviate_api_key=os.environ["WEAVIATE_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        csv_file_path=args.csv,
        header_row=args.header_row,
        data_start_row=args.data_start_row,
        column_mapping=mapping,
    )
    asyncio.run(loader.run())

if __name__ == "__main__":
    main()
