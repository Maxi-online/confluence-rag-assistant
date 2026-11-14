#!/usr/bin/env python3
"""
Test FAQ search after ingesting data into Weaviate.
Verify access boundaries between applicant and student records.
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



# Logging settings
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
    Search FAQ based on the user's authorization.
    """
    try:
        # Get a collection.
        collection = weaviate_client.collections.get(class_name)
        logger.info(f"Number of objects in collection: {collection.aggregate.over_all().total_count}")
        
        # Create a user-type filter if the user is not authorized
        where_filter = None
        if not is_authorized:
            # Using API v4 to create a filter
            where_filter = Filter.by_property("user_type").equal("Абитуриент")
        
        # Perform a hybrid search using a new API.
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
            # Check the error message from the server OpenAI or Weaviate
            if "Технические работы" in error_msg or "invalid character '<'" in error_msg:
                logger.error('OpenAI API service is temporarily unavailable - technical work on the server.')
                return 'SERVIS temporarily unavailable: Technical work is carried out on OpenAI API. Please try later.'
            # Remove the other errors further
            raise
        
        # Processing the results
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
        
        logger.info(f"Found {len(filtered_items)} appropriate facilities")
        
        if not filtered_items:
            return ""
            
        # Form the answer.
        response_parts = []
        for i, item in enumerate(filtered_items, 1):
            score = item["_additional"]["score"]
            keywords_str = ", ".join(item.get("keywords", []))
            
            response_parts.append(
                f"\n Question {i}: {item['question']}\n"
                f"Answer: {item['answer']}\n"
                f"Theme: {item['topic']}\n"
                f"User type: {item['user_type']}\n"
                f"Keywords: {keywords_str}\n"
                f"Relevance: {score:.2f}"
            )
            
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Errors in search in FAQ: {e}")
        import traceback
        logger.error(f"Trace: {traceback.format_exc()}")
        return ""

def main():
    parser = argparse.ArgumentParser(description='FAQ Search Test')
    parser.add_argument('--query', type=str, required=True, help='Search request')
    parser.add_argument('--authorized', action='store_true', help='Authorisation flag (false default - applicant)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--weaviate-url', type=str, help='URL Weaviate server')
    parser.add_argument('--weaviate-api-key', type=str, help='API Key')
    parser.add_argument('--openai-api-key', type=str, help='OpenAI key')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = _resolve_settings_env(json.load(f))
        except Exception as e:
            logger.error(f"Error in the configuration: {e}")
    
    # Priority: Command line arguments > configuration file > default values
    weaviate_url = args.weaviate_url or config.get("WEAVIATE_URL") or os.environ.get("WEAVIATE_URL")
    weaviate_api_key = args.weaviate_api_key or config.get("WEAVIATE_API_KEY") or os.environ.get("WEAVIATE_API_KEY")
    openai_api_key = args.openai_api_key or config.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if not all((weaviate_url, weaviate_api_key, openai_api_key)):
        parser.error("Configure WEAVIATE_URL, WEAVIATE_API_KEY and OPENAI_API_KEY through an approved local configuration or process environment")

    try:
        logger.info("Connecting to the explicitly configured Weaviate instance")
        
        # Connect using API version 4
        client = connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(api_key=weaviate_api_key),
            headers={
                "X-OpenAI-Api-Key": openai_api_key,
                "X-OpenAI-BaseURL": "https://api.openai.com/v1/"
            }
        )
        
        # Get a collection list.
        collections = client.collections.list_all()
        logger.info(f"Type of coalitions: {type(collections)}")
        logger.info(f"The contents of the permits are: {collections}")

        # In a newer version of API, collections can be a dictionary
        try:
            # In version 4 API collections returns as a dictionary with collection names as keys
            if isinstance(collections, dict):
                collection_names = list(collections.keys())
            else:
                collection_names = [str(collections)]  # In case the format is completely unexpected.
            
            logger.info(f"Connectivity Successful. Available collections: {collection_names}")
        except Exception as e:
            logger.error(f"Error in processing a list of collections: {e}")
            collection_names = []
        
        # Checking FAQ Collection
        if "FAQ" not in collection_names:
            logger.error(f"The FAQ collection is not found in the Weaviate!")
            print('Error: FAQ is not found in the database. First upload the data using load_faq_to_weaviate.py')
            client.close()
            return 1
        
        logger.info(f"FAQ Collection Found. Move on to search.")
        logger.info(f"Search request: '{args.query}'")
        logger.info(f"User type: {'Student (authorised)' if args.authorized else 'Apprentice (unauthorised)'}")
        
        # Search using search_faq
        results = search_faq(
            query=args.query,
            weaviate_client=client,
            is_authorized=args.authorized,
            limit=5,
            min_score=0.5,
            class_name="FAQ"
        )
        
        # Get the results out.
        if results:
            print('\n== sync, corrected by elderman == @elder_man')
            print(results)
            print("\n=========================")
        else:
            print("\nThere's nothing on your request.")
        
        # Close the connection.
        client.close()
        
    except Exception as e:
        logger.error(f"Error when connecting or searching: {e}")
        import traceback
        logger.error(f"Trace: {traceback.format_exc()}")
        print(f"There was a mistake: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
