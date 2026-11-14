#!/usr/bin/env python3
"""
Script to remove and recreate the FAQ class in Weaviate with correct vector settings.
"""

import os
import logging
import argparse
import weaviate
from weaviate import connect_to_weaviate_cloud, connect_to_local
from weaviate.auth import AuthApiKey

# Logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The default parameters
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CLASS_NAME = "FAQ"
EXPECTED_DIM = 3072

def reset_faq_class(force: bool = False):
    """
    Removes the FAQ class from the Web, if it exists, and recreates it with correct settings.
    
    :param force: Forced disposal without confirmation
    """
    try:
        # Connect to Weaviate
        client = connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
        )
        
        # Checking the existence of the class
        class_exists = False
        try:
            collection = client.collections.get(CLASS_NAME)
            class_exists = True
            logger.info(f"Grade {CLASS_NAME} It's found in Weaviate.")
        except Exception:
            logger.info(f"Grade {CLASS_NAME} It's not found in Weaviate.")
        
        if class_exists:
            if not force:
                confirm = input(f"You're sure you want to delete the class. {CLASS_NAME}? (y/n): ")
                if confirm.lower() != 'y':
                    logger.info('Operation canceled by user.')
                    return
            
            # Remove class
            try:
                client.collections.delete(CLASS_NAME)
                logger.info(f"Grade {CLASS_NAME} has been successfully removed from the Weaviate.")
            except Exception as e:
                logger.error(f"Error in removing class {CLASS_NAME}: {e}")
                return
        
        # Create FAQ class with correct settings
        schema_obj = {
            "class": CLASS_NAME,
            "description": 'FAQ knowledge base with user type split',
            "vectorizer": "text2vec-openai",  # Point out the vector.
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
                {"name": "question", "dataType": ["text"], "description": 'User Question', "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "answer", "dataType": ["text"], "description": 'Answer to the question', "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "keywords", "dataType": ["text[]"], "description": 'Keywords for Search', "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "topic", "dataType": ["text"], "description": "Тема запроса", "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "main_point", "dataType": ["text"], "description": 'Key point', "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "user_type", "dataType": ["text"], "description": 'Type of user (academic/student)', "moduleConfig": {"text2vec-openai": {"skip": False, "vectorizePropertyName": False}}},
                {"name": "implementation_period", "dataType": ["text"], "description": "Срок реализации", "moduleConfig": {"text2vec-openai": {"skip": True}}}
            ]
        }
        
        # Create a class with a new scheme using a new API
        client.collections.create_from_dict(schema_obj)
        logger.info(f"Grade {CLASS_NAME} It's been successfully designed with the right vector settings.")
        
        # Close the connection.
        client.close()
        
    except Exception as e:
        logger.error(f"Mistakes in working with Weaviate: {e}")
        import traceback
        logger.error(f"Trace: {traceback.format_exc()}")
        try:
            client.close()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='FAQ Class Reset in Weaviate')
    parser.add_argument('--force', action='store_true', help='Forced disposal without confirmation')
    
    args = parser.parse_args()
    reset_faq_class(args.force)

if __name__ == "__main__":
    main() 