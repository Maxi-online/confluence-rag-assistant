#!/usr/bin/env python3
"""
Script to check the number of objects in the FAQ class at the Weaviate.
"""
import os

import logging
import weaviate
from weaviate import connect_to_weaviate_cloud, connect_to_local
from weaviate.auth import AuthApiKey

# Logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Connectivity Options
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "FAQ"

def main():
    """
    The main function is to check the number of objects.
    """
    try:
        # Connect to Weaviate
        client = connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY)
        )
        
        logger.info(f"We've got a connection to the Web: {WEAVIATE_URL}")
        
        try:
            # Checking the existence of the class by trying to get a collection
            collection = client.collections.get(CLASS_NAME)
            logger.info(f"Grade {CLASS_NAME} exists")
            
            # Get the number of objects.
            total_count = collection.aggregate.over_all().total_count
            logger.info(f"Number of objects in class {CLASS_NAME}: {total_count}")
            
            # We receive examples of objects.
            objects = collection.query.fetch_objects(limit=5)
            logger.info(f"Received {len(objects)} of inspection facilities")
            
            if objects:
                for i, obj in enumerate(objects):
                    logger.info(f"\n Subject #{i+1}:")
                    for key, value in obj.properties.items():
                        logger.info(f"  {key}: {value}")
            else:
                logger.info('No facilities')
        except Exception as class_error:
            logger.error(f"Grade {CLASS_NAME} is not available or cannot be obtained: {class_error}")
        
        # Checking all classes.
        try:
            all_collections = client.collections.list_all()
            logger.info(f"All classes at Weaviate: {[c.name for c in all_collections]}")
        except Exception as list_error:
            logger.error(f"Error in obtaining a class list: {list_error}")
        
        # Close the connection.
        client.close()
        logger.info('Audit completed')
        
    except Exception as e:
        logger.error(f"Error in checking objects: {e}")
        import traceback
        logger.error(f"Trace: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 