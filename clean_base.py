import os
import logging
import sys
from dotenv import load_dotenv
import weaviate
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey

# Loading the environment variables
load_dotenv()

# Logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Getting an uRL and an API key for the Web
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not (WEAVIATE_URL and WEAVIATE_API_KEY):
    logging.error('WEAVIATE_URL or WEAVIATE_API_KEY environment variables not installed.')
    sys.exit(1)

# Connecting to the Weaviate Cloud (v4)
client = connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY)
)

try:
    # Remove all collections (classes) and data in them
    client.collections.delete_all()
    logging.info('The database is completely cleaned: all collections and their contents are removed.')
except Exception as e:
    logging.error(f"Error in cleaning the database: {e}")
finally:
    client.close()
    logging.info("The connection's closed.")
