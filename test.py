import os
import weaviate
import logging
import json
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)

def search_documents(query_text: str, host: str = "localhost", port: int = 8080):
    """
    Searches the existing ConfluenceSection_v2 class in a local copy of the Weaviate.
    Request embeddings are calculated on the client using OpenAIEmbeddings (through proxy)),
    and then search through near_vector.
    Returns the JSON line with the results of the fields: title, column, pageId and _additional.score.
    
    It is assumed that the ConfluenceSection_v2 class already exists in the Weaviate and uses 1536 vectors.
    """
    # Ask for the OpenAI key (let's leave it in the code)
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        logging.error('OpenAI API key not specified.')
        return json.dumps([])

    # Set the environment variables for the operation of the text2vec-openai module:
    # IMPORTANT: Use the OPENAI_API_BASE_URL variable (without final slash))
    os.environ["OPENAI_APIKEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_BASE_URL"] = "https://api.example.org/v1"

    logging.info('Connecting to a local copy of the Weaviate...')
    try:
        client = weaviate.connect_to_local(
            host=host,
            port=port,
            skip_init_checks=True
        )
        logging.info('The connection has been successfully installed.')
    except Exception as e:
        logging.error(f"Connection error to Weaviate: {e}")
        return json.dumps([])

    # We calculate the embedding of the request through OpenAIEmbeddings (text-embedding-ada-002 model, returning a 1536-dimensional vector)
    try:
        embeddings_client = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base="https://api.example.org/v1"
        )
        query_vector = embeddings_client.embed_query(query_text)
        logging.info(f"Embeding requests successfully calculated. Size: {len(query_vector)}")
    except Exception as e:
        logging.error(f"Error in calculating embeddings: {e}")
        client.close()
        return json.dumps([])

    # Search near_vector using a computed vector
    try:
        # Getting an object of the collection through a current call
        collection = client.collections.get("ConfluenceSection_v2")
        # Create and execute a search request
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=5,
            return_properties=["title", "content", "pageId"]
        ).with_additional_fields(["score"])
    except Exception as e:
        logging.error(f"Error in Request to Weaviate: {e}")
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
        logging.error(f"Error in response processing: {e}")
        client.close()
        return json.dumps([])

    logging.info(f"Documents found: {len(results)}")
    client.close()
    return json.dumps(results, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    query = 'What are the UAV courses?'
    docs = search_documents(query)
    print(docs)
