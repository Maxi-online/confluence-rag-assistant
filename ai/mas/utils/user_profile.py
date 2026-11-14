import logging
from datetime import datetime
import weaviate
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey

logger = logging.getLogger(__name__)

class UserProfileWeaviateClient:
    def __init__(self, weaviate_url: str, weaviate_api_key: str, openai_api_key: str, class_name: str = "UserProfile"):
        """
        Initiating client to work with user profile.
        
        :param weaviate_url: URL Weaviate server.
        :param weaviate_api_key: API key Weaviate.
        :Param openai_api_key: API-key OpenAI (used for transmission in the title).
        :Param class_name: Name of the identity storage class (by default "UserProfile"").
        """
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.openai_api_key = openai_api_key
        self.class_name = class_name

        self.client = connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=AuthApiKey(api_key=self.weaviate_api_key),
            headers={"X-OpenAI-Api-Key": self.openai_api_key},
            skip_init_checks=True
        )
        self.ensure_schema_exists()

    def ensure_schema_exists(self) -> None:
        """
        Checks whether the UserProfile class exists in the Web and, if not, creates it.
        """
        try:
            self.client.collections.get(self.class_name)
            logger.info(f"Grade {self.class_name} It already exists.")
        except Exception:
            logger.info(f"Grade {self.class_name} Not found. I'm making a scheme...")
            schema_obj = {
                "class": self.class_name,
                "description": 'User Profile',
                "vectorizer": "none",  # No vectorization required for structured data
                "properties": [
                    {"name": "user_profile_id", "dataType": ["text"]},
                    {"name": "user_name", "dataType": ["text"]},
                    {"name": "contact_phone", "dataType": ["text"]},
                    {"name": "email", "dataType": ["text"]},
                    {"name": "how_heard_about_us", "dataType": ["text"]},
                    {"name": "program_level", "dataType": ["text"]},
                    {"name": "program_area", "dataType": ["text"]},
                    {"name": "study_format", "dataType": ["text"]},
                    {"name": "budget_preference", "dataType": ["text"]},
                    {"name": "start_date", "dataType": ["date"]},
                    {"name": "updated_at", "dataType": ["date"]}
                ]
            }
            self.client.collections.create_from_dict(schema_obj)
            logger.info(f"Grade {self.class_name} It's a success story.")

    def upsert_user_profile(self, user_profile_id: str, profile_data: dict) -> None:
      """
      Creates or updates a user profile in the Web.
      If the profile exists, the merge with new data is executed and the replay is called.
      If the profile is not available, it is executed insert.
      """
      # Update the time of last update
      profile_data["updated_at"] = datetime.utcnow().isoformat()
      profile_data["user_profile_id"] = user_profile_id

      query = f'''
      {{
        Get {{
          {self.class_name}(where: {{
            path: ["user_profile_id"],
            operator: Equal,
            valueString: "{user_profile_id}"
          }}) {{
            user_profile_id
            _additional {{
              id
            }}
          }}
        }}
      }}
      '''
      try:
          result = self.client.graphql_raw_query(query)
          docs = result.get("data", {}).get("Get", {}).get(self.class_name, [])
          if docs:
              doc_id = docs[0]["_additional"]["id"]
              # Use self.client.data instead of self.client.data_object
              self.client.data.replace(
                  data_object=({**docs[0], **profile_data}),
                  class_name=self.class_name,
                  uuid=doc_id
              )
              logger.info(f"User Profile {user_profile_id} updated (uwid={doc_id}).")
          else:
              self.client.data.insert(profile_data, uuid=user_profile_id)
              logger.info(f"User Profile {user_profile_id} It's created.")
      except Exception as e:
          logger.error(f"Error in updating the profile for {user_profile_id}: {e}")

    def get_user_profile(self, user_profile_id: str) -> dict:
        """
        Returns the user profile to user_profile_id.
        
        :Paragraph user_profile_id: unique user identifier.
        :Return: Profile dictionary or empty dictionary if no profile is found.
        """
        query = f'''
        {{
          Get {{
            {self.class_name}(where: {{
              path: ["user_profile_id"],
              operator: Equal,
              valueString: "{user_profile_id}"
            }}) {{
              user_profile_id
              user_name
              contact_phone
              email
              how_heard_about_us
              program_level
              program_area
              study_format
              budget_preference
              start_date
              updated_at
            }}
          }}
        }}
        '''
        try:
            result = self.client.graphql_raw_query(query)
            docs = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            if docs:
                return docs[0]
            return {}
        except Exception as e:
            logger.error(f"Error in obtaining the user profile {user_profile_id}: {e}")
            return {}
