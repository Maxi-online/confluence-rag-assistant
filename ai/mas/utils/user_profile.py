import logging
from datetime import datetime
import weaviate
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey

logger = logging.getLogger(__name__)

class UserProfileWeaviateClient:
    def __init__(self, weaviate_url: str, weaviate_api_key: str, openai_api_key: str, class_name: str = "UserProfile"):
        """
        Инициализирует клиента для работы с профилем пользователя.
        
        :param weaviate_url: URL сервера Weaviate.
        :param weaviate_api_key: API-ключ Weaviate.
        :param openai_api_key: API-ключ OpenAI (используется для передачи в заголовке).
        :param class_name: Название класса для хранения профилей (по умолчанию "UserProfile").
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
        Проверяет, существует ли класс UserProfile в Weaviate, и если нет, создает его.
        """
        try:
            self.client.collections.get(self.class_name)
            logger.info(f"Класс {self.class_name} уже существует.")
        except Exception:
            logger.info(f"Класс {self.class_name} не найден. Создаю схему...")
            schema_obj = {
                "class": self.class_name,
                "description": "Профиль пользователя",
                "vectorizer": "none",  # Для структурированных данных векторизация не требуется
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
            logger.info(f"Класс {self.class_name} успешно создан.")

    def upsert_user_profile(self, user_profile_id: str, profile_data: dict) -> None:
      """
      Создает или обновляет профиль пользователя в Weaviate.
      Если профиль существует, выполняется merge с новыми данными и вызывается replace.
      Если профиль отсутствует, выполняется insert.
      """
      # Обновляем время последнего обновления
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
              # Используем self.client.data вместо self.client.data_object
              self.client.data.replace(
                  data_object=({**docs[0], **profile_data}),
                  class_name=self.class_name,
                  uuid=doc_id
              )
              logger.info(f"Профиль пользователя {user_profile_id} обновлен (uuid={doc_id}).")
          else:
              self.client.data.insert(profile_data, uuid=user_profile_id)
              logger.info(f"Профиль пользователя {user_profile_id} создан.")
      except Exception as e:
          logger.error(f"Ошибка обновления профиля для {user_profile_id}: {e}")

    def get_user_profile(self, user_profile_id: str) -> dict:
        """
        Возвращает профиль пользователя по user_profile_id.
        
        :param user_profile_id: Уникальный идентификатор пользователя.
        :return: Словарь с данными профиля или пустой словарь, если профиль не найден.
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
            logger.error(f"Ошибка получения профиля пользователя {user_profile_id}: {e}")
            return {}
