import os
import logging
import sys
from dotenv import load_dotenv
import weaviate
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Получаем URL и API-ключ для Weaviate
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not (WEAVIATE_URL and WEAVIATE_API_KEY):
    logging.error("Не установлены переменные окружения WEAVIATE_URL или WEAVIATE_API_KEY.")
    sys.exit(1)

# Подключаемся к Weaviate Cloud (v4)
client = connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY)
)

try:
    # Удаляем все коллекции (классы) и данные в них
    client.collections.delete_all()
    logging.info("База данных полностью очищена: все коллекции и их содержимое удалены.")
except Exception as e:
    logging.error(f"Ошибка при очистке базы данных: {e}")
finally:
    client.close()
    logging.info("Соединение закрыто.")
