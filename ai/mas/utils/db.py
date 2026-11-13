"""
Функции для работы с базой данных (Weaviate).
Реализованы функции для сохранения данных пользователя (UserProfile).
В данном примере используется глобальный кэш (словарь) для демонстрации постоянного хранения.
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальный кэш профилей (имитация БД)
USER_PROFILES: Dict[str, Dict[str, Any]] = {}

USER_PROFILE_CLASS = "UserProfile"

def store_user_data_in_db(user_profile_id: str, data: dict) -> None:
    """
    Сохраняет промежуточные данные для пользователя.
    Здесь используется логирование для демонстрации.
    
    :param user_profile_id: Идентификатор пользователя.
    :param data: Словарь с данными для сохранения.
    """
    logger.info(f"[DB] Сохранены данные для {user_profile_id}: {data}")

def update_language_in_context(client: Any, user_profile_id: str, language_code: str) -> None:
    """
    Обновляет информацию о языке в профиле пользователя.
    
    :param client: Клиент для работы с профилем (не используется, работает с глобальным кэшем).
    :param user_profile_id: Идентификатор пользователя.
    :param language_code: Новый код языка.
    """
    profile = USER_PROFILES.get(user_profile_id, {})
    profile["detected_language"] = language_code
    USER_PROFILES[user_profile_id] = profile
    logger.info(f"[DB] Обновлен язык для {user_profile_id}: {language_code}")

def get_user_profile(client: Any, user_profile_id: str) -> Dict[str, Any]:
    """
    Получает профиль пользователя из глобального кэша.
    
    :param client: Клиент (не используется).
    :param user_profile_id: Идентификатор пользователя.
    :return: Словарь с данными профиля или пустой словарь, если профиль не найден.
    """
    return USER_PROFILES.get(user_profile_id, {})

def update_user_profile(client: Any, user_profile_id: str, data: dict) -> None:
    """
    Обновляет или создает профиль пользователя в глобальном кэше.
    
    :param client: Клиент (не используется).
    :param user_profile_id: Идентификатор пользователя.
    :param data: Словарь с данными для обновления.
    """
    profile = USER_PROFILES.get(user_profile_id, {})
    profile.update(data)
    USER_PROFILES[user_profile_id] = profile
    logger.info(f"[DB] Профиль для {user_profile_id} обновлен: {profile}")
