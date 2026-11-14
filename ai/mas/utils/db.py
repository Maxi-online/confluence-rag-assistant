"""
Functions for working with the database (Weaviate).
UserProfile (UserProfile) data storage functions implemented).
This example uses global cache (slovar) to demonstrate permanent storage.
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Cash Profiles (database Simulation)
USER_PROFILES: Dict[str, Dict[str, Any]] = {}

USER_PROFILE_CLASS = "UserProfile"

def store_user_data_in_db(user_profile_id: str, data: dict) -> None:
    """
    Saves intermediate data for the user.
    The logic is used here to demonstrate.
    
    :param user_profile_id: User Identifier.
    :param data: Data dictionary for preservation.
    """
    logger.info(f"[DB] Saved data for {user_profile_id}: {data}")

def update_language_in_context(client: Any, user_profile_id: str, language_code: str) -> None:
    """
    Updates the language information in the user profile.
    
    :param client: Client for profile management (not used, global cache)).
    :param user_profile_id: User Identifier.
    :param language_code: New language code.
    """
    profile = USER_PROFILES.get(user_profile_id, {})
    profile["detected_language"] = language_code
    USER_PROFILES[user_profile_id] = profile
    logger.info(f"[DB] Updated language for {user_profile_id}: {language_code}")

def get_user_profile(client: Any, user_profile_id: str) -> Dict[str, Any]:
    """
    Gets a user profile from a global cache.
    
    :param client: Client (not used)
    :param user_profile_id: User Identifier.
    :Return: Profile dictionary or empty dictionary if no profile is found.
    """
    return USER_PROFILES.get(user_profile_id, {})

def update_user_profile(client: Any, user_profile_id: str, data: dict) -> None:
    """
    Updates or creates a user profile in global cache.
    
    :param client: Client (not used)
    :param user_profile_id: User Identifier.
    :param data: A dictionary with data for updating.
    """
    profile = USER_PROFILES.get(user_profile_id, {})
    profile.update(data)
    USER_PROFILES[user_profile_id] = profile
    logger.info(f"[DB] Profile for {user_profile_id} updated: {profile}")
