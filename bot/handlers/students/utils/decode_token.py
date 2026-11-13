import jwt

from backend.logger import logger


def decode_token(token):
    try:
        decoded = jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256", "RS256"])
        return decoded
    except jwt.PyJWTError as e:
        logger.error(f"Ошибка при декодировании токена: {e}")
        return None
    except Exception as e:
        logger.error(f"Неизвестная ошибка при декодировании токена: {e}")
        return None
