from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter, TelegramForbiddenError, TelegramBadRequest

from bot.create_bot import bot
from backend.logger import logger


async def send_message_telegram(contact_id: int, contact_value: str, message: str) -> bool:
    """
    Sends a message via Telegram bot.

    Returns:
        True  -> if message was sent successfully.
        False -> if message sending failed.
    """
    logger.info(f"Sending message to contact '{contact_id}' via Telegram: message '{message}'")

    # if notification.image is None and notification.video is None:
    try:
        sent_message = await bot.send_message(contact_value, message)
        return True
    except TelegramRetryAfter as e:
        logger.error(f"Flood control: retry in some seconds.")
        return False
    except TelegramForbiddenError as e:
        logger.error(f"Bot was blocked by the user: contact id '{contact_id}' and value '{contact_value}'")
        return False
    except TelegramBadRequest as e:
        logger.error(f"Bad request: {e}")
        return False
    except TelegramAPIError as e:
        logger.error(f"Telegram API error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send message to contact '{contact_id}' via Telegram: {e}")
        return False
