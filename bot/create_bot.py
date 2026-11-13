from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.bot import DefaultBotProperties

from backend.settings import settings


bot = Bot(
    token=settings.telegram.bot_token,
    default=DefaultBotProperties(
        parse_mode=ParseMode.MARKDOWN
    )
)

dp = Dispatcher()
