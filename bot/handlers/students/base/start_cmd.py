from aiogram import F, Router
from aiogram import types
from aiogram.filters import CommandStart

from backend.db.model.chat_models import ContactType
from backend.logger import logger
from backend.service.chat.process_chat_message import process_message_from_user
from bot.keyboards.client import UserKeyboard


start_cmd_router = Router()


@start_cmd_router.message(CommandStart())
async def start_cmd(message: types.Message):
    user_id = message.from_user.id

    chat_message = (f'Hey, {message.from_user.first_name}!\n'
                    f'I\'m your NeuroCounter who\'s gonna help with questions and problems about the corporation '
                    f'Example Organization. Please authorise so I can process your requests. '
                    f'Enter your LMS/EFC records.')

    # answer to user
    await message.answer(chat_message, reply_markup=UserKeyboard.get_auth_kb())

    # create chat message or user+contact+message
    await process_message_from_user(
        contact_value=str(user_id),
        contact_type=ContactType.TELEGRAM,
        chat_message=chat_message,
    )
    logger.info(f"User {user_id} started conversation with bot.")


def register_start_cmd(dp):
    dp.include_router(start_cmd_router)
