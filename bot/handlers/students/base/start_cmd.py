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

    chat_message = (f'Привет, {message.from_user.first_name}!\n'
                    f'Я - твой Нейроконсультант, который поможет с вопросами и проблемами, касающиеся корпорации '
                    f'Example Organization. Пожалуйста, авторизуйтесь, чтобы я смог обрабатывать ваши запросы. '
                    f'Вводите данные от ваших учетных записей LMS/ЕЛК.')

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
