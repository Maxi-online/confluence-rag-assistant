# default_handler.py

from datetime import datetime
from aiogram import F, Router, types
from backend.logger import logger
from backend.service.chat.process_chat_message import (
    get_or_create_chat_contact,
    save_question_message_from_user,
    save_bot_response_to_user_question,
)
from backend.db.model.chat_models import ContactType
from backend.service.chat.process_chat_user import get_chat_user_by_id
from backend.service.manager.process_manager_message import create_message_for_manager
from bot.keyboards.client import UserKeyboard
# Импортируем вашу функцию get_gpt_answer из gpt_agent.py
from ai.answer.master_agent import get_gpt_answer

default_handler_router = Router()

async def process_request(message: types.Message):
    user_tg_id = message.from_user.id
    chat_message = message.text
    logger.info(f"Получен вопрос '{chat_message}' от пользователя {user_tg_id}")

    # Ищем или создаём контакт
    contact = await get_or_create_chat_contact(contact_value=str(user_tg_id), contact_type=ContactType.TELEGRAM)
    user = await get_chat_user_by_id(contact.chat_user_id)
    question_message = await save_question_message_from_user(contact_id=contact.id, chat_message=chat_message)

    # Проверка авторизации
    if not user.organization_id:
        await message.answer(
            'Пожалуйста, авторизуйтесь, чтобы я мог отвечать на ваши запросы.',
            reply_markup=UserKeyboard.get_auth_kb()
        )
        logger.info(f"Пользователь с id '{user.id}' и user_tg_id '{user_tg_id}' не авторизован для вопроса боту.")
        return

    # Проверка согласия на обработку данных (если нужно)
    if not user.is_agreed:
        # Здесь можно отправить клавиатуру соглашения
        pass

    # Отправляем вопрос в GPT-агент
    try:
        bot_message = await get_gpt_answer(chat_message)
        logger.info(f"Получен ответ '{bot_message}' от GPT-агента для пользователя {user_tg_id}")

        if bot_message and len(bot_message) > 0:
            answer = bot_message
        else:
            answer = (
                'Извините, база знаний активно пополняется и пока не содержит ответ на ваш вопрос. '
                'Мы передали его менеджеру. Пожалуйста, дождитесь ответа.'
            )
            # Создаём задачу для менеджера
            message_to_save = (
                f"Вопрос от пользователя: '{chat_message}'\n"
                f"Ответ бота: '{answer}'\n"
                f"Время создания вопроса: '{datetime.now()}'"
            )
            await create_message_for_manager(message=message_to_save, chat_contact_id=contact.id)

    except Exception as e:
        logger.error(f"Ошибка при обращении к GPT-агенту: {e}")
        answer = "Извините, произошла ошибка при генерации ответа."

    # Сохраняем ответ бота
    await save_bot_response_to_user_question(
        message_id=question_message.id,
        answer=answer,
    )
    # Отправляем ответ пользователю
    await message.answer(answer)


@default_handler_router.message(F.text)
async def request_in_chatgpt_cmd(message: types.Message):
    await process_request(message)


def register_default_handler_cmd(dp):
    dp.include_router(default_handler_router)
