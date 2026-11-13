import json

from aiogram import F, Router
from aiogram import types

from backend.db.model.chat_models import ContactType
from backend.service.chat.process_chat_message import save_service_message_to_contact, get_or_create_chat_contact
from backend.service.chat.process_chat_user import get_chat_user_by_organization_id, create_chat_user_with_organization_id
from backend.service.chat.process_contact import update_chat_contact_user_id
from bot.handlers.students.utils.decode_token import decode_token
from bot.keyboards.client import UserKeyboard
from backend.logger import logger

authorize_handler_router = Router()


@authorize_handler_router.message(F.web_app_data)
async def authorize_cmd(message: types.Message):
    user_tg_id = message.from_user.id
    logger.info(f"Получен запрос на авторизацию от пользователя {user_tg_id}")

    contact = await get_or_create_chat_contact(contact_value=str(user_tg_id), contact_type=ContactType.TELEGRAM)

    # Извлечение данных из web_app_data
    data = message.web_app_data.data
    data_dict = json.loads(data)
    logger.info(f"Получены данные из web_app_data: {data_dict}")

    access_token = data_dict.get('access_token')
    if not access_token:
        await message.answer('Не удалось получить токен доступа.', reply_markup=UserKeyboard.get_auth_kb())
        return None

    decoded_access_token = decode_token(access_token)
    logger.info(f"Декодированный токен: {decoded_access_token}")

    if decoded_access_token is None:
        await message.answer('Не удалось декодировать токен доступа.', reply_markup=UserKeyboard.get_auth_kb())
        return None

    organization_id = decoded_access_token.get('sub')
    if organization_id:
        logger.info(f"Получен ExampleOrganization ID: {organization_id}")
        chat_message = f"Авторизация прошла успешно."

        # Here we need to search for user by contact_id first. If not found, we need to search users by organization_id.
        # If exists, we need to update chat_contact with user_id,
        # if not - create new user and update chat_contact with user_id.
        if not contact.chat_user_id:  # If contact is not linked to user, it means that the contact is new
            user = await get_chat_user_by_organization_id(organization_id)  # Search user by organization_id if the user exists
            if not user:  # if the user is not exists
                user = await create_chat_user_with_organization_id(organization_id=organization_id)  # Create new user

            contact = await update_chat_contact_user_id(contact_id=contact.id, chat_user_id=user.id)

        await message.answer(chat_message)
        await save_service_message_to_contact(contact_id=contact.id, chat_message=chat_message)
        logger.info(f"Пользователь c tg_id '{user_tg_id}' авторизован.")
        return None

    logger.error(f"Не удалось получить organization_id для пользователя c tg_id '{user_tg_id}'")
    await message.answer('Не удалось получить данные пользователя.', reply_markup=UserKeyboard.get_auth_kb)
    return None


def register_authorize_handler_cmd(dp):
    dp.include_router(authorize_handler_router)
