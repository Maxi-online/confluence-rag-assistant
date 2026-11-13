from aiogram import Router
from aiogram import types

from bot.create_bot import bot
from bot.keyboards.client import UserKeyboard

logout_cmd_router = Router()


# todo Нарек: Нужно импортировать Персон и добавить эвэйт на логаут
@logout_cmd_router.callback_query(lambda query: query.data == "logout_cmd")
async def logout_cmd(callback_query: types.CallbackQuery):
    await callback_query.message.answer('Вы успешно вышли.', reply_markup=UserKeyboard.get_auth_kb())
    await bot.delete_message(callback_query.message.chat.id, callback_query.message.message_id)


def register_logout_cmd(dp):
    dp.include_router(logout_cmd_router)
