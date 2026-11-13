from aiogram import Router
from aiogram import types

from bot.create_bot import bot
from bot.keyboards.client import UserKeyboard

profile_cmd_router = Router()


@profile_cmd_router.callback_query(lambda query: query.data == "profile_cmd")
async def profile_cmd(callback_query: types.CallbackQuery):
    await callback_query.message.answer(f'Ваш профиль.')
    await callback_query.answer()


def register_profile_cmd(dp):
    dp.include_router(profile_cmd_router)
