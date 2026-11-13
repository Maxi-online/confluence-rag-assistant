from aiogram import Router
from aiogram import types

from bot.keyboards.client import UserKeyboard

help_cmd_router = Router()


@help_cmd_router.message(lambda message: message.text == "/help")
async def help_cmd(message: types.Message):
    await message.answer('Инструкция для Нейрокуратор Example Organization', reply_markup=UserKeyboard.get_instructions_kb())


def register_help_cmd(dp):
    dp.include_router(help_cmd_router)
