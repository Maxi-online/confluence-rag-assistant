from aiogram import Router
from aiogram import types

from bot.keyboards.client import UserKeyboard

menu_cmd_router = Router()


@menu_cmd_router.message(lambda message: message.text == "/menu")
async def menu_cmd(message: types.Message):
    await message.answer('Меню Нейрокуратора', reply_markup=UserKeyboard.get_menu_kb())


def register_menu_cmd(dp):
    dp.include_router(menu_cmd_router)
