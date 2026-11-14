from aiogram import F, Router
from aiogram.types import Message

from bot.create_bot import dp
from bot.keyboards import admin_kb

admin_cmd_router = Router()


@admin_cmd_router.message(F.text=="/to_admin")
async def login_to_main_admin_panel_cmd(message: Message):
    await message.answer("The administrator's board.", reply_markup=admin_kb.AdminKB.get_main_admin_kb().as_markup())


def register_admin_cmd(dp):
    dp.include_router(admin_cmd_router)
