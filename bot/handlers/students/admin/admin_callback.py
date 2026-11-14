from aiogram import F, Router
from aiogram.types import CallbackQuery
from aiogram.types.input_file import FSInputFile

from bot.create_bot import dp, bot
from bot.keyboards import admin_kb

admin_callback_router = Router()


@admin_callback_router.callback_query(F.data == 'list_users')
async def get_info_on_all_users_callback(callback_query: CallbackQuery):
    await callback_query.message.answer(f'User list.')
    await callback_query.answer()


def register_admin_callback_cmd(dp):
    dp.include_router(admin_callback_router)
