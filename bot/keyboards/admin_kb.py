from aiogram.types import InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


# Todo Narek and Genya: current logic does not provide for Admin, functionally deleted if Elijah confirms
class AdminKB:
    def get_main_admin_kb():
        admin_kb = InlineKeyboardBuilder()

        admin_kb.row(
            InlineKeyboardButton(text='➕ Add Admin', callback_data='add_admin'),
            InlineKeyboardButton(text='➖ Remove Admin', callback_data='remove_admin'),
        )
        admin_kb.row(
            InlineKeyboardButton(text='👥 Number of users', callback_data='user_count'),
        )
        admin_kb.row(
            InlineKeyboardButton(text='📜 Release all users', callback_data='list_users'),
        )

        return admin_kb
