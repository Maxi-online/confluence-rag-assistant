from aiogram.types import InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


# todo Нарек и Женя: текущая логика не предусматривает наличия Админа, функционал удалить, если Илья подтверждает
class AdminKB:
    def get_main_admin_kb():
        admin_kb = InlineKeyboardBuilder()

        admin_kb.row(
            InlineKeyboardButton(text='➕ Добавить админа', callback_data='add_admin'),
            InlineKeyboardButton(text='➖ Удалить админа', callback_data='remove_admin'),
        )
        admin_kb.row(
            InlineKeyboardButton(text='👥 Количество пользователей', callback_data='user_count'),
        )
        admin_kb.row(
            InlineKeyboardButton(text='📜 Вывести всех пользователей', callback_data='list_users'),
        )

        return admin_kb
