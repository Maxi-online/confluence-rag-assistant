from aiogram import types
from aiogram.types.web_app_info import WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder


class UserKeyboard():
    def get_menu_kb():
        profile_button = types.InlineKeyboardButton(
            text='Profile', 
            callback_data='profile_cmd'
        )
        logout_button = types.InlineKeyboardButton(
            text='Exit', 
            callback_data='logout_cmd'
        )
        instructions_button = types.InlineKeyboardButton(
            text='Instruction', 
            url='https://example.org/knowledge'
        )

        menu_kb = types.InlineKeyboardMarkup(inline_keyboard=[[profile_button, instructions_button], [logout_button]])
        return menu_kb


    def get_instructions_kb():
        instructions_button = types.InlineKeyboardButton(
            text='Instruction', 
            url='https://example.org/knowledge'
        )

        instructions_kb = types.InlineKeyboardMarkup(inline_keyboard=[[instructions_button]])
        return instructions_kb


    def get_auth_kb():
        auth_button = types.KeyboardButton(
            text='Authorize',
            web_app=WebAppInfo(url='https://assistant.example.org/tg/')
        )

        auth_kb = types.ReplyKeyboardMarkup(
            keyboard=[[auth_button]],
            resize_keyboard=True,
            one_time_keyboard=True
        )
        return auth_kb


    def get_agreement_kb():
        agreement_button = types.InlineKeyboardButton(
            text='Accept', 
            callback_data='agreement_cmd',
        )

        agreement_kb = types.InlineKeyboardMarkup(inline_keyboard=[[agreement_button]])
        return agreement_kb
