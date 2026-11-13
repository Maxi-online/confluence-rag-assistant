from aiogram import Router
from aiogram import types

from bot.create_bot import bot
from bot.keyboards.client import UserKeyboard
# from services.database.models import User
from services.database.models import Person

agreement_cmd_router = Router()


@agreement_cmd_router.callback_query(lambda query: query.data == "agreement_cmd")
async def agreement_cmd(callback_query: types.CallbackQuery):
    await Person.accept_agreement(callback_query.from_user.id)
    await callback_query.message.answer('Вы дали своё согласие на обработку данных.')
    await bot.delete_message(callback_query.message.chat.id, callback_query.message.message_id)


def register_agreement_cmd(dp):
    dp.include_router(agreement_cmd_router)
