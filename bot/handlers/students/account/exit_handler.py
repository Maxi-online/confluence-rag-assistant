from aiogram import F, Router
from aiogram import types

# from services.database.models import User
from bot.keyboards.client import UserKeyboard

exit_handler_router = Router()


# TODO: Narek says we need to upgrade the Uzer to Pearson, but there's no field in Person's class right now.
# There's a is_registered_in_telegram method in person, but we need to test his logic
@exit_handler_router.message(F.text=='Exit')
async def exit_cmd(message: types.Message):
    user_tg_id = message.from_user.id
    is_registered = await User.is_registered(user_tg_id)
    if not is_registered:
        User.register(
            user_tg_id=user_tg_id,
            username=message.from_user.username
        )

    is_authorized = await User.is_authorized(message.from_user.id)

    if is_authorized:
        await User.deauthorize(message.from_user.id)
        await message.answer("You're a success.", reply_markup=UserKeyboard.get_auth_kb())
    else:
        await message.answer("You haven't been authorized yet.", reply_markup=UserKeyboard.get_auth_kb())


def register_exit_handler_cmd(dp):
    dp.include_router(exit_handler_router)
