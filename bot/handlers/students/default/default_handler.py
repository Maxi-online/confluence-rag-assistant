# default_handler.py



from datetime import datetime

from aiogram import F, Router, types

from backend.logger import logger

from backend.service.chat.process_chat_message import (

    get_or_create_chat_contact,

    save_question_message_from_user,

    save_bot_response_to_user_question,

)

from backend.db.model.chat_models import ContactType

from backend.service.chat.process_chat_user import get_chat_user_by_id

from backend.service.manager.process_manager_message import create_message_for_manager

from bot.keyboards.client import UserKeyboard

# Import your get_gpt_answer function from gpt_agent.py

from ai.answer.master_agent import get_gpt_answer



default_handler_router = Router()



async def process_request(message: types.Message):

    user_tg_id = message.from_user.id

    chat_message = message.text

    logger.info(f"Got the question.{chat_message}' from the user {user_tg_id}")



    # Looking for or creating contact

    contact = await get_or_create_chat_contact(contact_value=str(user_tg_id), contact_type=ContactType.TELEGRAM)

    user = await get_chat_user_by_id(contact.chat_user_id)

    question_message = await save_question_message_from_user(contact_id=contact.id, chat_message=chat_message)



    # Authorisation test

    if not user.organization_id:

        await message.answer(

            'Please allow me to respond to your requests.',

            reply_markup=UserKeyboard.get_auth_kb()

        )

        logger.info(f"User with id '{user.id}' and user_tg_id '{user_tg_id}'is not authorized to ask the bogus question.")

        return



    # Verification of consent to data processing (if necessary)

    if not user.is_agreed:

        # You can send the agreement keyboard here.

        pass



    # Sending a question to GPT agent

    try:

        bot_message = await get_gpt_answer(chat_message)

        logger.info(f"Got the answer. '{bot_message}' from the GPT agent for the user {user_tg_id}")



        if bot_message and len(bot_message) > 0:

            answer = bot_message

        else:

            answer = (

                "I'm sorry, the knowledge base is actively expanding, and it doesn't yet provide an answer to your question. "

                'We handed it over to the manager. Please wait for the answer.'

            )

            # Create a problem for the manager.

            message_to_save = (

                f"User's question: '{chat_message}'\n"

                f"The bot's answer is: '{answer}'\n"

                f"Time to create the question: '{datetime.now()}'"

            )

            await create_message_for_manager(message=message_to_save, chat_contact_id=contact.id)



    except Exception as e:

        logger.error(f"Error when addressing the GPT agent: {e}")

        answer = 'Sorry, there was an error in generating the answer.'



    # Keep the nerd's answer.

    await save_bot_response_to_user_question(

        message_id=question_message.id,

        answer=answer,

    )

    # Send a response to the user

    await message.answer(answer)





@default_handler_router.message(F.text)

async def request_in_chatgpt_cmd(message: types.Message):

    await process_request(message)





def register_default_handler_cmd(dp):

    dp.include_router(default_handler_router)

