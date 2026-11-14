from aiogram import Router
from aiogram import types

from ai.multi_agent_system.agents.sql_agent import asql_agent, sql_agent

sql_query_cmd_router = Router()


@sql_query_cmd_router.message(lambda message: message.text[:4] == '/sq:')
async def sql_query_cmd(message: types.Message):
    query = message.text.split(':')[1].strip()

    await message.answer(f'Your request for database: {query}')

    inputs = {
        "user_tg_id": message.from_user.id,
        "student_request": query,
        "response": "",
    }

    bot_message = await asql_agent(inputs)
    response = bot_message['response']

    await message.answer(response.content)


def register_sql_query_cmd(dp):
    dp.include_router(sql_query_cmd_router)
