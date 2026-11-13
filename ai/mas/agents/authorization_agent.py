#!/usr/bin/env python3
"""
Authorization Agent.
Временно отключена фактическая логика регистрации/авторизации (код недоступен).
Агент предлагает пользователю зарегистрироваться, если он удовлетворён ответами на вопросы,
и при согласии возвращает сигнал "Transfer to Lead Qualification Agent".
"""

import logging
from typing import Optional, Any
from pydantic import PrivateAttr

from openai import AsyncOpenAI
from ai.mas.utils.db import store_user_data_in_db
from ai.mas.agents.base_agent import BaseAgent, Agent
from ai.mas.main import ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthorizationAgent(BaseAgent):
    _gpt_client: AsyncOpenAI = PrivateAttr()

    def __init__(
        self, 
        user_profile_id: str, 
        openai_api_key: str, 
        openai_base_url: str = "https://api.openai.com/v1/",
        request: Any = None,
        **kwargs
    ):
        super().__init__(
            name="Authorization Agent",
            model="gpt-4o-mini",
            instructions=(
                """You are the AUTHORIZATION AGENT. Your task is to verify if the user meets the necessary requirements to proceed with registration or access.

            Instructions:
            1. Evaluate the provided credentials or details from the user using your internal Chain‑of‑Thought reasoning.
            2. Check the following:
            - If login/password or token is provided, validate them.
            - If access is role‑based, confirm that the user possesses the required role.
            - Ensure that any other relevant criteria (e.g., age, eligibility) are met.
            3. If authorization is successful, respond with a concise message such as "Authorization successful" or "Access granted."
            4. If the verification fails, output a clear message: "Authorization failed: insufficient privileges" (do not reveal specific internal details).
            5. Do not ask for any real credentials; this is a guided authorization flow.
            6. Output only the final message without internal reasoning.

            Output: A clear confirmation or rejection of authorization.
            """
            ),
            tools=[self.transfer_to_lead_qualification]
        )
        self.user_profile_id = user_profile_id
        # Присваиваем приватный атрибут после вызова super().__init__
        self._gpt_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)

    async def transfer_to_lead_qualification(self) -> Agent:
        """
        Transfer control to the Lead Qualification Agent when authorization is successful.
        """
        from ai.mas.utils.handoffs import transfer_to_lead_qualification_agent
        return transfer_to_lead_qualification_agent(
            user_profile_id=self.user_profile_id,
            openai_api_key=self._gpt_client.api_key,
            knowledge_client=None
        )

    async def handle_message(self, message: str, context: Optional[list] = None) -> ChatResponse:
        """
        Обрабатывает входящее сообщение:
        - Отправляет запрос к GPT с рекомендацией о регистрации.
        - Если пользователь соглашается (например, "Да, хочу зарегистрироваться"),
          возвращает сигнал "Transfer to Lead Qualification Agent".
        - Иначе возвращает ответ GPT.
        """
        context = context or []
        messages_payload = (
            [{"role": "system", "content": self.instructions}] +
            context +
            [{"role": "user", "content": message}]
        )
        
        response = await self._gpt_client.chat.completions.create(
            model=self.model,
            messages=messages_payload
        )
        reply = response.choices[0].message.content.strip()
        logger.info(f"Authorization Agent для {self.user_profile_id} получил ответ: {reply}")

        store_user_data_in_db(self.user_profile_id, {"auth": reply})
        logger.info(f"Authorization Agent для {self.user_profile_id} завершил обработку (без фактической регистрации).")
        
        # Проверяем содержимое ответа GPT на наличие сигналов о успешной авторизации/регистрации
        lower_reply = reply.lower()
        if "да, хочу" in lower_reply or "я согласен" in lower_reply or "registration successful" in lower_reply:
            logger.info(f"Обнаружен сигнал о согласии на регистрацию, выполняем handoff")
            return ChatResponse(
                answer="[Handoff] lead_qualification Отлично! Теперь давайте соберем дополнительную информацию для подбора оптимальной программы.",
                user_profile_id=self.user_profile_id,
                is_authorized=True if hasattr(self, 'request') and self.request else False,
                user_profile=self.request.user_profile if hasattr(self, 'request') else None
            )
        
        return ChatResponse(
            answer=reply,
            user_profile_id=self.user_profile_id,
            is_authorized=False if hasattr(self, 'request') and self.request else False,
            user_profile=self.request.user_profile if hasattr(self, 'request') else None
        )
