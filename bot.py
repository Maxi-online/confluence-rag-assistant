#!/usr/bin/env python3
import os
import logging
import json
import traceback
import asyncio
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from ai.mas.main import process_message, ChatRequest, ChatResponse, UserProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
DB_PATH = "user_profiles.db"

# Загружаем конфигурацию из переменных окружения
CONFIG = {
    "USER_ID": "user_002",
    "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
    "CONFLUENCE_API_TOKEN": os.environ["CONFLUENCE_API_TOKEN"],
    "CONFLUENCE_PAGE_ID": 195635434,
    "WEAVIATE_URL": os.environ["WEAVIATE_URL"],
    "WEAVIATE_API_KEY": os.environ["WEAVIATE_API_KEY"],
    "EXPECTED_DIM": 1536,
    "MAX_CHUNK_SIZE": 1536,
    "MIN_SCORE": 0.5,
    "HYBRID_ALPHA": 0.5
}

def init_db():
    """Инициализация базы данных SQLite"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Создаем таблицу для хранения профилей пользователей
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_profile_id INTEGER PRIMARY KEY,
            language TEXT,
            user_name TEXT,
            program_area TEXT,
            education_goal TEXT,
            education_objections TEXT,
            age INTEGER,
            heard_about TEXT,
            after_study_goal TEXT,
            education_format TEXT,
            work_experience TEXT,
            additional_interests TEXT,
            selected_university TEXT,
            is_modified BOOLEAN DEFAULT FALSE,
            is_authorized BOOLEAN DEFAULT FALSE,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()

def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Получение профиля пользователя из БД"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user_profiles WHERE user_profile_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
    return None

def update_user_profile(profile_data: Dict[str, Any]) -> None:
    """Обновление профиля пользователя в БД"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Логируем данные до обновления
        logging.info(f"Обновление профиля в БД, данные: {profile_data}")
        
        try:
            # Получаем список полей модели UserProfile
            user_profile_fields = UserProfile.model_fields.keys()
            logging.info(f"Доступные поля модели UserProfile: {list(user_profile_fields)}")
            
            # Получаем список полей и значений для обновления
            fields = []
            values = []
            for key, value in profile_data.items():
                # Проверяем, является ли ключ полем модели UserProfile
                if key != 'user_profile_id' and key in user_profile_fields:
                    fields.append(f"{key} = ?")
                    values.append(value)
                    logging.info(f"Добавлено поле для обновления: {key} = {value}")
            
            # Добавляем время обновления и сбрасываем is_modified
            fields.append("last_updated = ?")
            fields.append("is_modified = ?")
            values.extend([datetime.now(), False])
            
            # Добавляем user_profile_id в конец списка значений
            values.append(profile_data['user_profile_id'])
            
            # Проверяем наличие полей для обновления
            if not fields:
                logging.warning(f"Нет полей для обновления профиля {profile_data['user_profile_id']}")
                return
            
            # Формируем и выполняем запрос
            query = f"""
            UPDATE user_profiles 
            SET {', '.join(fields)}
            WHERE user_profile_id = ?
            """
            logging.info(f"SQL UPDATE запрос: {query}")
            logging.info(f"SQL параметры: {values}")
            
            cursor.execute(query, values)
            
            # Если записи нет, создаем новую
            if cursor.rowcount == 0:
                logging.info(f"Запись не найдена, создаем новую для user_profile_id={profile_data['user_profile_id']}")
                
                # Собираем все доступные поля для вставки
                insert_fields = ['user_profile_id']
                insert_values = [profile_data['user_profile_id']]
                
                for key, value in profile_data.items():
                    if key != 'user_profile_id' and key in user_profile_fields:
                        insert_fields.append(key)
                        insert_values.append(value)
                
                # Добавляем метку времени и флаг модификации
                insert_fields.extend(['last_updated', 'is_modified'])
                insert_values.extend([datetime.now(), False])
                
                placeholders = ['?'] * len(insert_fields)
                
                query = f"""
                INSERT INTO user_profiles ({', '.join(insert_fields)})
                VALUES ({', '.join(placeholders)})
                """
                logging.info(f"SQL INSERT запрос: {query}")
                logging.info(f"SQL параметры: {insert_values}")
                
                cursor.execute(query, insert_values)
            
            # Проверяем изменения
            conn.commit()
            logging.info(f"SQL запрос выполнен успешно, изменено строк: {cursor.rowcount}")
            
            # Проверяем результат
            cursor.execute("SELECT language FROM user_profiles WHERE user_profile_id = ?", 
                          (profile_data['user_profile_id'],))
            result = cursor.fetchone()
            logging.info(f"Проверка после обновления - language: {result[0] if result else None}")
            
        except Exception as e:
            logging.error(f"Ошибка при обновлении профиля в БД: {str(e)}")
            logging.error(traceback.format_exc())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    logging.info(f"\n{'='*80}\nНОВОЕ СООБЩЕНИЕ\nПользователь {user_id}: {user_question}\n{'='*80}")

    try:
        # Получаем профиль пользователя из БД
        profile_data = get_user_profile(user_id)
        
        # Выводим состояние профиля ДО обработки
        logging.info(f"\n{'='*80}\nПРОФИЛЬ ДО ОБРАБОТКИ:")
        if profile_data:
            for key, value in profile_data.items():
                if key not in ['last_updated', 'is_modified']:
                    logging.info(f"{key:.<30} {value}")
        else:
            logging.info("Профиль не существует")
        logging.info('='*80)
        
        # Если профиль не существует, создаем базовый
        if not profile_data:
            profile_data = {
                "user_profile_id": user_id,
                "language": update.effective_user.language_code,
                "user_name": update.effective_user.first_name,
                "is_authorized": False,
                "is_modified": False,
                "program_area": None,
                "education_goal": None,
                "education_objections": None,
                "age": None,
                "heard_about": None,
                "after_study_goal": None,
                "education_format": None,
                "work_experience": None,
                "additional_interests": None,
                "selected_university": None
            }
            update_user_profile(profile_data)
            logging.info("Создан новый профиль пользователя")
        else:
            # Обновляем только отсутствующие поля
            profile_data.setdefault("language", update.effective_user.language_code)
            profile_data.setdefault("user_name", update.effective_user.first_name)
            profile_data.setdefault("is_authorized", False)
            profile_data.setdefault("is_modified", False)
            update_user_profile(profile_data)

        # Создаем объект UserProfile
        user_profile = UserProfile(
            user_profile_id=user_id,
            language=profile_data.get('language'),
            user_name=profile_data.get('user_name'),
            program_area=profile_data.get('program_area'),
            education_goal=profile_data.get('education_goal'),
            education_objections=profile_data.get('education_objections'),
            age=profile_data.get('age'),
            heard_about=profile_data.get('heard_about'),
            after_study_goal=profile_data.get('after_study_goal'),
            education_format=profile_data.get('education_format'),
            work_experience=profile_data.get('work_experience'),
            additional_interests=profile_data.get('additional_interests'),
            selected_university=profile_data.get('selected_university'),
            is_modified=False
        )

        logging.info("\nСОЗДАННЫЙ ОБЪЕКТ ПРОФИЛЯ:")
        logging.info(f"user_profile_id: {user_profile.user_profile_id}")
        logging.info(f"language: {user_profile.language}")
        logging.info(f"is_modified: {user_profile.is_modified}")

        # Создаем запрос с конфигурацией
        request = ChatRequest(
            message=user_question,
            user_profile_id=str(user_id),
            user_profile=user_profile,
            is_authorized=profile_data.get('is_authorized', False),
            **CONFIG
        )
        
        logging.info("\nОТПРАВКА ЗАПРОСА:")
        logging.info(f"message: {request.message}")
        logging.info(f"user_profile_id: {request.user_profile_id}")
        logging.info(f"user_profile.language: {request.user_profile.language}")
        logging.info(f"user_profile.is_modified: {request.user_profile.is_modified}")
        
        # Получаем ответ от process_message
        response: ChatResponse = await process_message(request)
        
        logging.info("\nПОЛУЧЕН ОТВЕТ:")
        logging.info(f"answer: {response.answer}")
        logging.info(f"user_profile_id: {response.user_profile_id}")
        logging.info(f"user_profile.language: {response.user_profile.language if response.user_profile else None}")
        logging.info(f"user_profile.is_modified: {response.user_profile.is_modified if response.user_profile else None}")
        
        # Проверяем, что response является ChatResponse
        if not isinstance(response, ChatResponse):
            raise ValueError(f"Expected ChatResponse, got {type(response)}")
        
        # Если профиль был модифицирован, обновляем его в БД
        if response.user_profile and response.user_profile.is_modified:
            logging.info("\nПРОФИЛЬ БЫЛ МОДИФИЦИРОВАН:")
            
            # Преобразуем в словарь для обновления БД и логирования
            profile_dict = response.user_profile.model_dump()
            profile_dict['is_authorized'] = response.is_authorized
            
            # Логируем ИЗМЕНЕНИЯ в профиле
            logging.info(f"\n{'='*80}\nИЗМЕНЕНИЯ В ПРОФИЛЕ:")
            changes_found = False
            for key, new_value in profile_dict.items():
                old_value = profile_data.get(key)
                if new_value != old_value and key not in ['is_modified', 'last_updated']:
                    changes_found = True
                    logging.info(f"{key:.<30} БЫЛО:  {old_value}")
                    logging.info(f"{' ':.<30} СТАЛО: {new_value}")
            
            if not changes_found:
                logging.info("Нет измененных полей")
            logging.info('='*80)
            
            # Обновляем профиль в БД
            logging.info("\nОБНОВЛЕНИЕ ПРОФИЛЯ В БД:")
            logging.info(f"profile_dict: {profile_dict}")
            update_user_profile(profile_dict)
            
            # Получаем обновленный профиль для логирования
            updated_profile = get_user_profile(user_id)
            
            # Обновляем локальный объект профиля
            if updated_profile:
                user_profile = UserProfile(
                    user_profile_id=user_id,
                    language=updated_profile.get('language'),
                    user_name=updated_profile.get('user_name'),
                    program_area=updated_profile.get('program_area'),
                    education_goal=updated_profile.get('education_goal'),
                    education_objections=updated_profile.get('education_objections'),
                    age=updated_profile.get('age'),
                    heard_about=updated_profile.get('heard_about'),
                    after_study_goal=updated_profile.get('after_study_goal'),
                    education_format=updated_profile.get('education_format'),
                    work_experience=updated_profile.get('work_experience'),
                    additional_interests=updated_profile.get('additional_interests'),
                    selected_university=updated_profile.get('selected_university'),
                    is_modified=False
                )
            
            # Выводим состояние профиля ПОСЛЕ обработки
            logging.info(f"\n{'='*80}\nПРОФИЛЬ ПОСЛЕ ОБРАБОТКИ:")
            if updated_profile:
                for key, value in updated_profile.items():
                    if key not in ['last_updated', 'is_modified']:
                        logging.info(f"{key:.<30} {value}")
            logging.info('='*80)
        else:
            logging.info("\nПРОФИЛЬ НЕ БЫЛ МОДИФИЦИРОВАН")

        # Отправляем ответ пользователю
        answer = response.answer
        max_length = 4096  # Максимальная длина сообщения в Telegram
        
        if len(answer) > max_length:
            parts = [answer[i:i + max_length] for i in range(0, len(answer), max_length)]
            for part in parts:
                await update.message.reply_text(part)
                await asyncio.sleep(0.5)  # Небольшая задержка между сообщениями
        else:
            await update.message.reply_text(answer)
            
        logging.info(f"\n{'='*80}\nОТВЕТ ОТПРАВЛЕН\nБот: {answer}\n{'='*80}")

    except Exception as e:
        logging.error(f"Error processing message: {e}\n{traceback.format_exc()}")
        await update.message.reply_text(
            "Извините, произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже."
        )

if __name__ == '__main__':
    # Инициализируем базу данных
    init_db()
    
    # Настраиваем расширенное логирование
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("bot.log")
        ]
    )
    
    try:
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        logging.info("Bot started successfully")
        application.run_polling()
    except Exception as e:
        logging.critical(f"Critical error starting bot: {e}\n{traceback.format_exc()}")
