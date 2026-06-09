import os
import sqlite3
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv, find_dotenv
from vkbottle.bot import Bot, Message
from vkbottle import Keyboard, KeyboardButtonColor, Text, EMPTY_KEYBOARD

from model import predict_complaint
from analyzer_module import analyzer

load_dotenv(find_dotenv())

TOKEN = os.getenv("TOKEN")
DB_PATH = r"D:\JKH_Diplom\jkh.db"

bot = Bot(token=TOKEN)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            address TEXT,
            category TEXT,
            emotion TEXT,
            urgency TEXT,
            name TEXT,
            phone TEXT,
            status TEXT NOT NULL DEFAULT 'новая'
        )
    """)
    conn.commit()
    conn.close()


def db_fetchone(query: str, params: tuple = ()) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, params)
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def db_fetchall(query: str, params: tuple = ()) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_execute(query: str, params: tuple = ()) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    lastrowid = cur.lastrowid
    conn.close()
    return lastrowid


def extract_address_from_analysis(analysis_result: Dict[str, Any]) -> Optional[str]:
    try:
        addresses = (
            analysis_result.get("combined_analysis", {})
            .get("extracted_data", {})
            .get("addresses", [])
        )
        if addresses:
            return addresses[0]
    except Exception:
        pass
    return None


def build_recommendations_text(analysis_result: Dict[str, Any]) -> str:
    recommendations = (
        analysis_result.get("combined_analysis", {})
        .get("recommendations", {})
    )

    actions = recommendations.get("actions", [])
    contacts = recommendations.get("contacts", [])
    specialists = recommendations.get("recommended_specialists", [])
    deadlines = recommendations.get("deadlines", "не указаны")

    lines = [
        "Рекомендации:",
        f"Сроки решения: {deadlines}",
    ]

    if actions:
        lines.append("Что делать:")
        for i, action in enumerate(actions, 1):
            lines.append(f"{i}. {action}")

    if contacts:
        lines.append("Контакты:")
        for contact in contacts:
            lines.append(f"• {contact}")

    if specialists:
        lines.append("Специалисты:")
        for specialist in specialists:
            lines.append(f"• {specialist}")

    return "\n".join(lines)


async def send_status_notification(user_id: int, complaint_id: int, new_status: str):
    status_messages = {
        "новая": "принята в обработку",
        "в_работе": "переведена в работу",
        "решена": "отмечена как решённая",
    }

    human_status = status_messages.get(new_status, new_status)

    text = (
        f"✅ Обновление по вашей заявке №{complaint_id}\n\n"
        f"Статус обращения изменён: {human_status}.\n"
        f"Вы можете открыть раздел «Мои обращения» и посмотреть актуальную информацию."
    )

    try:
        await bot.api.messages.send(
            peer_id=user_id,
            message=text,
            random_id=complaint_id
        )
        return True
    except Exception as e:
        print(f"Ошибка отправки уведомления пользователю {user_id}: {e}")
        return False


def send_status_notification_sync(user_id: int, complaint_id: int, new_status: str):
    try:
        return asyncio.run(send_status_notification(user_id, complaint_id, new_status))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                send_status_notification(user_id, complaint_id, new_status)
            )
        finally:
            loop.close()


def main_menu_kb():
    kb = Keyboard(inline=False)
    kb.add(Text("Подать обращение"), color=KeyboardButtonColor.PRIMARY)
    kb.row()
    kb.add(Text("Мои обращения"), color=KeyboardButtonColor.SECONDARY)
    kb.add(Text("Экстренные контакты"), color=KeyboardButtonColor.SECONDARY)
    return kb


def confirm_kb():
    kb = Keyboard(inline=False)
    kb.add(Text("Подтвердить"), color=KeyboardButtonColor.POSITIVE)
    kb.add(Text("Изменить"), color=KeyboardButtonColor.NEGATIVE)
    return kb


def skip_kb():
    kb = Keyboard(inline=False)
    kb.add(Text("Пропустить"), color=KeyboardButtonColor.SECONDARY)
    return kb


def emergency_kb():
    kb = Keyboard(inline=False)
    kb.add(Text("112"), color=KeyboardButtonColor.PRIMARY)
    kb.add(Text("104"), color=KeyboardButtonColor.PRIMARY)
    kb.row()
    kb.add(Text("Назад"), color=KeyboardButtonColor.SECONDARY)
    return kb


@dataclass
class UserState:
    step: str = "idle"
    text: str = ""
    address: Optional[str] = None
    category: Optional[str] = None
    emotion: Optional[str] = None
    urgency: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    prediction: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)


user_states: Dict[int, UserState] = {}


def get_state(user_id: int) -> UserState:
    if user_id not in user_states:
        user_states[user_id] = UserState()
    return user_states[user_id]


def reset_state(state: UserState):
    state.step = "idle"
    state.text = ""
    state.address = None
    state.category = None
    state.emotion = None
    state.urgency = None
    state.name = None
    state.phone = None
    state.prediction = {}
    state.analysis = {}


def save_complaint(user_id: int, state: UserState) -> int:
    return db_execute(
        """
        INSERT INTO complaints (
            timestamp, user_id, text, address, category, emotion,
            urgency, name, phone, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_id,
            state.text,
            state.address,
            state.category,
            state.emotion,
            state.urgency,
            state.name,
            state.phone,
            "новая",
        )
    )


def get_user_complaints(user_id: int) -> List[dict]:
    return db_fetchall(
        """
        SELECT id, timestamp, text, category, urgency, status
        FROM complaints
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,)
    )


init_db()


@bot.on.message(text=["/start", "Старт"])
async def start(message: Message):
    state = get_state(message.from_id)
    reset_state(state)
    await message.answer(
        "Привет! Я бот для обработки жалоб ЖКХ.\n"
        "Нажмите «Подать обращение», чтобы начать.",
        keyboard=main_menu_kb().get_json()
    )


@bot.on.message(text="Подать обращение")
async def start_complaint(message: Message):
    state = get_state(message.from_id)
    reset_state(state)
    state.step = "await_text"
    await message.answer(
        "Опишите проблему одним сообщением.\n"
        "Пример: «В подъезде не горит свет, очень страшно»",
        keyboard=EMPTY_KEYBOARD
    )


@bot.on.message(text="Мои обращения")
async def my_complaints(message: Message):
    rows = get_user_complaints(message.from_id)
    if not rows:
        await message.answer(
            "У вас пока нет обращений.",
            keyboard=main_menu_kb().get_json()
        )
        return

    lines = ["Ваши обращения:"]
    for row in rows[:10]:
        lines.append(
            f"#{row['id']} | {row['timestamp']} | "
            f"{row['category'] or 'не определена'} | "
            f"{row['urgency'] or 'средняя'} | статус: {row['status']}"
        )

    await message.answer("\n".join(lines), keyboard=main_menu_kb().get_json())


@bot.on.message(text="Экстренные контакты")
async def emergency_contacts(message: Message):
    await message.answer(
        "Экстренные контакты:\n"
        "112 — единый номер экстренных служб\n"
        "104 — аварийная газовая служба\n"
        "101 — пожарная служба",
        keyboard=emergency_kb().get_json()
    )


@bot.on.message(text="112")
async def contact_112(message: Message):
    await message.answer(
        "112 — единый номер экстренных служб.",
        keyboard=main_menu_kb().get_json()
    )


@bot.on.message(text="104")
async def contact_104(message: Message):
    await message.answer(
        "104 — аварийная газовая служба.",
        keyboard=main_menu_kb().get_json()
    )


@bot.on.message(text="Назад")
async def back_to_menu(message: Message):
    await message.answer(
        "Главное меню.",
        keyboard=main_menu_kb().get_json()
    )


@bot.on.message()
async def handle_all(message: Message):
    state = get_state(message.from_id)
    text = (message.text or "").strip()

    if not text:
        return

    if state.step == "await_text":
        state.text = text

        try:
            prediction = predict_complaint(text)
        except Exception as e:
            prediction = {
                "category": "не определена",
                "emotion": "нейтральная",
                "urgency": "средняя",
                "error": str(e),
            }

        try:
            analysis_result = analyzer.analyze_complaint(text)
        except Exception as e:
            analysis_result = {
                "error": str(e),
                "combined_analysis": {
                    "extracted_data": {"addresses": []},
                    "recommendations": {}
                }
            }

        state.prediction = prediction
        state.analysis = analysis_result
        state.category = prediction.get("category", "не определена")
        state.emotion = prediction.get("emotion", "нейтральная")
        state.urgency = prediction.get("urgency", "средняя")
        state.address = extract_address_from_analysis(analysis_result)

        if not state.address:
            state.step = "await_address"
            await message.answer(
                "Адрес не найден. Пожалуйста, отправьте адрес вручную.",
                keyboard=EMPTY_KEYBOARD
            )
            return

        state.step = "await_name"
        await message.answer(
            "Введите имя или нажмите «Пропустить».",
            keyboard=skip_kb().get_json()
        )
        return

    if state.step == "await_address":
        if text.lower() != "пропустить":
            state.address = text

        state.step = "await_name"
        await message.answer(
            "Введите имя или нажмите «Пропустить».",
            keyboard=skip_kb().get_json()
        )
        return

    if state.step == "await_name":
        if text.lower() != "пропустить":
            state.name = text

        state.step = "await_phone"
        await message.answer(
            "Введите телефон или нажмите «Пропустить».",
            keyboard=skip_kb().get_json()
        )
        return

    if state.step == "await_phone":
        if text.lower() != "пропустить":
            state.phone = text

        state.step = "confirm"
        summary = (
            "Проверьте данные заявки:\n\n"
            f"Текст: {state.text}\n"
            f"Адрес: {state.address or 'не указан'}\n"
            f"Категория: {state.category or 'не определена'}\n"
            f"Срочность: {state.urgency or 'средняя'}\n"
            f"Имя: {state.name or 'не указано'}\n"
            f"Телефон: {state.phone or 'не указан'}"
        )
        await message.answer(summary, keyboard=confirm_kb().get_json())
        return

    if state.step == "confirm":
        if text == "Подтвердить":
            complaint_id = save_complaint(message.from_id, state)
            state.step = "idle"

            rec_text = build_recommendations_text(state.analysis or {})
            await message.answer(
                f"Заявка #{complaint_id} создана со статусом «новая».\n\n{rec_text}",
                keyboard=main_menu_kb().get_json()
            )
            return

        if text == "Изменить":
            reset_state(state)
            state.step = "await_text"
            await message.answer(
                "Хорошо. Напишите жалобу заново.",
                keyboard=EMPTY_KEYBOARD
            )
            return

    await message.answer(
        "Выберите действие в меню.",
        keyboard=main_menu_kb().get_json()
    )


if __name__ == "__main__":
    bot.run_forever()