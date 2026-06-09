from model import get_advice

def safe_get(lst, idx, default="не указано"):
    try:
        return lst[idx]
    except (IndexError, TypeError):
        return default


def format_specialists(rag_data):
    """Форматирует блок специалистов"""
    specialists = rag_data.get('specialists', [])
    specialists_text = ""

    for spec_info in specialists[:3]:
        spec = spec_info.get('specialist', {})
        specialists_text += f"""
<b>{spec.get('name', 'Неизвестно')}</b>
Должность: {spec.get('position', 'не указана')}
Телефон: <code>{spec.get('phone', 'не указан')}</code>
Рейтинг: {spec.get('rating', 'не указан')}/5 | Район: {spec.get('district', 'не указан')}
───────────────────
"""

    return f"👨‍💼 **РЕКОМЕНДУЕМЫЕ СПЕЦИАЛИСТЫ:**\n{specialists_text or 'Специалисты не найдены'}"


def generate_beautiful_response(prediction, analysis_result):
    """Генерирует красивый и лаконичный ответ с рекомендациями и специалистами"""

    combined_data = analysis_result.get('combined_analysis', {})
    rag_data = analysis_result.get('rag_result', {})

    extracted_data = combined_data.get('extracted_data', {})
    addresses = extracted_data.get('addresses', [])
    formatted_address = "не указан"

    if addresses:
        addr = str(addresses[0])
        if 'улица' in addr:
            street = addr.split("value='")[1].split("'")[0] if "value='" in addr else "неизвестно"
            formatted_address = f"ул. {street}"
        elif 'дом' in addr:
            house = addr.split("value='")[1].split("'")[0] if "value='" in addr else "неизвестно"
            formatted_address = f"д. {house}"
        elif 'квартира' in addr:
            apt = addr.split("value='")[1].split("'")[0] if "value='" in addr else "неизвестно"
            formatted_address = f"кв. {apt}"
        else:
            formatted_address = addr

    category = rag_data.get('category', 'other').replace('_', ' ').title()

    recommendations = rag_data.get('recommendations', {})
    main_actions = recommendations.get('actions', [])[:3]
    contacts = recommendations.get('contacts', [])
    deadlines = recommendations.get('deadlines', 'не указаны')

    contact_1 = safe_get(contacts, 0, "не указан")
    contact_2 = safe_get(contacts, 1, "не указан")

    specialists_text = ""
    for i, specialist_info in enumerate(rag_data.get('specialists', [])[:3]):
        spec = specialist_info.get('specialist', {})
        specialists_text += f"👤 {spec.get('name', 'Неизвестно')}\n"
        specialists_text += f"   📞 {spec.get('phone', 'не указан')}\n"
        specialists_text += f"   💼 {spec.get('position', 'не указана')}\n"
        if i < 2:
            specialists_text += "   ───────────────────\n"

    urgency = str(prediction.get('urgency', 'средняя')).upper() if isinstance(prediction, dict) else "СРЕДНЯЯ"
    emotion = str(prediction.get('emotion', 'нейтральная')) if isinstance(prediction, dict) else "нейтральная"

    response = f"""
🎯 **АНАЛИЗ ВАШЕЙ ЗАЯВКИ**


🚀 **ЧТО ДЕЛАТЬ ПРЯМО СЕЙЧАС:**
"""

    for i, action in enumerate(main_actions, 1):
        response += f"   {i}. {action}\n"

    response += f"""
📞 **КОНТАКТЫ ДЛЯ СВЯЗИ:**
   • {contact_1} - основной контакт
   • {contact_2} - дополнительный


👨‍💼 **РЕКОМЕНДУЕМЫЕ СПЕЦИАЛИСТЫ:**
{specialists_text or 'Специалисты не найдены'}


⏱️ **Сроки решения:** {deadlines}


💡 **ВАЖНО:** {get_advice(prediction)}


💡 **Проверяем данные:**
📍 **Адрес:** {formatted_address}
🏷️ **Категория проблемы:** {category}
⚡ **Уровень срочности:** {urgency}
😊 **Эмоциональный фон:** {emotion}




"""

    return response