import torch
import torch.nn as nn
from transformers import BertModel, AutoConfig, AutoTokenizer, pipeline
import pickle

TOKENIZER = 'D:/JKH_Diplom/data/ruber_riny2_tokenizer'

ENCODERS = 'D:/JKH_Diplom/data/encoders/tiny2_encoders.pkl'
WEIGHTS = 'D:/JKH_Diplom/data/weights/multitask_model_weights_tiny2.pth'



MODEL_NAME = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

emotion_classifier = pipeline("text-classification", model="cointegrated/rubert-tiny2-cedr-emotion-detection")

try:
    with open(ENCODERS, 'rb') as f:
        binarizers = pickle.load(f)
    print("Binarizers загружены!")
    print("Binarizers keys:", list(binarizers.keys()))
except FileNotFoundError:
    print("Файл binarizers не найден!")
    binarizers = {}


def safe_label(classes_obj, idx, prefix):
    try:
        classes = list(classes_obj)
        if 0 <= idx < len(classes):
            return classes[idx]
        return f"{prefix}_{idx}"
    except Exception as e:
        print(f"safe_label error for {prefix}: {e}")
        return f"{prefix}_{idx}"


class SimpleMultiTaskBERT(nn.Module):
    def __init__(self, config, num_categories, num_emotions, num_urgencies):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME, config=config)
        self.dropout = nn.Dropout(0.8)
        self.category_dropout = nn.Dropout(0.9)

        self.classifier_category = nn.Linear(config.hidden_size, num_categories)
        self.classifier_emotion = nn.Linear(config.hidden_size, num_emotions)
        self.classifier_urgency = nn.Linear(config.hidden_size, num_urgencies)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        category_output = self.category_dropout(pooled_output)
        emotion_output = self.dropout(pooled_output)
        urgency_output = self.dropout(pooled_output)

        category_logits = self.classifier_category(category_output)
        emotion_logits = self.classifier_emotion(emotion_output)
        urgency_logits = self.classifier_urgency(urgency_output)

        return {
            'category_logits': category_logits,
            'emotion_logits': emotion_logits,
            'urgency_logits': urgency_logits
        }


if not binarizers:
    num_categories = 11
    num_emotions = 6
    num_urgencies = 3
else:
    num_categories = len(binarizers.get('category', {}).get('classes', []))
    num_emotions = len(binarizers.get('emotion', {}).get('classes', []))
    num_urgencies = len(binarizers.get('urgency', {}).get('classes', []))

print("num_categories:", num_categories)
print("num_emotions:", num_emotions)
print("num_urgencies:", num_urgencies)

config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_categories = num_categories
config.num_emotions = num_emotions
config.num_urgencies = num_urgencies

model = SimpleMultiTaskBERT(
    config,
    num_categories=num_categories,
    num_emotions=num_emotions,
    num_urgencies=num_urgencies,
)

state_dict = torch.load(WEIGHTS, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
print("Модель загружена")


def predict_complaint(text):
    try:
        print("\nTEXT TO MODEL:", text)

        # 1. Запрос к старой модели (для Категории и Срочности)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        category_logits = outputs['category_logits']
        urgency_logits = outputs['urgency_logits']

        category_pred = torch.argmax(category_logits, dim=1).item()
        urgency_pred = torch.argmax(urgency_logits, dim=1).item()

        # 2. Запрос к НОВОЙ хорошей модели (только для Эмоций)
        new_emotion_output = emotion_classifier(text)[0]  # Извлечение из списка
        emotion_label = new_emotion_output['label']

        print("RAW PRED IDS (OLD MODEL):")
        print("category_pred:", category_pred)
        print("urgency_pred:", urgency_pred)

        print("CLASS LIST LENGTHS:")
        print("category:", len(binarizers.get('category', {}).get('classes', [])) if binarizers else 0)
        print("urgency:", len(binarizers.get('urgency', {}).get('classes', [])) if binarizers else 0)

        category_classes = binarizers.get('category', {}).get('classes', []) if binarizers else []
        urgency_classes = binarizers.get('urgency', {}).get('classes', []) if binarizers else []

        category_label = safe_label(category_classes, category_pred, "Категория")
        urgency_label = safe_label(urgency_classes, urgency_pred, "Срочность")

        # ==================== КОСТЫЛЬ ДЛЯ КОРРЕКЦИИ СРОЧНОСТИ И КАТЕГОРИИ ====================
        text_lower = text.lower()
        
        # Исправление неверной категории для отопления
        if any(word in text_lower for word in ['батарея', 'батареи', 'отопление', 'холодно', 'не греют']):
            category_label = 'Отопление'

        # 1. СВЕРХКРИТИЧЕСКИЕ СИТУАЦИИ (Пожар, Газ) — переопределяем и категорию, и срочность
        if any(word in text_lower for word in ['пожар', 'горит', 'дым', 'задымление', 'взрыв', 'газ', 'запах газа']):
            category_label = 'Экстренная ситуация (МЧС)'
            urgency_label = 'критическая'
            
        # 2. АВАРИЙНЫЕ СИТУАЦИИ ЖКХ (Лифт, Затопление) — завышаем срочность
        else:
            critical_keywords = [
                'застрял', 'лифт', 'затопило', 'потоп', 'льется вода', 
                'прорвало трубу', 'кипяток', 'бьет ток', 'задыхаюсь', 'сломался лифт'
            ]
            
            # Повышаем только если есть жесткие маркеры аварии ИЛИ сильный страх (паника в лифте)
            if (
                any(word in text_lower for word in critical_keywords) or
                category_label.strip().lower() in ['лифт'] or
                emotion_label in ['fear']
            ):
                if urgency_label in ['средняя', 'низкая', 'medium', 'low', 'surprise']:
                    urgency_label = 'высокая'
        # =====================================================================================

        print("FINAL LABELS:")
        print("category_label:", category_label)
        print("emotion_label (NEW MODEL):", emotion_label)
        print("urgency_label:", urgency_label)

        return {
            'category': category_label,
            'emotion': emotion_label,
            'urgency': urgency_label,
            'raw_category_id': category_pred,
            'raw_emotion_id': None,            
            'raw_urgency_id': urgency_pred
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'category': 'не определена',
            'emotion': 'нейтральная',
            'urgency': 'средняя',
            'error': str(e)
        }

def get_advice(prediction):
    """Генерация совета на основе предсказания с учетом экстренных ситуаций"""
    advice_templates = {
        'критическая': "🚨 НЕМЕДЛЕННО покиньте опасную зону и звоните в экстренные службы по телефону 112 или 101!",
        'critical': "🚨 НЕМЕДЛЕННО покиньте опасную зону и звоните в экстренные службы по телефону 112 или 101!",
        'высокая': "Рекомендуем срочно обратиться в аварийную службу!",
        'high': "Рекомендуем срочно обратиться в аварийную службу!",
        'средняя': "Обратитесь в управляющую компанию в рабочее время.",
        'medium': "Обратитесь в управляющую компанию в рабочее время.",
        'низкая': "Проблема будет решена в плановом порядке.",
        'low': "Проблема будет решена в плановом порядке."
    }

    # Если категория определена как ЧС / МЧС, жестко форсим критический совет
    category = prediction.get('category', '') if isinstance(prediction, dict) else ''
    if 'мчс' in category.lower() or 'экстренная' in category.lower():
        return advice_templates.get('критическая')

    urgency = prediction.get('urgency', 'средняя') if isinstance(prediction, dict) else 'средняя'
    return advice_templates.get(urgency, "Специалист свяжется с вами в ближайшее время.")