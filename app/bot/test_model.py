import torch
import torch.nn as nn
from transformers import BertModel, AutoConfig, AutoTokenizer
import pickle
import pandas as pd
from sklearn import metrics
import os


# === ПУТИ (ПОДСТАВЬ СВОИ) ===
TOKENIZER_PATH = 'D:/JKH_Diplom/data/ruber_riny2_tokenizer'
ENCODERS_PATH = 'D:/JKH_Diplom/data/encoders/tiny2_encoders.pkl'
WEIGHTS_PATH = 'D:/JKH_Diplom/data/weights/multitask_model_weights_tiny2.pth'
MODEL_NAME = "cointegrated/rubert-tiny2"
TEST_CSV = 'D:/JKH_Diplom/data/test_dataset.csv'  # колонки: text,category_true,emotion_true,urgency_true


# === ТОКЕНИЗАТОР, BINARIZERS, МОДЕЛЬ ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

try:
    with open(ENCODERS_PATH, 'rb') as f:
        binarizers = pickle.load(f)
    print("Binarizers загружены!")
except FileNotFoundError:
    print("Binarizers не найдены, используем числа по умолчанию.")
    binarizers = {}


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


def safe_label(classes_obj, idx, prefix):
    try:
        classes = list(classes_obj)
        if 0 <= idx < len(classes):
            return classes[idx]
        return f"{prefix}_{idx}"
    except Exception as e:
        print(f"safe_label error for {prefix}: {e}")
        return f"{prefix}_{idx}"


# параметры классов
if not binarizers:
    num_categories = 11
    num_emotions = 6
    num_urgencies = 5
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

state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
print("Модель загружена и готова к тестированию.")


# === ЧИСТО ТВОЯ МОДЕЛЬ, БЕЗ ВНЕШНЕГО ЭМОЦИОННОГО КЛАССИФИКАТОРА ===
def predict_complaint_lite(text):
    try:
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
        emotion_logits  = outputs['emotion_logits']
        urgency_logits  = outputs['urgency_logits']

        category_pred = torch.argmax(category_logits, dim=1).item()
        emotion_pred  = torch.argmax(emotion_logits,  dim=1).item()
        urgency_pred  = torch.argmax(urgency_logits,  dim=1).item()

        # декодируем по binarizers
        category_classes = binarizers.get('category', {}).get('classes', []) if binarizers else []
        emotion_classes  = binarizers.get('emotion',  {}).get('classes', []) if binarizers else []
        urgency_classes  = binarizers.get('urgency',  {}).get('classes', []) if binarizers else []

        category_label = safe_label(category_classes, category_pred, "Категория")
        emotion_label  = safe_label(emotion_classes,  emotion_pred,  "Эмоция")
        urgency_label  = safe_label(urgency_classes,  urgency_pred,  "Срочность")

        # если хочешь оставить костыль по тексту — можно оставить,
        # но будь аккуратен с разметкой в тесте
        text_lower = text.lower()
        if any(word in text_lower for word in ['батарея', 'батареи', 'отопление', 'холодно', 'не греют']):
            category_label = 'Отопление'
        if any(word in text_lower for word in ['пожар', 'горит', 'дым', 'задымление', 'взрыв', 'газ', 'запах газа']):
            category_label = 'Экстренная ситуация (МЧС)'
            urgency_label = 'критическая'

        return {
            'category': category_label,
            'emotion': emotion_label,
            'urgency': urgency_label
        }

    except Exception as e:
        return {
            'category': 'не определена',
            'emotion': 'нейтральная',
            'urgency': 'средняя',
        }


# === ФУНКЦИЯ ОЦЕНКИ МЕТРИК ===
def eval_and_print(y_true, y_pred, task_name, save_path=None):
    labels = sorted(list(set(y_true) | set(y_pred)))
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"\n=== {task_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report (precision, recall, F1):")
    print(metrics.classification_report(y_true, y_pred, labels=labels, zero_division=0))

    if save_path:
        report = metrics.classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).T
        df_report.to_csv(save_path, float_format='%.4f')
        print(f"Отчёт сохранён в: {save_path}")


# === ЗАГРУЗКА ТЕСТОВОГО CSV ===
df = pd.read_csv(TEST_CSV)
print(f"Тестовый датасет: {len(df)} записей.")

y_true_cat = df['category'].astype(str).tolist()
y_true_em  = df['emotion'].astype(str).tolist()
y_true_urg = df['urgency'].astype(str).tolist()

y_pred_cat = []
y_pred_em  = []
y_pred_urg = []

for i, row in df.iterrows():
    text = str(row['text'])

    pred = predict_complaint_lite(text)

    y_pred_cat.append(pred['category'])
    y_pred_em.append(pred['emotion'])
    y_pred_urg.append(pred['urgency'])

    if i % 100 == 0:
        print(f"Обработано {i}/{len(df)} записей")

# === ВЫЧИСЛЕНИЕ МЕТРИК ПО ЗАДАЧАМ ===
print("\nОЦЕНКА МЕТРИК:\n")

eval_and_print(
    y_true_cat, y_pred_cat,
    task_name="CATEGORY",
    save_path=os.path.splitext(TEST_CSV)[0] + "_category_metrics.csv"
)

eval_and_print(
    y_true_em, y_pred_em,
    task_name="EMOTION",
    save_path=os.path.splitext(TEST_CSV)[0] + "_emotion_metrics.csv"
)

eval_and_print(
    y_true_urg, y_pred_urg,
    task_name="URGENCY",
    save_path=os.path.splitext(TEST_CSV)[0] + "_urgency_metrics.csv"
)

print("Confusion matrix (emotion):")
print(metrics.confusion_matrix(y_true_em, y_pred_em))