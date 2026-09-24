"""
ДЗ урок 10: Fine-tuning модели
Задача: классификация тональности отзывов (позитив/негатив)
Модель: distilbert-base-uncased
Датасет: IMDb (урезанная выборка для быстрого обучения на CPU)
"""

import re
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# ==========================================================
# ШАГ 1. Загрузка датасета
# ==========================================================
print("Шаг 1: Загружаю датасет IMDb...")

dataset = load_dataset("stanfordnlp/imdb")

# Берём небольшую выборку, чтобы обучение не занимало часы на CPU
small_train = dataset["train"].shuffle(seed=42).select(range(500))
small_test = dataset["test"].shuffle(seed=42).select(range(200))

print(f"Обучающая выборка: {len(small_train)} примеров")
print(f"Тестовая выборка: {len(small_test)} примеров")
print("Пример записи до очистки:")
print(small_train[0]["text"][:200], "...")

# ==========================================================
# ШАГ 2. Подготовка данных (очистка + токенизация)
# ==========================================================
print("\nШаг 2: Очищаю и токенизирую данные...")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def clean_text(example):
    """Убираем HTML-теги и лишние пробелы из текста отзыва."""
    text = example["text"]
    text = re.sub(r"<.*?>", " ", text)  # HTML-теги (в IMDb много <br />)
    text = re.sub(r"\s+", " ", text).strip()  # лишние пробелы
    example["text"] = text
    return example


small_train = small_train.map(clean_text)
small_test = small_test.map(clean_text)

print("Пример записи после очистки:")
print(small_train[0]["text"][:200], "...")


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)


train_tokenized = small_train.map(tokenize, batched=True)
test_tokenized = small_test.map(tokenize, batched=True)

# ==========================================================
# ШАГ 3. Fine-tuning модели
# ==========================================================
print("\nШаг 3: Запускаю fine-tuning...")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_strategy="no",  # не сохраняем чекпоинты на каждой эпохе, чтобы не тратить место
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==========================================================
# ШАГ 4. Оценка качества модели
# ==========================================================
print("\nШаг 4: Оцениваю качество модели на тестовой выборке...")

results = trainer.evaluate()
print("Результаты на тестовой выборке:")
print(results)

# Проверка на собственных примерах
test_texts = [
    "This movie was absolutely wonderful, great acting and story!",
    "Waste of time, terrible plot and bad acting.",
    "It was okay, not great but not terrible either.",
]

inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

print("\nПроверка на собственных примерах:")
for text, pred in zip(test_texts, predictions):
    label = "positive" if pred.item() == 1 else "negative"
    print(f"'{text}' -> {label}")

print("\nГотово!")