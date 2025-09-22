# Восстановление пробелов в тексте

## Задача

Цель проекта — восстановление пропусков пробелов в пользовательских строках (поисковые запросы, описания, заголовки).
Пример:

```
"куплюайфон14про" → "куплю айфон 14 про"
```

Для решения задачи используется дообучение предобученной модели `ai-forever/ruT5-large`.

---

## Обучение модели

Скрипт обучения: `src/train_seq2seq_spacing.py`

Запуск:

```bash
python3.12 src/train_seq2seq_spacing.py \
  --train_json data/word_segmentation_10k_labels.json \
  --model_name ai-forever/ruT5-large \
  --output_dir models/ruT5Large \
  --epochs 10 \
  --batch_size 20 \
  --lr 2e-4 \
  --val_frac 0.1 \
  --seed 42 \
  --freeze_encoder
```

### Детали обучения

* **Видеокарта**: NVIDIA RTX 5070
* **Загрузка видеопамяти**: \~9 ГБ только моделью
* **Время тренировки**: \~20 минут на 10 эпох
* **Датасет был сгенерирован синтетически**
* **Флаги**:

  * `--freeze_encoder` — заморозка энкодера (ускоряет обучение и стабилизирует на небольших датасетах).
  * `--val_frac 0.1` — 10% данных используется для валидации.
  * `--batch_size 20` и `--lr 2e-4` подобраны под GPU и модель `ruT5-large`.

---

## Инференс

Скрипт инференса: `src/infer_seq2seq_spacing.py`

Запуск:

```bash
python3.12 src/infer_seq2seq_spacing.py \
  --model_dir models/ruT5Large \
  --in_file data/dataset_1937770_3.csv \
  --in_format csv \
  --out_json data/pred8.jsonl \
  --out_txt data/pred_8.txt
```

### Детали инференса

* **Время инференса**: \~30 секунд на 1000 строк.
  
* Результаты сохраняются:

  * `pred.jsonl` — JSONL с `{"id":..., "spaced":..., "splits":[...]}`
  * `pred.txt` — текст с пробелами (по одной строке)

---

## Ресурсы и окружение

* Видеокарта: NVIDIA RTX 5070 (9 ГБ VRAM занято только моделью на обучении)
* Модель: models/ruT5Large 737M
* Дообученную модель можно скачать по ссылке: https://disk.yandex.ru/d/uDlgMV18LZMM7g
