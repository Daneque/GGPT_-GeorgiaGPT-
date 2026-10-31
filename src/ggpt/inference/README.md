# Inference

Модуль объединяет загрузку обученной модели, авторегрессионную генерацию и работу с токенизатором.

В `artifact.py` реализованы экспорт обучающего checkpoint и загрузка самостоятельного артефакта модели. При экспорте сохраняются веса, конфигурация модели, токенизатор и манифест с контрольными суммами.

Генерация реализована в `generate.py`. Поддерживаются greedy decoding и вероятностное декодирование с настройками temperature, top-k и top-p.

`model_runner.py` объединяет токенизацию входного текста, генерацию последовательности и декодирование результата.

### Схема инференса

```text
Исходный текст
      ↓
Byte-level BPE
      ↓
Input token IDs
      ↓
    GGPT
      ↓
Logits следующего токена
      ↓
Greedy / Sampling
      ↓
Добавление нового токена
      │
      └── Повтор до EOS или max_new_tokens
      ↓
BPE decode
      ↓
Сгенерированный текст
```

### Запуск

Экспорт обученной модели:

```bash
python scripts/export_model.py \
  --run runs/ggpt/RUN_NAME \
  --checkpoint latest \
  --output artifacts/exports/MODEL_NAME
```

Проверка генерации и экспорта:

```bash
python -m pytest -q tests/inference tests/benchmark/test_inference_artifact.py
```
