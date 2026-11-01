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

### Готовый inference-артефакт

В репозитории находится экспорт модели 49M/300k:

```text
artifacts/exports/llama_49m_300k/
├── model.pt
├── tokenizer.json
└── manifest.json
```

Веса `model.pt` хранятся через Git LFS; для их получения после клонирования используется `git lfs pull`. Модель загружается напрямую из этого каталога. Для интерактивной генерации предусмотрен `notebooks/04_inference.ipynb` с настройками Top-k (`k=40`, `temperature=0.8`).

### Запуск

Экспорт обученной модели из другого checkpoint (для готового артефакта не требуется):

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
