# Scripts

В каталоге находятся точки входа для подготовки данных, обучения модели и проведения экспериментов.

### Основные скрипты

- `data/` — загрузка FineWeb2 и формирование train/validation-разбиения.
- `tokenizer/` — подготовка обучающего корпуса для BPE, обучение токенизатора и кодирование данных.
- `train_ggpt.py` — создание, восстановление и продолжение обучения.
- `export_model.py` — экспорт весов, конфигурации и токенизатора для инференса.
- `benchmark_v3.py` — запуск Benchmark v3 на экспортированной модели.

Основные вычисления реализованы в `src/ggpt/`, а скрипты отвечают за обработку аргументов и запуск соответствующих модулей.

### Запуск

Обучение:

```bash
python scripts/train_ggpt.py start \
  --config configs/training/llama_49m.toml
```

Экспорт:

```bash
python scripts/export_model.py \
  --run runs/ggpt/RUN_NAME \
  --output artifacts/exports/MODEL_NAME
```

Оценивание:

```bash
python scripts/benchmark_v3.py \
  --model artifacts/exports/MODEL_NAME \
  --device cuda \
  --output reports/benchmarks/RESULT_NAME
```
