# Benchmark

Модуль реализует запуск и сбор результатов тестирования грузинской языковой модели.

Для оценки был разработан Benchmark v3, состоящий из 54 заданий: 18 на свободное продолжение текста, 18 на знание фактов о Грузии и 18 на удержание контекста.

Набор заданий загружается и проверяется в `v3_dataset.py`. Модуль `v3_runner.py` управляет выполнением заданий в пяти режимах генерации: greedy, top-k, top-p, conservative и diverse.

Для заданий с эталонными ответами в `v3_scoring.py` реализованы вычисление log-probability правильного ответа, ранжирование четырёх альтернатив и получение десяти наиболее вероятных следующих BPE-токенов. Контекстные задания дополнительно сравниваются с вариантами без ключевой информации.

### Схема оценивания

```text
Benchmark v3 (54 задания)
           ↓
      v3_runner.py
           │
           ├── Генерация в 5 режимах
           │         ↓
           │   generations.jsonl
           │
           └── Вероятностная оценка
                     ↓
                scores.jsonl
           ↓
    manifest.json / summary.json
```

Бенчмарк загружает модель через `ggpt.inference` из экспортированного артефакта. Результаты записываются по мере выполнения; поддерживается возобновление прерванного запуска.

### Запуск

Полный Benchmark v3:

```bash
python scripts/benchmark_v3.py \
  --model artifacts/exports/MODEL_NAME \
  --device cuda \
  --output reports/benchmarks/RESULT_NAME
```

Пробный запуск по два задания из каждой категории:

```bash
python scripts/benchmark_v3.py \
  --model artifacts/exports/MODEL_NAME \
  --device cuda \
  --limit 2 \
  --output reports/benchmarks/SMOKE_TEST
```

Возобновление:

```bash
python scripts/benchmark_v3.py \
  --model artifacts/exports/MODEL_NAME \
  --device cuda \
  --output reports/benchmarks/RESULT_NAME \
  --resume
```

Проверка:

```bash
python -m pytest -q tests/benchmark
```
