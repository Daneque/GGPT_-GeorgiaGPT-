# Configs

Параметры обучения и генерации вынесены в TOML-конфигурации.

### Конфигурации обучения

В `training/` находятся конфигурации моделей размером около 40 и 49 млн параметров, а также отдельных стадий продолжения обучения.

Файлы включают следующие секции:

```text
[run]         — название и каталог эксперимента
[model]       — архитектура модели
[data]        — токенизированный корпус и параметры загрузки
[training]    — количество шагов, batch size, precision, seed
[optimizer]   — параметры AdamW
[scheduler]   — расписание learning rate
[monitoring]  — валидация, checkpoints и логирование
```

Для первоначального обучения используется `warmup_cosine`, для дополнительных стадий — `continuation_cosine`. В последнем случае число дополнительных шагов и новое расписание LR задаются отдельной конфигурацией.

Конфигурации загружаются и проверяются в `ggpt.training.experiment_config`.

### Конфигурации бенчмарка

Режимы генерации Benchmark v3 определены в `ggpt.benchmark.v3_runner` и запускаются через `scripts/benchmark_v3.py`.

### Запуск

```bash
python scripts/train_ggpt.py start \
  --config configs/training/llama_49m.toml

python scripts/train_ggpt.py extend \
  --from runs/ggpt/PARENT_RUN \
  --config configs/training/EXTEND_CONFIG.toml
```
