# Training

Система обучения организована вокруг конфигурации эксперимента, цикла оптимизации и сохранения состояния.

Параметры загружаются из TOML через `experiment_config.py`. Модуль `experiment.py` управляет тремя сценариями запуска:

- `start` — создание нового эксперимента;
- `resume` — восстановление прерванного обучения;
- `extend` — продолжение завершённого эксперимента с новым расписанием learning rate.

Основной цикл реализован в `trainer.py`. Для оптимизации используется AdamW.

Через заданные интервалы выполняются валидация, сохранение checkpoint и запись метрик. Валидация рассчитывает cross-entropy loss и perplexity на фиксированных окнах корпуса. Метрики сохраняются в JSONL и TensorBoard.

Checkpoint включает параметры модели, состояния оптимизатора и scheduler, генераторов случайных чисел, семплера и текущий прогресс обучения.

При `resume` восстанавливается исходный эксперимент. При `extend` создаётся отдельный эксперимент на основе завершённого запуска с сохранением состояний обучения и возможностью задать новое расписание LR.

### Запуск

Начало обучения:

```bash
python scripts/train_ggpt.py start \
  --config configs/training/llama_49m.toml
```

Восстановление прерванного эксперимента:

```bash
python scripts/train_ggpt.py resume \
  --run runs/ggpt/RUN_NAME
```

Продолжение завершённого эксперимента:

```bash
python scripts/train_ggpt.py extend \
  --from runs/ggpt/PARENT_RUN \
  --config configs/training/EXTEND_CONFIG.toml
```

Мониторинг обучения:

```bash
tensorboard --logdir runs/ggpt --port 6006
```

Проверка системы обучения:

```bash
python -m pytest -q tests/training
```
