# Utils

В модуле `logger.py` реализована общая система логирования GGPT на основе стандартного Python `logging`.

`setup_logging` настраивает вывод сообщений в консоль и файл с автоматической ротацией. `get_logger` предоставляет именованные логгеры в пространстве `ggpt`.

Логирование используется при подготовке данных, обучении и запуске бенчмарков.

Структурированные метрики обучения обрабатываются отдельно в `ggpt.training.logging` и записываются в JSONL и TensorBoard.

### Проверка

```bash
python -m pytest -q \
  tests/training/test_training_logger_integration.py \
  tests/benchmark/test_v3_logging.py
```
