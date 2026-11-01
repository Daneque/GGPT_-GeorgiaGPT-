# GeorgiaGPT

В проекте исследовалось обучение небольшой LLaMA-like модели для грузинского языка на веб-корпусе FineWeb2. Работа включала подготовку данных, выбор токенизатора, реализацию модели, обучение и отдельную проверку качества генерации.

### Данные и токенизация

Использовался грузинский раздел FineWeb2 — `kat_Geor`. Сначала был исследован корпус, затем создано воспроизводимое документное разбиение: 99% train и 1% validation.

Для токенизатора сравнивались варианты Byte-level BPE с разными размерами словаря. Итоговый словарь составил 16 384 токена; его обучение проводилось на выборке из 100 тысяч train-документов. Затем одним токенизатором был обработан весь корпус, в результате чего получены `train.bin` и `val.bin` в формате `uint16`.

Таким образом, выборка для BPE и данные для обучения модели различались по объёму: 100 тысяч документов использовались только при обучении токенизатора, а модель получала окна из основного корпуса.

### Архитектура и обучение

Модель была реализована на PyTorch. Использовался LLaMA-like decoder-only Transformer; в конфигурациях изменялся размер модели при сохранении контекста 512 токенов. Для экспериментов были подготовлены варианты примерно на 40 и 49 млн параметров.

Обучение было организовано через единый CLI с тремя командами: `start`, `resume`, `extend`. Были добавлены валидация, checkpoint с состоянием оптимизатора и scheduler, журнал метрик и мониторинг через TensorBoard. Команда `extend` позволяет продолжать обучение в отдельном запуске с новым расписанием LR, сохраняя параметры модели.

### Оценивание и инференс

Для сравнения генерации был составлен Benchmark v3: 18 заданий на продолжение текста, 18 на знание фактов и 18 на удержание контекста. Бенчмарк поддерживает пять режимов декодирования. Помимо сгенерированных текстов сохраняются top-10 следующих токенов, log-probability эталонного ответа и его ранг среди заданных альтернатив.

Были сопоставлены пять состояний моделей: 40M/200k и 49M после 100k, 200k, 300k и 400k шагов. Результаты бенчмарка, журналы обучения и качественные оценки продолжений собраны в `data/evaluation/ggpt_v3/`. Их анализ и визуализация выполнены в `notebooks/03_comparison.ipynb`.

Экспорт из training-checkpoint был вынесен в отдельный скрипт. Бенчмарк и инференс загружают экспортированные веса вместе с конфигурацией и токенизатором без запуска training-кода.

### Готовая модель

Для демонстрации генерации использована модель 49M после 300 тысяч шагов обучения с декодированием Top-k (`k=40`, `temperature=0.8`). Её inference-артефакт опубликован в репозитории:

```text
artifacts/exports/llama_49m_300k/
├── model.pt         # веса модели, Git LFS
├── tokenizer.json   # обученный BPE-токенизатор
└── manifest.json    # конфигурация и контрольные суммы
```

Файл `model.pt` хранится через Git LFS. Для получения весов после клонирования репозитория необходимо установить Git LFS и выполнить из корня проекта:

```bash
git lfs install
git lfs pull
python -m pip install -e '.[dev]'
```

После установки зависимостей откройте `notebooks/04_inference.ipynb`. Ноутбук загружает готовый артефакт, использует выбранные настройки Top-k и позволяет вводить собственный грузинский промпт.

Другие исследовательские ноутбуки описаны в `notebooks/README.md`.

### Основные команды

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Подготовка данных:

```bash
python scripts/data/download_fineweb2.py
python scripts/data/split_fineweb2.py
python scripts/tokenizer/prepare_sample.py --n-docs 100000 --seed 1729
python scripts/tokenizer/train_bpe.py --n-docs 100000 --seed 2718 --vocab-size 16384 --output-dir artifacts/tokenizer/bpe_100k_16384
python scripts/tokenizer/encode_corpus.py
```

Обучение и оценка:

```bash
python scripts/train_ggpt.py start --config configs/training/llama_49m.toml
python scripts/train_ggpt.py resume --run runs/ggpt/RUN_NAME
python scripts/train_ggpt.py extend --from runs/ggpt/PARENT_RUN --config configs/training/EXTEND_CONFIG.toml
python scripts/export_model.py --run runs/ggpt/RUN_NAME --output artifacts/exports/MODEL_NAME
python scripts/benchmark_v3.py --model artifacts/exports/MODEL_NAME --device cuda --output reports/benchmarks/RESULT_NAME
```

Основной код находится в `src/ggpt/`, сценарии запуска — в `scripts/`, конфигурации — в `configs/`, а наборы заданий — в `benchmarks/georgian/`.
