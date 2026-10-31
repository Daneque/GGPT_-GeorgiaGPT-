# Data Preparation

В каталоге находятся скрипты загрузки корпуса FineWeb2 и формирования train/validation-разбиения.

`download_fineweb2.py` скачивает раздел `kat_Geor` из FineWeb2 и сохраняет манифест с ревизией датасета.

`split_fineweb2.py` формирует воспроизводимое разбиение исходного train на обучающую и валидационную части в пропорции 99/1. Для валидации сохраняются индексы документов, отобранные с фиксированным seed.

### Схема подготовки

```text
FineWeb2 / kat_Geor
         ↓
download_fineweb2.py
         ↓
Исходные Parquet-файлы
         ↓
split_fineweb2.py
         ↓
validation_indices.npy
manifest.json
         ↓
Подготовка токенизатора
и кодирование корпуса
```

### Запуск

Загрузка корпуса:

```bash
python scripts/data/download_fineweb2.py
```

Создание train/validation-разбиения:

```bash
python scripts/data/split_fineweb2.py
```
