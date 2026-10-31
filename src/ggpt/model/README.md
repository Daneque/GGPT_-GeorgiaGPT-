# Model

В модуле реализована LLaMA-like языковая модель на PyTorch.

Архитектура задаётся через `ModelConfig`, содержащий параметры размерности, количества слоёв, голов внимания и максимальной длины контекста. Создание модели вынесено в `factory.py`.

Основная реализация находится в `llama.py`. Модель преобразует входные идентификаторы токенов в эмбеддинги, последовательно обрабатывает их блоками Transformer и возвращает logits по словарю. При передаче целевых токенов дополнительно вычисляется cross-entropy loss.

### Архитектура

```text
Input token IDs [B, T]
        │
        ▼
Token Embeddings
        │
        ▼
┌──────────────────────────────────────┐
│          Transformer Block           │
│                                      │
│  RMSNorm                             │
│      ↓                               │
│  Causal Multi-Head Attention         │
│  (RoPE для Q и K)                    │
│      ↓                               │
│  Residual Connection                 │
│      ↓                               │
│  RMSNorm                             │
│      ↓                               │
│  SwiGLU FFN                          │
│      ↓                               │
│  Residual Connection                 │
│                                      │
└──────────────────────────────────────┘
             × N layers
                 │
                 ▼
            Final RMSNorm
                 │
                 ▼
               LM Head
       (weight tying с Embeddings)
                 │
                 ▼
         Logits [B, T, V]
                 │
                 ▼
       Cross-Entropy Loss
       (при наличии targets)
```

Компоненты блоков разделены между `attention.py`, `rope.py` и `layers.py`. Нормализация выполняется перед Attention и FFN; выходы обоих подблоков складываются с остаточными соединениями.

### Проверка

Тестирование конфигурации, компонентов Transformer и полного forward pass:

```bash
python -m pytest -q tests/model
```
