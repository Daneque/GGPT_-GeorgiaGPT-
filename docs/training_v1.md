# GeorgiaGPT: trainer v1

## Purpose

This adds restartable **single-GPU** pretraining for the existing LLaMA-like model. It does not change the model, dataset, tokenizer or tokenized binary files. It depends on the **repaired** RoPE cache implementation in `src/ggpt/model/rope.py` (the fix for inference tensors used in backward).

Add these files into `~/projects/GGPT` preserving their paths. This package deliberately includes **only new files plus the updated `src/ggpt/training/__init__.py`**, not old model / data files.

## Dependencies

In the active `ggpt` conda environment:

```bash
uv pip install --python "$(which python)" tensorboard
```

The project already uses `torch`, `numpy`, `tokenizers`, `pytest`. An installation of PyTorch compatible with the RTX 5060 and BF16 is needed for GPU training.

## Tests

```bash
cd ~/projects/GGPT
PYTHONPATH=src python -m pytest tests/ -q
```

The CLI defaults to the real FineWeb2 tokenized files and existing BPE:
- `data/tokenized/kat_Geor/train.bin`
- `data/tokenized/kat_Geor/val.bin`
- `artifacts/tokenizer/bpe_100k_16384/tokenizer.json`

## Smoke run with checkpoint resume

```bash
bash scripts/run_smoke.sh start
bash scripts/run_smoke.sh resume
```

The first command stops at **global step 12**, leaving `runs/ggpt/llama_40m_smoke/checkpoints/latest.pt`. The second resumes exactly the same run until **global step 40**. Check `metrics.jsonl` for `resume`, `train`, `val`, `generation`, `checkpoint` and `finished` events. Look for finite losses and decreasing validation loss, but expect noisy short-run fluctuations.

## Main run

Once smoke/resume is confirmed, run:

```bash
bash scripts/run_main.sh start
```

To continue after a clean pause or previous checkpoint:

```bash
bash scripts/run_main.sh resume
```

Main defaults: **40,020,480** parameters, **512** context, **BF16 autocast**, AdamW base learning rate **2e-4**, **400** warmup updates, cosine LR down to **2e-5**, microbatch **4**, accumulation **4** (16 sequences = 8192 target tokens per optimizer step), gradient clipping **1.0**, 60,000 optimizer updates. About **491.5 million training target tokens** if completed. This is sampled across the entire train.bin stream, not a sequential complete pass or full coverage guarantee.

The smoke results previously showed microbatch 4 was viable on RTX 5060, but full-training stability and speed remain to be measured.

## Safe pause and recovery

The preferred way to stop a running main experiment is to create a marker file in a **second terminal**:

```bash
touch runs/ggpt/llama_40m_v1/STOP
```

The trainer notices it **after completing** an optimizer step, writes an atomic `latest.pt` checkpoint, removes the marker, and exits. To resume, use `bash scripts/run_main.sh resume`.

SIGINT (Ctrl-C) can interrupt a PyTorch update mid-step. When that happens the trainer deliberately **does not checkpoint potentially half-updated optimizer state**. Resume from the last previously written `latest.pt` instead; any updates after it will be repeated.

Never change LR, scheduler horizon, batch size, corpus, tokenizer, training seed, or other run settings while resuming the same run. The saved run specification is validated; use a new `--run-name` for a different experiment. The `--stop-after` argument is allowed to change because it only controls when to pause (it is an absolute global update number).

Each run uses these files:

```
runs/ggpt/llama_40m_v1/
├── run_spec.json
├── metrics.jsonl
├── tensorboard/
└── checkpoints/
    ├── latest.pt
    └── best.pt
```

Checkpoint files store model, AdamW state, scheduler state, global step, token counter, committed sampler cursor, Python/NumPy/PyTorch/CUDA RNG state. `latest.pt` is atomically replaced after successful save; `best.pt` is a separate copy when validation reaches a new minimum. Do not put `runs/ggpt/` into Git or DVC while working on this stage. Keep an independent backup of checkpoints if they matter.

Check TensorBoard:

```bash
tensorboard --logdir runs/ggpt --port 6006
```

Open `http://localhost:6006/` in a browser. JSONL has the same key metrics: `train/loss`, `train/grad_norm`, `train/lr`, `train/tokens_per_second`, `train/peak_vram_mib`, `val/loss`, `val/perplexity` and generated text.

## Limitations / intended next improvements

- CUDA floating-point kernels are not all necessarily bitwise-deterministic. Tests show identical **CPU** weights for uninterrupted vs resumed training, and the sampler resumes exactly even with worker prefetch.
- Validation is on **256 fixed windows**, distributed across the held-out validation bin. This is a fast, stable validation sample, not a full-corpus quality estimate.
- Generation is deliberately *without* a KV cache, so it can take time at 64 new tokens x 4 prompts.
- Tokens per second from the trainer excludes time spent loading a microbatch into host RAM; it does include host-to-GPU transfer and optimizer step. Evaluation/checkpoint/generation time is also excluded from the per-step metric.
- Checkpoint integrity checks ensure tokenizer SHA256 and training-file size plus SHA256 of first/last MiB. This is not a full cryptographic digest of large `.bin` files.
- A failed/corrupt checkpoint is not automatically repaired; restart from a verified copy. `best.pt` is selected by the small fixed validation sample.
- No early stopping, multi-GPU, mixed precision FP16, or DVC remote at this stage.
