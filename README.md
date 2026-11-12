# Preprocessing Mismatch in Arabic Dialect Identification

Code accompanying the paper **“Preprocessing Mismatch in Arabic Dialect
Identification: A Worst-Case Study of Inconsistent Train–Test
Normalization.”**

This repository contains code only. It does not distribute the research data,
generated results, paper figures, model weights, or manuscript.

## Contents

```text
.
├── src/
│   ├── run_dialect_norm_experiments.py
│   ├── enhanced_norm_robustness.py
│   ├── deep_analysis_norm_mismatch.py
│   ├── revision_diagnostics.py
│   ├── transformer_robustness_check.py
│   └── balanced.py
├── README.md
├── LICENSE
└── requirements.txt
```

## Data preparation

Obtain the source data from its original provider and confirm that your use
complies with its license. Prepare one UTF-8 CSV file with these columns:

```csv
text,dialect
<Arabic text>,<dialect label>
```

The paper evaluates seven labels:

- Iraqi
- Lebanese
- Libyan
- Palestinian
- Sudanese
- Syrian
- Yemeni

The scripts accept other column names through `--text_col` and `--label_col`.
The reported balanced analyses use 833 examples per dialect, sampled with seed
42. Place your CSV anywhere and supply its path using `--data`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The classical experiments run on CPU. The MARBERT experiment is much faster on
a CUDA-capable GPU and downloads `UBC-NLP/MARBERT` from Hugging Face on first
use. The pinned environment uses the PyTorch 2.5.1 CUDA 12.1 build. If that
wheel is unavailable from your configured package index, install the matching
PyTorch build from the official PyTorch package index before installing the
remaining requirements.

## Reproduce the experiments

Run the following commands from the repository root. Replace
`path/to/my_7dialect_data.csv` with your prepared dataset.

### 1. Primary classical experiment

```bash
python src/run_dialect_norm_experiments.py \
  --data path/to/my_7dialect_data.csv \
  --out reproduced_results/classical_primary
```

This evaluates word, character, and combined word–character TF-IDF LinearSVC
models under matched and mismatched normalization conditions. Its outputs
include overall and per-class scores, confusion matrices, feature tables, and
`robustness_drops_raw_train.csv`.

For a quick smoke test:

```bash
python src/run_dialect_norm_experiments.py \
  --data path/to/my_7dialect_data.csv \
  --out reproduced_results/smoke_test \
  --seeds 42 \
  --models wordchar \
  --normalizers raw,glyph_fold_only,aggressive_glyph_fold,mask_dialect_glyphs
```

### 2. Multi-seed full-versus-balanced analysis

```bash
python src/enhanced_norm_robustness.py \
  --data path/to/my_7dialect_data.csv \
  --out reproduced_results/enhanced \
  --seeds 42,43,44
```

### 3. Deep error and feature analysis

```bash
python src/deep_analysis_norm_mismatch.py \
  --data path/to/my_7dialect_data.csv \
  --out reproduced_results/deep_analysis \
  --seed 42
```

### 4. Revision diagnostics

```bash
python src/revision_diagnostics.py \
  --data path/to/my_7dialect_data.csv \
  --out reproduced_results/revision_diagnostics \
  --dataset balanced \
  --seeds 42,43,44
```

### 5. MARBERT robustness check

```bash
python src/transformer_robustness_check.py \
  --data path/to/my_7dialect_data.csv \
  --out reproduced_results/transformer \
  --model_name UBC-NLP/MARBERT \
  --seeds 42,43,44 \
  --epochs 3 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_length 128 \
  --fp16
```

When CUDA is unavailable, remove `--fp16` and add `--cpu`.

### 6. Create a balanced dataset

```bash
python src/balanced.py \
  --data path/to/my_7dialect_data.csv \
  --out reproduced_results/balanced_833_per_dialect.csv \
  --rows_per_class 833 \
  --seed 42
```

## Reported experimental settings

The classical experiments use seeds 42, 43, and 44 and an 80/20 stratified
train–test split. Balanced analyses use 833 examples per dialect. The MARBERT
check uses the same seeds and split, 3 epochs, batch size 16, evaluation batch
size 32, maximum sequence length 128, learning rate 2e-5, weight decay 0.01,
and warmup ratio 0.06.

Use each script's `--help` option to see all available parameters, for example:

```bash
python src/run_dialect_norm_experiments.py --help
```
