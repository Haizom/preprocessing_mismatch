#!/usr/bin/env python3
"""
Minimal transformer robustness check for Arabic dialect normalization mismatch.

Purpose:
    Add one modern Arabic encoder experiment to support the paper's claim that
    preprocessing mismatch is not only an n-gram artifact.

Default model:
    UBC-NLP/MARBERT

Runs on the balanced dataset:
    1. raw -> raw
    2. raw -> aggressive
    3. aggressive -> aggressive
    4. aggressive -> raw

For each seed, the script trains:
    - one model on raw training text
    - one model on aggressively normalized training text

Outputs:
    transformer_results/
        all_results.csv
        summary_with_std.csv
        mcnemar_raw_vs_aggressive.csv
        per_class_results.csv
        confusion__*.csv
        predictions__*.csv
        experiment_metadata.json

Install:
    pip install pandas numpy scipy scikit-learn torch transformers tqdm

Example:
    python transformer_robustness_check.py \
      --data your_data.csv \
      --out transformer_results \
      --model_name UBC-NLP/MARBERT \
      --seeds 42,43,44 \
      --epochs 3 \
      --batch_size 16 \
      --max_length 128

Fast smoke test:
    python transformer_robustness_check.py \
      --data your_data.csv \
      --out transformer_smoke \
      --seeds 42 \
      --epochs 1 \
      --batch_size 16 \
      --max_length 128
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


# -----------------------------
# Normalization
# -----------------------------

AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}،؛؟…«»“”‘’ـ]+")
REPEAT_RE = re.compile(r"(.)\1{2,}")

ALEF_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
})

HAMZA_SEAT_MAP = str.maketrans({
    "ؤ": "و",
    "ئ": "ي",
})


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def raw(text: str) -> str:
    return normalize_spaces(text)


def remove_diacritics_tatweel(text: str) -> str:
    text = AR_DIACRITICS_RE.sub("", str(text))
    text = text.replace(TATWEEL, "")
    return text


def light(text: str) -> str:
    text = remove_diacritics_tatweel(text)
    text = text.translate(ALEF_MAP)
    text = text.replace("ى", "ي")
    return normalize_spaces(text)


def aggressive(text: str) -> str:
    text = light(text)
    text = text.translate(HAMZA_SEAT_MAP)
    text = text.replace("ة", "ه")
    text = PUNCT_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return normalize_spaces(text)


NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "raw": raw,
    "aggressive": aggressive,
}


# -----------------------------
# Data
# -----------------------------

def load_data(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns {text_col!r}, {label_col!r}. Found: {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_spaces)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df.replace({text_col: {"": np.nan}, label_col: {"": np.nan}})
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

    # Remove identical texts with multiple labels.
    nlabels = df.groupby(text_col)[label_col].nunique()
    ambiguous = set(nlabels[nlabels > 1].index)
    if ambiguous:
        df = df[~df[text_col].isin(ambiguous)].reset_index(drop=True)

    return df


def make_balanced(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    min_n = df[label_col].value_counts().min()
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=min_n, random_state=seed))
        .reset_index(drop=True)
    )


# -----------------------------
# Torch dataset
# -----------------------------

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


# -----------------------------
# Training / prediction
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model_name: str,
    train_texts: List[str],
    train_labels: List[int],
    tokenizer,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    args: argparse.Namespace,
    seed: int,
) -> torch.nn.Module:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    dataset = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.fp16))

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        total_loss = 0.0
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.fp16)):
                out = model(**batch)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=total_loss / max(1, pbar.n))

    return model


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    texts: List[str],
    labels: List[int],
    tokenizer,
    args: argparse.Namespace,
) -> np.ndarray:
    device = next(model.parameters()).device
    dataset = TextDataset(texts, labels, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)

    model.eval()
    preds = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        pred = out.logits.argmax(dim=-1).detach().cpu().numpy()
        preds.extend(pred.tolist())

    return np.array(preds)


# -----------------------------
# Metrics
# -----------------------------

def metric_row(y_true: np.ndarray, y_pred: np.ndarray, labels_int: List[int]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels_int),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels_int),
    }


def per_class_rows(y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict[int, str], labels_int: List[int]) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels_int, output_dict=True, zero_division=0)
    rows = []
    for i in labels_int:
        key = str(i)
        rows.append({
            "dialect": id2label[i],
            "precision": rep[key]["precision"],
            "recall": rep[key]["recall"],
            "f1": rep[key]["f1-score"],
            "support": rep[key]["support"],
        })
    return pd.DataFrame(rows)


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    discordant = b + c

    if discordant == 0:
        exact_p = 1.0
        chi2_stat = 0.0
        chi2_p = 1.0
    else:
        exact_p = float(binomtest(k=min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)
        chi2_stat = ((abs(b - c) - 1) ** 2) / discordant
        chi2_p = float(chi2.sf(chi2_stat, df=1))

    return {
        "b_raw_correct_aggressive_wrong": b,
        "c_raw_wrong_aggressive_correct": c,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": exact_p,
        "mcnemar_chi2_cc": chi2_stat,
        "mcnemar_chi2_p": chi2_p,
    }


# -----------------------------
# Main
# -----------------------------

def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data), args.text_col, args.label_col)
    balanced = make_balanced(df, args.label_col, args.balance_seed)

    labels = sorted(balanced[args.label_col].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    labels_int = list(range(len(labels)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    all_rows = []
    per_class_all = []
    mcnemar_rows = []

    metadata = {
        "model_name": args.model_name,
        "input_rows_after_cleaning": len(df),
        "balanced_total_rows": len(balanced),
        "balanced_rows_per_class": int(balanced[args.label_col].value_counts().min()),
        "labels": labels,
        "test_size": args.test_size,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
    }
    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        train_df, test_df = train_test_split(
            balanced,
            test_size=args.test_size,
            random_state=seed,
            stratify=balanced[args.label_col],
        )

        y_train = train_df[args.label_col].map(label2id).to_numpy()
        y_test = test_df[args.label_col].map(label2id).to_numpy()

        train_raw = train_df[args.text_col].map(raw).tolist()
        train_aggr = train_df[args.text_col].map(aggressive).tolist()
        test_raw = test_df[args.text_col].map(raw).tolist()
        test_aggr = test_df[args.text_col].map(aggressive).tolist()

        print("Training raw model...")
        raw_model = train_model(
            args.model_name, train_raw, y_train.tolist(), tokenizer,
            len(labels), id2label, label2id, args, seed
        )

        print("Evaluating raw-trained model...")
        pred_raw_raw = predict_model(raw_model, test_raw, y_test.tolist(), tokenizer, args)
        pred_raw_aggr = predict_model(raw_model, test_aggr, y_test.tolist(), tokenizer, args)

        print("Training aggressive model...")
        aggr_model = train_model(
            args.model_name, train_aggr, y_train.tolist(), tokenizer,
            len(labels), id2label, label2id, args, seed + 1000
        )

        print("Evaluating aggressive-trained model...")
        pred_aggr_aggr = predict_model(aggr_model, test_aggr, y_test.tolist(), tokenizer, args)
        pred_aggr_raw = predict_model(aggr_model, test_raw, y_test.tolist(), tokenizer, args)

        condition_preds = {
            ("raw", "raw"): pred_raw_raw,
            ("raw", "aggressive"): pred_raw_aggr,
            ("aggressive", "aggressive"): pred_aggr_aggr,
            ("aggressive", "raw"): pred_aggr_raw,
        }

        # Save predictions and metrics.
        for (train_norm, test_norm), pred in condition_preds.items():
            metrics = metric_row(y_test, pred, labels_int)
            all_rows.append({
                "seed": seed,
                "model": args.model_name,
                "train_norm": train_norm,
                "test_norm": test_norm,
                "mode": "same_pipeline" if train_norm == test_norm else "mismatch",
                **metrics,
            })

            pc = per_class_rows(y_test, pred, id2label, labels_int)
            pc.insert(0, "seed", seed)
            pc.insert(1, "model", args.model_name)
            pc.insert(2, "train_norm", train_norm)
            pc.insert(3, "test_norm", test_norm)
            pc.insert(4, "mode", "same_pipeline" if train_norm == test_norm else "mismatch")
            per_class_all.append(pc)

            cm = confusion_matrix(y_test, pred, labels=labels_int)
            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{id2label[i]}" for i in labels_int],
                columns=[f"pred_{id2label[i]}" for i in labels_int],
            )
            cm_df.to_csv(out_dir / f"confusion__seed-{seed}__train-{train_norm}__test-{test_norm}.csv", encoding="utf-8-sig")

            pred_df = test_df[[args.text_col, args.label_col]].copy()
            pred_df["gold_id"] = y_test
            pred_df["pred_id"] = pred
            pred_df["pred_label"] = [id2label[int(x)] for x in pred]
            pred_df["correct"] = pred == y_test
            pred_df["test_text_normalized"] = test_aggr if test_norm == "aggressive" else test_raw
            pred_df.to_csv(out_dir / f"predictions__seed-{seed}__train-{train_norm}__test-{test_norm}.csv", index=False, encoding="utf-8-sig")

        sig = mcnemar(y_test, pred_raw_raw, pred_raw_aggr)
        raw_metrics = metric_row(y_test, pred_raw_raw, labels_int)
        aggr_metrics = metric_row(y_test, pred_raw_aggr, labels_int)
        mcnemar_rows.append({
            "seed": seed,
            "model": args.model_name,
            "comparison": "raw->raw vs raw->aggressive",
            "raw_macro_f1": raw_metrics["macro_f1"],
            "raw_aggressive_macro_f1": aggr_metrics["macro_f1"],
            "macro_f1_drop": raw_metrics["macro_f1"] - aggr_metrics["macro_f1"],
            "raw_accuracy": raw_metrics["accuracy"],
            "raw_aggressive_accuracy": aggr_metrics["accuracy"],
            "accuracy_drop": raw_metrics["accuracy"] - aggr_metrics["accuracy"],
            **sig,
        })

        # Free memory between seeds.
        del raw_model
        del aggr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results = pd.DataFrame(all_rows)
    per_class = pd.concat(per_class_all, ignore_index=True)
    sig_df = pd.DataFrame(mcnemar_rows)

    all_results.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    sig_df.to_csv(out_dir / "mcnemar_raw_vs_aggressive.csv", index=False, encoding="utf-8-sig")

    summary = (
        all_results
        .groupby(["model", "train_norm", "test_norm", "mode"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
        )
    )
    summary.to_csv(out_dir / "summary_with_std.csv", index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(summary.to_string(index=False))
    print("\nMcNemar:")
    print(sig_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="transformer_results")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="dialect")
    parser.add_argument("--model_name", default="UBC-NLP/MARBERT")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--balance_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
