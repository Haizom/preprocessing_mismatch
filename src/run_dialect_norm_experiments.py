#!/usr/bin/env python3
"""
Arabic dialect normalization robustness experiments.

Input CSV format:
    text,dialect
    گولو نريد نفتح الطريق للايرانيين وفضوها,Iraqi
    ...

Recommended:
    python run_dialect_norm_experiments.py --data your_data.csv --out results

Outputs:
    - data_profile.csv
    - dialect_glyph_profile.csv
    - all_results.csv
    - summary_results.csv
    - robustness_drops_raw_train.csv
    - per_class_results.csv
    - confusion matrices
    - top feature files
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}،؛؟…«»“”‘’ـ]+")
REPEAT_RE = re.compile(r"(.)\1{2,}")

DIALECT_GLYPH_MAP = str.maketrans({
    "گ": "ك", "ڨ": "ك", "ڬ": "ك", "ڭ": "ك", "ک": "ك", "ڪ": "ك",
    "چ": "ج",
    "ڤ": "ف", "ڥ": "ف",
    "پ": "ب",
})

ALEF_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
})

HAMZA_SEAT_MAP = str.maketrans({
    "ؤ": "و", "ئ": "ي",
})


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def raw(text: str) -> str:
    return normalize_spaces(text)


def remove_diacritics_tatweel(text: str) -> str:
    text = AR_DIACRITICS_RE.sub("", str(text))
    text = text.replace(TATWEEL, "")
    return text


def light_ar_norm(text: str) -> str:
    text = remove_diacritics_tatweel(text)
    text = text.translate(ALEF_MAP)
    text = text.replace("ى", "ي")
    return normalize_spaces(text)


def punct_removed(text: str) -> str:
    text = light_ar_norm(text)
    text = PUNCT_RE.sub(" ", text)
    return normalize_spaces(text)


def aggressive_ar_norm(text: str) -> str:
    text = light_ar_norm(text)
    text = text.translate(HAMZA_SEAT_MAP)
    text = text.replace("ة", "ه")
    text = PUNCT_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return normalize_spaces(text)


def glyph_fold_only(text: str) -> str:
    text = str(text).translate(DIALECT_GLYPH_MAP)
    return normalize_spaces(text)


def aggressive_plus_glyph_fold(text: str) -> str:
    text = aggressive_ar_norm(text)
    text = text.translate(DIALECT_GLYPH_MAP)
    return normalize_spaces(text)


def mask_dialect_glyphs(text: str) -> str:
    text = str(text)
    text = re.sub(r"[گڨڬڭکڪچڤڥپ]", " <DIALECT_GLYPH> ", text)
    return normalize_spaces(text)


log = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m{s:.0f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m}m"


def _count_evals_per_block(train_norm: str, selected_norms: List[str], full_matrix: bool) -> int:
    n = 0
    for test_norm in selected_norms:
        if not full_matrix:
            if not (train_norm == test_norm or train_norm == "raw" or test_norm == "raw"):
                continue
        n += 1
    return n


NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "raw": raw,
    "light": light_ar_norm,
    "punct_removed": punct_removed,
    "aggressive": aggressive_ar_norm,
    "glyph_fold_only": glyph_fold_only,
    "aggressive_glyph_fold": aggressive_plus_glyph_fold,
    "mask_dialect_glyphs": mask_dialect_glyphs,
}


def build_model(model_name: str, max_word_features: int, max_char_features: int) -> Pipeline:
    if model_name == "word":
        feats = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=max_word_features, sublinear_tf=True,
        )
    elif model_name == "char":
        feats = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            max_features=max_char_features, sublinear_tf=True,
        )
    elif model_name == "wordchar":
        feats = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=max_word_features, sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2,
                max_features=max_char_features, sublinear_tf=True,
            )),
        ])
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    clf = LinearSVC(C=1.0, class_weight="balanced", random_state=0)
    return Pipeline([("features", feats), ("clf", clf)])


def get_feature_names(pipe: Pipeline) -> np.ndarray:
    return pipe.named_steps["features"].get_feature_names_out()


def load_and_clean_data(path: Path, text_col: str, label_col: str, drop_ambiguous_texts: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns {text_col!r} and {label_col!r}. Found: {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_spaces)
    df[label_col] = df[label_col].astype(str).str.strip()

    df = df.replace({text_col: {"": np.nan}, label_col: {"": np.nan}})
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

    if drop_ambiguous_texts:
        nlabels = df.groupby(text_col)[label_col].nunique()
        ambiguous_texts = set(nlabels[nlabels > 1].index)
        if ambiguous_texts:
            df = df[~df[text_col].isin(ambiguous_texts)].reset_index(drop=True)

    return df


def save_data_profile(df: pd.DataFrame, out_dir: Path, text_col: str, label_col: str) -> None:
    profile = (
        df.groupby(label_col)[text_col]
        .agg(rows="count", avg_chars=lambda s: float(np.mean([len(x) for x in s])))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    profile.to_csv(out_dir / "data_profile.csv", index=False, encoding="utf-8-sig")

    glyph_counts = []
    glyph_re = re.compile(r"[گڨڬڭکڪچڤڥپ]")
    for label, part in df.groupby(label_col):
        total = len(part)
        has_glyph = int(part[text_col].map(lambda x: bool(glyph_re.search(x))).sum())
        glyph_counts.append({
            "dialect": label,
            "rows": total,
            "rows_with_dialect_glyph": has_glyph,
            "pct_with_dialect_glyph": has_glyph / total if total else 0.0,
        })
    pd.DataFrame(glyph_counts).sort_values("pct_with_dialect_glyph", ascending=False).to_csv(
        out_dir / "dialect_glyph_profile.csv", index=False, encoding="utf-8-sig"
    )


def evaluate_one(pipe: Pipeline, X_test: List[str], y_test: List[str], labels: List[str]) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    pred = pipe.predict(X_test)
    row = {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }

    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    per_class_rows = []
    for label in labels:
        per_class_rows.append({
            "dialect": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        })
    per_class_df = pd.DataFrame(per_class_rows)

    cm = confusion_matrix(y_test, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    return row, per_class_df, cm_df


def save_top_features(pipe: Pipeline, labels: List[str], out_path: Path, top_k: int = 40) -> None:
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return

    feature_names = get_feature_names(pipe)
    rows = []
    for class_idx, label in enumerate(labels):
        coefs = clf.coef_[class_idx]
        top_idx = np.argsort(coefs)[-top_k:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "dialect": label,
                "rank": rank,
                "feature": feature_names[idx],
                "weight": float(coefs[idx]),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    _bc_kwargs = {
        "level": logging.DEBUG if args.verbose else logging.INFO,
        "format": "%(asctime)s | %(levelname)-5s | %(message)s",
        "datefmt": "%H:%M:%S",
    }
    if sys.version_info >= (3, 8):
        _bc_kwargs["force"] = True
    logging.basicConfig(**_bc_kwargs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_run0 = time.perf_counter()

    log.info("Loading CSV: %s", Path(args.data).resolve())
    t0 = time.perf_counter()
    df = load_and_clean_data(
        Path(args.data),
        args.text_col,
        args.label_col,
        drop_ambiguous_texts=not args.keep_ambiguous_texts,
    )
    log.info("Loaded and cleaned: %s rows in %s", f"{len(df):,}", _fmt_duration(time.perf_counter() - t0))

    t0 = time.perf_counter()
    save_data_profile(df, out_dir, args.text_col, args.label_col)
    log.info("Wrote data profiles in %s", _fmt_duration(time.perf_counter() - t0))

    labels = sorted(df[args.label_col].unique().tolist())
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    selected_norms = args.normalizers.split(",")
    for n in selected_norms:
        if n not in NORMALIZERS:
            raise ValueError(f"Unknown normalizer {n!r}. Available: {list(NORMALIZERS)}")

    selected_models = args.models.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    total_blocks = len(seeds) * len(selected_models) * len(selected_norms)
    total_evals = sum(
        _count_evals_per_block(tn, selected_norms, args.full_matrix)
        for _ in seeds
        for _ in selected_models
        for tn in selected_norms
    )
    log.info(
        "Plan: %s seeds × %s models × %s train normalizers = %s fit blocks; ~%s test evaluations total; full_matrix=%s",
        len(seeds),
        len(selected_models),
        len(selected_norms),
        total_blocks,
        total_evals,
        args.full_matrix,
    )

    all_rows = []
    per_class_all = []

    block_idx = 0
    block_times: List[float] = []
    for seed in seeds:
        log.info("--- Seed %s (split stratified, test_size=%s) ---", seed, args.test_size)
        t_split = time.perf_counter()
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=seed,
            stratify=df[args.label_col],
        )
        log.info(
            "Train/test split: train=%s test=%s in %s",
            f"{len(train_df):,}",
            f"{len(test_df):,}",
            _fmt_duration(time.perf_counter() - t_split),
        )

        y_train = train_df[args.label_col].tolist()
        y_test = test_df[args.label_col].tolist()

        train_texts: Dict[str, List[str]] = {}
        test_texts: Dict[str, List[str]] = {}
        t_mat0 = time.perf_counter()
        for norm_name in selected_norms:
            t_n = time.perf_counter()
            train_texts[norm_name] = train_df[args.text_col].map(NORMALIZERS[norm_name]).tolist()
            log.info("  train texts [%s]: %s", norm_name, _fmt_duration(time.perf_counter() - t_n))
            t_n = time.perf_counter()
            test_texts[norm_name] = test_df[args.text_col].map(NORMALIZERS[norm_name]).tolist()
            log.info("  test texts  [%s]: %s", norm_name, _fmt_duration(time.perf_counter() - t_n))
        log.info("All normalizer variants materialized in %s", _fmt_duration(time.perf_counter() - t_mat0))

        for model_name in selected_models:
            for train_norm in selected_norms:
                block_idx += 1
                n_eval_this = _count_evals_per_block(train_norm, selected_norms, args.full_matrix)
                eta_blocks = "?"
                if block_times:
                    avg_b = float(np.mean(block_times))
                    eta_blocks = _fmt_duration(avg_b * (total_blocks - block_idx + 1))
                log.info(
                    "Block %s/%s | seed=%s model=%s train_norm=%s (%s evals) | ETA remaining ~%s (from fit blocks)",
                    block_idx,
                    total_blocks,
                    seed,
                    model_name,
                    train_norm,
                    n_eval_this,
                    eta_blocks,
                )
                t_block = time.perf_counter()
                pipe = build_model(model_name, args.max_word_features, args.max_char_features)
                t_fit = time.perf_counter()
                pipe.fit(train_texts[train_norm], y_train)
                log.info("  fit done in %s", _fmt_duration(time.perf_counter() - t_fit))

                if seed == seeds[0] and model_name in {"word", "char", "wordchar"} and train_norm in {"raw", "glyph_fold_only", "aggressive_glyph_fold"}:
                    t_tf = time.perf_counter()
                    save_top_features(
                        pipe, labels,
                        out_dir / f"top_features__model-{model_name}__train-{train_norm}.csv",
                        top_k=args.top_k,
                    )
                    log.debug("  top_features saved in %s", _fmt_duration(time.perf_counter() - t_tf))

                eval_i = 0
                t_eval0 = time.perf_counter()
                for test_norm in selected_norms:
                    if not args.full_matrix:
                        keep = train_norm == test_norm or train_norm == "raw" or test_norm == "raw"
                        if not keep:
                            continue

                    eval_i += 1
                    t_ev = time.perf_counter()
                    metrics, per_class_df, cm_df = evaluate_one(pipe, test_texts[test_norm], y_test, labels)
                    log.debug(
                        "    eval %s/%s test_norm=%s in %s | acc=%.4f macro_f1=%.4f",
                        eval_i,
                        n_eval_this,
                        test_norm,
                        _fmt_duration(time.perf_counter() - t_ev),
                        metrics["accuracy"],
                        metrics["macro_f1"],
                    )

                    mode = "same_pipeline" if train_norm == test_norm else "mismatch"
                    row = {
                        "seed": seed,
                        "model": model_name,
                        "train_norm": train_norm,
                        "test_norm": test_norm,
                        "mode": mode,
                        **metrics,
                    }
                    all_rows.append(row)

                    per_class_df.insert(0, "seed", seed)
                    per_class_df.insert(1, "model", model_name)
                    per_class_df.insert(2, "train_norm", train_norm)
                    per_class_df.insert(3, "test_norm", test_norm)
                    per_class_df.insert(4, "mode", mode)
                    per_class_all.append(per_class_df)

                    if seed == seeds[0] and model_name == args.main_model:
                        cm_df.to_csv(
                            out_dir / f"confusion__model-{model_name}__train-{train_norm}__test-{test_norm}__seed-{seed}.csv",
                            encoding="utf-8-sig",
                        )

                block_dt = time.perf_counter() - t_block
                block_times.append(block_dt)
                rem = total_blocks - block_idx
                eta_after = _fmt_duration(float(np.mean(block_times)) * rem) if rem else "0s"
                log.info(
                    "  block done in %s (evaluations %s in %s) | progress %s/%s | ETA remaining ~%s",
                    _fmt_duration(block_dt),
                    eval_i,
                    _fmt_duration(time.perf_counter() - t_eval0),
                    block_idx,
                    total_blocks,
                    eta_after,
                )

    log.info("Writing result tables...")
    t_w = time.perf_counter()
    all_results = pd.DataFrame(all_rows)
    all_results.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8-sig")

    summary = (
        all_results
        .groupby(["model", "train_norm", "test_norm", "mode"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
        )
        .sort_values(["model", "macro_f1_mean"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "summary_results.csv", index=False, encoding="utf-8-sig")

    per_class_results = pd.concat(per_class_all, ignore_index=True)
    per_class_results.to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")

    drops = []
    for model in selected_models:
        base_rows = summary[
            (summary["model"] == model) &
            (summary["train_norm"] == "raw") &
            (summary["test_norm"] == "raw")
        ]
        if base_rows.empty:
            continue
        base_f1 = float(base_rows.iloc[0]["macro_f1_mean"])
        for _, r in summary[(summary["model"] == model) & (summary["train_norm"] == "raw")].iterrows():
            drops.append({
                "model": model,
                "baseline_train_test": "raw->raw",
                "test_condition": f"raw->{r['test_norm']}",
                "baseline_macro_f1": base_f1,
                "condition_macro_f1": r["macro_f1_mean"],
                "drop_points": base_f1 - r["macro_f1_mean"],
            })
    pd.DataFrame(drops).sort_values(["model", "drop_points"], ascending=[True, False]).to_csv(
        out_dir / "robustness_drops_raw_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log.info("Aggregations and CSV writes finished in %s", _fmt_duration(time.perf_counter() - t_w))

    log.info("Done. Total wall time %s", _fmt_duration(time.perf_counter() - t_run0))
    log.info("Rows after cleaning: %s", f"{len(df):,}")
    log.info("Labels: %s", labels)
    log.info("Results saved to: %s", out_dir.resolve())
    log.info("Top summary rows:\n%s", summary.head(20).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file.")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="dialect")
    parser.add_argument("--out", default="results_dialect_norm")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--models", default="word,char,wordchar")
    parser.add_argument(
        "--normalizers",
        default="raw,light,punct_removed,aggressive,glyph_fold_only,aggressive_glyph_fold,mask_dialect_glyphs",
    )
    parser.add_argument("--main_model", default="wordchar")
    parser.add_argument("--max_word_features", type=int, default=120000)
    parser.add_argument("--max_char_features", type=int, default=200000)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--full_matrix", action="store_true", help="Evaluate every train_norm x test_norm pair.")
    parser.add_argument("--keep_ambiguous_texts", action="store_true", help="Keep identical texts with multiple labels.")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging (per-test-eval timings).")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
