
import argparse
import json
import re
import string
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


# -----------------------------
# Normalization functions
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

DIALECT_GLYPH_MAP = str.maketrans({
    "گ": "ك", "ڨ": "ك", "ڬ": "ك", "ڭ": "ك", "ک": "ك", "ڪ": "ك",
    "چ": "ج",
    "ڤ": "ف", "ڥ": "ف",
    "پ": "ب",
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


def punct_removed(text: str) -> str:
    text = light(text)
    text = PUNCT_RE.sub(" ", text)
    return normalize_spaces(text)


def aggressive(text: str) -> str:
    text = light(text)
    text = text.translate(HAMZA_SEAT_MAP)
    text = text.replace("ة", "ه")
    text = PUNCT_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return normalize_spaces(text)


def glyph_fold_only(text: str) -> str:
    text = str(text).translate(DIALECT_GLYPH_MAP)
    return normalize_spaces(text)


def aggressive_glyph_fold(text: str) -> str:
    text = aggressive(text)
    text = text.translate(DIALECT_GLYPH_MAP)
    return normalize_spaces(text)


NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "raw": raw,
    "light": light,
    "punct_removed": punct_removed,
    "aggressive": aggressive,
    "glyph_fold_only": glyph_fold_only,
    "aggressive_glyph_fold": aggressive_glyph_fold,
}


# -----------------------------
# Model builders
# -----------------------------

def build_model(model_name: str, max_word_features: int, max_char_features: int) -> Pipeline:
    if model_name == "word":
        features = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_word_features,
            sublinear_tf=True,
        )
    elif model_name == "char":
        features = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=max_char_features,
            sublinear_tf=True,
        )
    elif model_name == "wordchar":
        features = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=max_word_features,
                sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=max_char_features,
                sublinear_tf=True,
            )),
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    clf = LinearSVC(C=1.0, class_weight="balanced", random_state=0)
    return Pipeline([("features", features), ("clf", clf)])


# -----------------------------
# Data utilities
# -----------------------------

def load_data(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns {text_col!r} and {label_col!r}. Found {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_spaces)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df.replace({text_col: {"": np.nan}, label_col: {"": np.nan}})
    df = df.dropna(subset=[text_col, label_col])

    # Remove exact duplicate text-label rows.
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

    # Remove ambiguous identical texts assigned to more than one dialect.
    label_counts = df.groupby(text_col)[label_col].nunique()
    ambiguous_texts = set(label_counts[label_counts > 1].index)
    if ambiguous_texts:
        df = df[~df[text_col].isin(ambiguous_texts)].reset_index(drop=True)

    return df


def make_balanced(df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    min_n = df[label_col].value_counts().min()
    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(n=min_n, random_state=seed))
          .reset_index(drop=True)
    )


def save_dataset_profile(df: pd.DataFrame, out_dir: Path, dataset_name: str, text_col: str, label_col: str) -> None:
    rows = []
    for label, part in df.groupby(label_col):
        rows.append({
            "dataset": dataset_name,
            "dialect": label,
            "rows": len(part),
            "avg_chars": float(part[text_col].map(len).mean()),
        })
    pd.DataFrame(rows).sort_values(["dataset", "rows"], ascending=[True, False]).to_csv(
        out_dir / f"{dataset_name}_data_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )


# -----------------------------
# Metrics and significance
# -----------------------------

def evaluate_predictions(y_true: List[str], y_pred: List[str], labels: List[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels),
    }


def per_class_table(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for label in labels:
        rows.append({
            "dialect": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        })
    return pd.DataFrame(rows)


def mcnemar_test(y_true: List[str], pred_a: List[str], pred_b: List[str]) -> dict:
    """
    McNemar test over paired correctness.

    A = raw->raw prediction.
    B = raw->aggressive prediction.

    b = A correct, B wrong
    c = A wrong, B correct

    Exact test uses binomial test over discordant pairs.
    Chi-square with continuity correction is also reported.
    """
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)

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
        "b_raw_correct_aggr_wrong": b,
        "c_raw_wrong_aggr_correct": c,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": exact_p,
        "mcnemar_chi2_cc": chi2_stat,
        "mcnemar_chi2_p": chi2_p,
    }


def paired_bootstrap_macro_f1(
    y_true: List[str],
    pred_a: List[str],
    pred_b: List[str],
    labels: List[str],
    n_boot: int = 2000,
    seed: int = 123,
) -> dict:
    """
    Optional but useful: paired bootstrap CI for macro-F1 difference.

    Difference = macro_f1(A) - macro_f1(B).
    A is usually raw->raw, B is raw->aggressive.
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    n = len(y_true)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro", labels=labels)
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro", labels=labels)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    return {
        "bootstrap_diff_mean": float(np.mean(diffs)),
        "bootstrap_diff_ci_low": float(np.percentile(diffs, 2.5)),
        "bootstrap_diff_ci_high": float(np.percentile(diffs, 97.5)),
    }


# -----------------------------
# Main experiment
# -----------------------------

def run_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(df[args.label_col].unique().tolist())
    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")

    normalizers_to_run = [
        "raw",
        "light",
        "punct_removed",
        "aggressive",
        "glyph_fold_only",
        "aggressive_glyph_fold",
    ]

    all_result_rows = []
    per_class_rows = []
    mcnemar_rows = []
    split_rows = []

    for seed in seeds:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=seed,
            stratify=df[args.label_col],
        )

        split_rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "test_size_ratio": args.test_size,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "n_labels": len(labels),
        })

        y_train = train_df[args.label_col].tolist()
        y_test = test_df[args.label_col].tolist()

        train_texts = {
            n: train_df[args.text_col].map(NORMALIZERS[n]).tolist()
            for n in normalizers_to_run
        }
        test_texts = {
            n: test_df[args.text_col].map(NORMALIZERS[n]).tolist()
            for n in normalizers_to_run
        }

        for model_name in models:
            print(f"[{dataset_name}] seed={seed} model={model_name}")

            # Fit models for each train-side normalizer needed.
            fitted = {}
            for train_norm in ["raw", "light", "punct_removed", "aggressive"]:
                model = build_model(model_name, args.max_word_features, args.max_char_features)
                model.fit(train_texts[train_norm], y_train)
                fitted[train_norm] = model

            # Conditions to report formally.
            conditions = [
                ("raw", "raw"),
                ("light", "light"),
                ("punct_removed", "punct_removed"),
                ("aggressive", "aggressive"),
                ("raw", "light"),
                ("raw", "punct_removed"),
                ("raw", "aggressive"),
                ("aggressive", "raw"),
                ("raw", "glyph_fold_only"),
                ("raw", "aggressive_glyph_fold"),
            ]

            predictions_for_sig = {}

            for train_norm, test_norm in conditions:
                model = fitted[train_norm]
                pred = model.predict(test_texts[test_norm])
                predictions_for_sig[(train_norm, test_norm)] = pred

                metrics = evaluate_predictions(y_test, pred, labels)
                all_result_rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "model": model_name,
                    "train_norm": train_norm,
                    "test_norm": test_norm,
                    "mode": "same_pipeline" if train_norm == test_norm else "mismatch",
                    **metrics,
                })

                pc = per_class_table(y_test, pred, labels)
                pc.insert(0, "dataset", dataset_name)
                pc.insert(1, "seed", seed)
                pc.insert(2, "model", model_name)
                pc.insert(3, "train_norm", train_norm)
                pc.insert(4, "test_norm", test_norm)
                pc.insert(5, "mode", "same_pipeline" if train_norm == test_norm else "mismatch")
                per_class_rows.append(pc)

                # Save confusion matrices for the main conditions.
                if train_norm == "raw" and test_norm in {"raw", "aggressive"}:
                    cm = confusion_matrix(y_test, pred, labels=labels)
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"true_{x}" for x in labels],
                        columns=[f"pred_{x}" for x in labels],
                    )
                    cm_df.to_csv(
                        dataset_dir / f"confusion__model-{model_name}__train-{train_norm}__test-{test_norm}__seed-{seed}.csv",
                        encoding="utf-8-sig",
                    )

            # McNemar and bootstrap compare paired predictions:
            # raw->raw vs raw->aggressive.
            pred_raw = predictions_for_sig[("raw", "raw")]
            pred_aggr = predictions_for_sig[("raw", "aggressive")]

            sig = mcnemar_test(y_test, pred_raw, pred_aggr)
            boot = paired_bootstrap_macro_f1(
                y_true=y_test,
                pred_a=pred_raw,
                pred_b=pred_aggr,
                labels=labels,
                n_boot=args.bootstrap_samples,
                seed=seed,
            )

            raw_metrics = evaluate_predictions(y_test, pred_raw, labels)
            aggr_metrics = evaluate_predictions(y_test, pred_aggr, labels)

            mcnemar_rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "model": model_name,
                "comparison": "raw->raw vs raw->aggressive",
                "raw_accuracy": raw_metrics["accuracy"],
                "aggressive_accuracy": aggr_metrics["accuracy"],
                "accuracy_drop": raw_metrics["accuracy"] - aggr_metrics["accuracy"],
                "raw_macro_f1": raw_metrics["macro_f1"],
                "aggressive_macro_f1": aggr_metrics["macro_f1"],
                "macro_f1_drop": raw_metrics["macro_f1"] - aggr_metrics["macro_f1"],
                **sig,
                **boot,
            })

    all_results = pd.DataFrame(all_result_rows)
    per_class_results = pd.concat(per_class_rows, ignore_index=True)
    mcnemar_results = pd.DataFrame(mcnemar_rows)
    split_report = pd.DataFrame(split_rows)

    all_results.to_csv(dataset_dir / "all_results.csv", index=False, encoding="utf-8-sig")
    per_class_results.to_csv(dataset_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    mcnemar_results.to_csv(dataset_dir / "mcnemar_and_bootstrap.csv", index=False, encoding="utf-8-sig")
    split_report.to_csv(dataset_dir / "split_report.csv", index=False, encoding="utf-8-sig")

    summary = (
        all_results
        .groupby(["dataset", "model", "train_norm", "test_norm", "mode"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
        )
    )
    summary.to_csv(dataset_dir / "summary_with_std.csv", index=False, encoding="utf-8-sig")

    return all_results, summary, mcnemar_results


def make_full_vs_balanced_table(all_summaries: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Formal table for the paper:
    full vs balanced raw->raw, raw->aggressive, and drop.
    """
    rows = []
    for dataset in ["full", "balanced"]:
        for model in sorted(all_summaries["model"].unique()):
            base = all_summaries[
                (all_summaries["dataset"] == dataset) &
                (all_summaries["model"] == model) &
                (all_summaries["train_norm"] == "raw") &
                (all_summaries["test_norm"] == "raw")
            ]
            aggr = all_summaries[
                (all_summaries["dataset"] == dataset) &
                (all_summaries["model"] == model) &
                (all_summaries["train_norm"] == "raw") &
                (all_summaries["test_norm"] == "aggressive")
            ]
            if base.empty or aggr.empty:
                continue

            base = base.iloc[0]
            aggr = aggr.iloc[0]
            rows.append({
                "dataset": dataset,
                "model": model,
                "raw_raw_macro_f1_mean": base["macro_f1_mean"],
                "raw_raw_macro_f1_std": base["macro_f1_std"],
                "raw_aggressive_macro_f1_mean": aggr["macro_f1_mean"],
                "raw_aggressive_macro_f1_std": aggr["macro_f1_std"],
                "drop_mean": base["macro_f1_mean"] - aggr["macro_f1_mean"],
            })

    table = pd.DataFrame(rows).sort_values(["model", "dataset"])
    table.to_csv(out_dir / "full_vs_balanced_comparison.csv", index=False, encoding="utf-8-sig")
    return table


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data), args.text_col, args.label_col)

    full_df = df.copy()
    balanced_df = make_balanced(df, args.label_col, seed=args.balance_seed)

    save_dataset_profile(full_df, out_dir, "full", args.text_col, args.label_col)
    save_dataset_profile(balanced_df, out_dir, "balanced", args.text_col, args.label_col)

    metadata = {
        "input_rows_after_cleaning": len(df),
        "test_size_ratio": args.test_size,
        "train_size_ratio": 1 - args.test_size,
        "seeds": args.seeds,
        "models": args.models,
        "balanced_rows_per_class": int(balanced_df[args.label_col].value_counts().min()),
        "balanced_total_rows": int(len(balanced_df)),
    }
    with open(out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    all_results_list = []
    summary_list = []
    sig_list = []

    for dataset_name, dataset_df in [("full", full_df), ("balanced", balanced_df)]:
        all_results, summary, sig = run_dataset(dataset_df, dataset_name, out_dir, args)
        all_results_list.append(all_results)
        summary_list.append(summary)
        sig_list.append(sig)

    combined_all = pd.concat(all_results_list, ignore_index=True)
    combined_summary = pd.concat(summary_list, ignore_index=True)
    combined_sig = pd.concat(sig_list, ignore_index=True)

    combined_all.to_csv(out_dir / "combined_all_results.csv", index=False, encoding="utf-8-sig")
    combined_summary.to_csv(out_dir / "combined_summary_with_std.csv", index=False, encoding="utf-8-sig")
    combined_sig.to_csv(out_dir / "combined_mcnemar_and_bootstrap.csv", index=False, encoding="utf-8-sig")

    comparison = make_full_vs_balanced_table(combined_summary, out_dir)

    print("\nDone.")
    print(f"Output directory: {out_dir.resolve()}")
    print("\nFull vs balanced comparison:")
    print(comparison.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV with text,dialect columns.")
    parser.add_argument("--out", default="enhanced_results")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="dialect")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--models", default="word,char,wordchar")
    parser.add_argument("--balance_seed", type=int, default=42)
    parser.add_argument("--max_word_features", type=int, default=120000)
    parser.add_argument("--max_char_features", type=int, default=200000)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    main()

