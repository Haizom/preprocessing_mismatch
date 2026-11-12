
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


# -----------------------------
# Normalization
# -----------------------------

AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
PUNCT_CHARS = string.punctuation + "،؛؟…«»“”‘’ـ"
PUNCT_RE = re.compile(rf"[{re.escape(PUNCT_CHARS)}]+")
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


def count_punct_groups(text: str) -> int:
    return len(PUNCT_RE.findall(str(text)))


def has_punct(text: str) -> bool:
    return count_punct_groups(text) > 0


# -----------------------------
# Data
# -----------------------------

def load_data(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns {text_col!r} and {label_col!r}; found {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_spaces)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df.replace({text_col: {"": np.nan}, label_col: {"": np.nan}})
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

    # Remove exact identical texts with multiple labels.
    nlabels = df.groupby(text_col)[label_col].nunique()
    ambiguous = set(nlabels[nlabels > 1].index)
    if ambiguous:
        df = df[~df[text_col].isin(ambiguous)].reset_index(drop=True)

    return df


def make_balanced(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    min_n = df[label_col].value_counts().min()
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda part: part.sample(n=min_n, random_state=seed))
        .reset_index(drop=True)
    )


# -----------------------------
# Model
# -----------------------------

def build_wordchar_model(max_word_features: int, max_char_features: int) -> Pipeline:
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
    clf = LinearSVC(C=1.0, class_weight="balanced", random_state=0)
    return Pipeline([("features", features), ("clf", clf)])


def get_feature_names(pipe: Pipeline) -> np.ndarray:
    return pipe.named_steps["features"].get_feature_names_out()


# -----------------------------
# Feature-change utilities
# -----------------------------

def strip_feature_prefix(feature_name: str) -> tuple[str, str]:
    feature_name = str(feature_name)
    if feature_name.startswith("word__"):
        return "word", feature_name.replace("word__", "", 1)
    if feature_name.startswith("char__"):
        return "char", feature_name.replace("char__", "", 1)
    return "unknown", feature_name


def classify_feature_change(feature: str) -> tuple[str, str]:
    norm = aggressive(feature)

    if norm == feature:
        return norm, "unchanged"

    if norm == "":
        return norm, "removed_empty"

    if PUNCT_RE.search(feature) and not PUNCT_RE.search(norm):
        return norm, "punctuation_removed_or_changed"

    if REPEAT_RE.search(feature):
        return norm, "repeat_collapsed_or_changed"

    if any(ch in feature for ch in ["ة", "ئ", "ؤ", "أ", "إ", "آ", "ٱ", "ى"]):
        return norm, "arabic_letter_normalized"

    return norm, "other_changed"


def rank_weight(rank: int) -> float:
    # Heuristic rank weight: top-ranked features count more.
    return 1.0 / rank


# -----------------------------
# Metric helpers
# -----------------------------

def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for lab in labels:
        rows.append({
            "dialect": lab,
            "precision": rep[lab]["precision"],
            "recall": rep[lab]["recall"],
            "f1": rep[lab]["f1-score"],
            "support": rep[lab]["support"],
        })
    return pd.DataFrame(rows)


def decision_margin(pipe: Pipeline, X: List[str]) -> np.ndarray:
    scores = pipe.decision_function(X)
    if scores.ndim == 1:
        # Binary fallback.
        return np.abs(scores)
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


# -----------------------------
# Main diagnostics per seed
# -----------------------------

def run_one_seed(
    df: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    labels = sorted(df[args.label_col].unique().tolist())

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=seed,
        stratify=df[args.label_col],
    )

    y_train = train_df[args.label_col].to_numpy()
    y_test = test_df[args.label_col].to_numpy()

    X_train_raw = train_df[args.text_col].map(raw).tolist()
    X_train_aggr = train_df[args.text_col].map(aggressive).tolist()

    X_test_raw = test_df[args.text_col].map(raw).tolist()
    X_test_aggr = test_df[args.text_col].map(aggressive).tolist()
    punct_mask = test_df[args.text_col].map(has_punct).to_numpy()
    punct_counts = test_df[args.text_col].map(count_punct_groups).to_numpy()

    raw_model = build_wordchar_model(args.max_word_features, args.max_char_features)
    raw_model.fit(X_train_raw, y_train)

    aggr_model = build_wordchar_model(args.max_word_features, args.max_char_features)
    aggr_model.fit(X_train_aggr, y_train)

    pred_raw_raw = raw_model.predict(X_test_raw)
    pred_raw_aggr = raw_model.predict(X_test_aggr)
    pred_aggr_aggr = aggr_model.predict(X_test_aggr)

    margin_raw_raw = decision_margin(raw_model, X_test_raw)
    margin_raw_aggr = decision_margin(raw_model, X_test_aggr)
    margin_aggr_aggr = decision_margin(aggr_model, X_test_aggr)

    # Confusion rows for raw->raw and raw->aggressive.
    cm_raw = confusion_matrix(y_test, pred_raw_raw, labels=labels)
    cm_aggr = confusion_matrix(y_test, pred_raw_aggr, labels=labels)

    confusion_rows = []
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            confusion_rows.append({
                "seed": seed,
                "true_dialect": true_lab,
                "predicted_as": pred_lab,
                "raw_count": int(cm_raw[i, j]),
                "aggressive_count": int(cm_aggr[i, j]),
                "delta_aggressive_minus_raw": int(cm_aggr[i, j] - cm_raw[i, j]),
                "raw_rate": float(cm_raw[i, j] / cm_raw[i].sum()) if cm_raw[i].sum() else 0.0,
                "aggressive_rate": float(cm_aggr[i, j] / cm_aggr[i].sum()) if cm_aggr[i].sum() else 0.0,
                "delta_rate": float(cm_aggr[i, j] / cm_aggr[i].sum() - cm_raw[i, j] / cm_raw[i].sum()) if cm_raw[i].sum() and cm_aggr[i].sum() else 0.0,
            })

    # Per-dialect F1 under raw/raw, raw/aggr, aggr/aggr.
    f1_raw = per_class_f1(y_test, pred_raw_raw, labels).rename(columns={"f1": "raw_raw_f1"})
    f1_aggr = per_class_f1(y_test, pred_raw_aggr, labels).rename(columns={"f1": "raw_aggressive_f1"})
    f1_aggr_matched = per_class_f1(y_test, pred_aggr_aggr, labels).rename(columns={"f1": "aggressive_aggressive_f1"})

    f1_merge = (
        f1_raw[["dialect", "raw_raw_f1", "support"]]
        .merge(f1_aggr[["dialect", "raw_aggressive_f1"]], on="dialect")
        .merge(f1_aggr_matched[["dialect", "aggressive_aggressive_f1"]], on="dialect")
    )
    f1_merge["seed"] = seed
    f1_merge["raw_to_aggressive_drop"] = f1_merge["raw_raw_f1"] - f1_merge["raw_aggressive_f1"]
    f1_merge["matched_aggressive_recovery"] = f1_merge["aggressive_aggressive_f1"] - f1_merge["raw_aggressive_f1"]

    # Survival and margin diagnostics by dialect.
    raw_correct = pred_raw_raw == y_test
    aggr_correct = pred_raw_aggr == y_test
    harmed = raw_correct & ~aggr_correct
    helped = ~raw_correct & aggr_correct

    resilience_rows = []
    for lab in labels:
        mask = y_test == lab
        raw_correct_mask = mask & raw_correct
        survival = float(aggr_correct[raw_correct_mask].mean()) if raw_correct_mask.sum() else np.nan

        resilience_rows.append({
            "seed": seed,
            "dialect": lab,
            "rows": int(mask.sum()),
            "raw_accuracy": float(raw_correct[mask].mean()),
            "raw_aggressive_accuracy": float(aggr_correct[mask].mean()),
            "accuracy_drop": float(raw_correct[mask].mean() - aggr_correct[mask].mean()),
            "raw_correct_survival_under_aggressive": survival,
            "harmed_rate": float(harmed[mask].mean()),
            "helped_rate": float(helped[mask].mean()),
            "mean_margin_raw_raw": float(np.mean(margin_raw_raw[mask])),
            "mean_margin_raw_aggressive": float(np.mean(margin_raw_aggr[mask])),
            "mean_margin_aggressive_aggressive": float(np.mean(margin_aggr_aggr[mask])),
            "margin_drop_raw_model": float(np.mean(margin_raw_raw[mask]) - np.mean(margin_raw_aggr[mask])),
        })

    # Punctuation vulnerability.
    punct_rows = []
    groups = [
        ("has_punctuation", punct_mask),
        ("no_punctuation", ~punct_mask),
        ("punct_count_ge_2", punct_counts >= 2),
        ("punct_count_ge_4", punct_counts >= 4),
    ]

    for group_name, gmask in groups:
        if gmask.sum() == 0:
            continue
        punct_rows.append({
            "seed": seed,
            "dialect": "ALL",
            "group": group_name,
            "rows": int(gmask.sum()),
            "raw_accuracy": float(raw_correct[gmask].mean()),
            "raw_aggressive_accuracy": float(aggr_correct[gmask].mean()),
            "accuracy_drop": float(raw_correct[gmask].mean() - aggr_correct[gmask].mean()),
            "harmed_rate": float(harmed[gmask].mean()),
            "helped_rate": float(helped[gmask].mean()),
        })

    for lab in labels:
        dmask = y_test == lab
        for group_name, pmask in [("has_punctuation", punct_mask), ("no_punctuation", ~punct_mask)]:
            gmask = dmask & pmask
            if gmask.sum() == 0:
                continue
            punct_rows.append({
                "seed": seed,
                "dialect": lab,
                "group": group_name,
                "rows": int(gmask.sum()),
                "raw_accuracy": float(raw_correct[gmask].mean()),
                "raw_aggressive_accuracy": float(aggr_correct[gmask].mean()),
                "accuracy_drop": float(raw_correct[gmask].mean() - aggr_correct[gmask].mean()),
                "harmed_rate": float(harmed[gmask].mean()),
                "helped_rate": float(helped[gmask].mean()),
            })

    # Weighted feature analysis from raw-trained WordChar.
    feature_names = get_feature_names(raw_model)
    coefs = raw_model.named_steps["clf"].coef_

    feature_rows = []
    for class_idx, lab in enumerate(labels):
        coef = coefs[class_idx]
        top_idx = np.argsort(coef)[-args.top_k:][::-1]
        top_positive_weights = np.maximum(coef[top_idx], 0)
        total_positive_mass = float(top_positive_weights.sum())
        total_rank_mass = float(sum(rank_weight(r) for r in range(1, len(top_idx) + 1)))

        for rank, idx in enumerate(top_idx, start=1):
            raw_feature_name = str(feature_names[idx])
            feature_type, feature = strip_feature_prefix(raw_feature_name)
            norm_feature, change_type = classify_feature_change(feature)
            changed = change_type != "unchanged"
            positive_weight = float(max(coef[idx], 0))
            feature_rows.append({
                "seed": seed,
                "dialect": lab,
                "rank": rank,
                "feature_type": feature_type,
                "feature": feature,
                "weight": float(coef[idx]),
                "positive_weight": positive_weight,
                "positive_weight_share_within_topk": positive_weight / total_positive_mass if total_positive_mass > 0 else 0.0,
                "rank_weight": rank_weight(rank),
                "rank_weight_share_within_topk": rank_weight(rank) / total_rank_mass if total_rank_mass > 0 else 0.0,
                "aggressive_version": norm_feature,
                "change_type": change_type,
                "changed_by_aggressive": changed,
            })

    return {
        "confusion": pd.DataFrame(confusion_rows),
        "f1": f1_merge,
        "resilience": pd.DataFrame(resilience_rows),
        "punctuation": pd.DataFrame(punct_rows),
        "features": pd.DataFrame(feature_rows),
    }


# -----------------------------
# Aggregation
# -----------------------------

def mean_std_agg(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    agg_spec = {}
    for col in value_cols:
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_std"] = (col, "std")
    return df.groupby(group_cols, as_index=False).agg(**agg_spec)


def aggregate_and_save(results: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    confusion = pd.concat([r["confusion"] for r in results], ignore_index=True)
    f1 = pd.concat([r["f1"] for r in results], ignore_index=True)
    resilience = pd.concat([r["resilience"] for r in results], ignore_index=True)
    punctuation = pd.concat([r["punctuation"] for r in results], ignore_index=True)
    features = pd.concat([r["features"] for r in results], ignore_index=True)

    confusion.to_csv(out_dir / "confusion_shift_by_seed.csv", index=False, encoding="utf-8-sig")
    f1.to_csv(out_dir / "per_dialect_f1_by_seed.csv", index=False, encoding="utf-8-sig")
    resilience.to_csv(out_dir / "dialect_resilience_by_seed.csv", index=False, encoding="utf-8-sig")
    punctuation.to_csv(out_dir / "punctuation_vulnerability_by_seed.csv", index=False, encoding="utf-8-sig")
    features.to_csv(out_dir / "top_features_weighted_by_seed.csv", index=False, encoding="utf-8-sig")

    confusion_agg = mean_std_agg(
        confusion,
        ["true_dialect", "predicted_as"],
        ["raw_count", "aggressive_count", "delta_aggressive_minus_raw", "raw_rate", "aggressive_rate", "delta_rate"],
    )
    confusion_agg.to_csv(out_dir / "confusion_shift_averaged.csv", index=False, encoding="utf-8-sig")

    top_increased = (
        confusion_agg[confusion_agg["true_dialect"] != confusion_agg["predicted_as"]]
        .sort_values("delta_aggressive_minus_raw_mean", ascending=False)
    )
    top_increased.to_csv(out_dir / "top_increased_confusions_averaged.csv", index=False, encoding="utf-8-sig")

    focus = confusion_agg[confusion_agg["true_dialect"].isin(["Syrian", "Palestinian", "Lebanese"])]
    focus.to_csv(out_dir / "focus_confusion_shift_syrian_palestinian_lebanese.csv", index=False, encoding="utf-8-sig")

    f1_agg = mean_std_agg(
        f1,
        ["dialect"],
        ["raw_raw_f1", "raw_aggressive_f1", "aggressive_aggressive_f1", "raw_to_aggressive_drop", "matched_aggressive_recovery"],
    )
    f1_agg.to_csv(out_dir / "per_dialect_f1_averaged.csv", index=False, encoding="utf-8-sig")

    resilience_agg = mean_std_agg(
        resilience,
        ["dialect"],
        [
            "raw_accuracy", "raw_aggressive_accuracy", "accuracy_drop",
            "raw_correct_survival_under_aggressive", "harmed_rate", "helped_rate",
            "mean_margin_raw_raw", "mean_margin_raw_aggressive", "mean_margin_aggressive_aggressive",
            "margin_drop_raw_model",
        ],
    )
    resilience_agg.to_csv(out_dir / "dialect_resilience_averaged.csv", index=False, encoding="utf-8-sig")

    punctuation_agg = mean_std_agg(
        punctuation,
        ["dialect", "group"],
        ["rows", "raw_accuracy", "raw_aggressive_accuracy", "accuracy_drop", "harmed_rate", "helped_rate"],
    )
    punctuation_agg.to_csv(out_dir / "punctuation_vulnerability_averaged.csv", index=False, encoding="utf-8-sig")

    # Feature weighted aggregation by seed first, then across seeds.
    feature_summary_seed = (
        features
        .groupby(["seed", "dialect"], as_index=False)
        .agg(
            top_features=("feature", "count"),
            changed_count=("changed_by_aggressive", "sum"),
            changed_count_share=("changed_by_aggressive", "mean"),
            changed_weight_mass_share=("positive_weight_share_within_topk", lambda s: float(s[features.loc[s.index, "changed_by_aggressive"]].sum())),
            changed_rank_mass_share=("rank_weight_share_within_topk", lambda s: float(s[features.loc[s.index, "changed_by_aggressive"]].sum())),
            punctuation_related_count=("change_type", lambda s: int((s == "punctuation_removed_or_changed").sum())),
            punctuation_related_weight_mass_share=("positive_weight_share_within_topk", lambda s: float(s[features.loc[s.index, "change_type"] == "punctuation_removed_or_changed"].sum())),
            arabic_letter_related_count=("change_type", lambda s: int((s == "arabic_letter_normalized").sum())),
            arabic_letter_related_weight_mass_share=("positive_weight_share_within_topk", lambda s: float(s[features.loc[s.index, "change_type"] == "arabic_letter_normalized"].sum())),
        )
    )
    feature_summary_seed.to_csv(out_dir / "feature_change_weighted_by_seed.csv", index=False, encoding="utf-8-sig")

    feature_summary_avg = mean_std_agg(
        feature_summary_seed,
        ["dialect"],
        [
            "changed_count", "changed_count_share", "changed_weight_mass_share", "changed_rank_mass_share",
            "punctuation_related_count", "punctuation_related_weight_mass_share",
            "arabic_letter_related_count", "arabic_letter_related_weight_mass_share",
        ]
    )
    feature_summary_avg.to_csv(out_dir / "feature_change_weighted_averaged.csv", index=False, encoding="utf-8-sig")

    # Combine several diagnostics to inspect the Lebanese anomaly.
    lebanese_anomaly = (
        f1_agg
        .merge(resilience_agg, on="dialect", how="left")
        .merge(feature_summary_avg, on="dialect", how="left")
    )
    # Add punctuation has/no punctuation drops.
    p_has = punctuation_agg[punctuation_agg["group"] == "has_punctuation"][[
        "dialect", "accuracy_drop_mean", "accuracy_drop_std", "harmed_rate_mean", "harmed_rate_std"
    ]].rename(columns={
        "accuracy_drop_mean": "has_punct_accuracy_drop_mean",
        "accuracy_drop_std": "has_punct_accuracy_drop_std",
        "harmed_rate_mean": "has_punct_harmed_rate_mean",
        "harmed_rate_std": "has_punct_harmed_rate_std",
    })
    p_no = punctuation_agg[punctuation_agg["group"] == "no_punctuation"][[
        "dialect", "accuracy_drop_mean", "accuracy_drop_std", "harmed_rate_mean", "harmed_rate_std"
    ]].rename(columns={
        "accuracy_drop_mean": "no_punct_accuracy_drop_mean",
        "accuracy_drop_std": "no_punct_accuracy_drop_std",
        "harmed_rate_mean": "no_punct_harmed_rate_mean",
        "harmed_rate_std": "no_punct_harmed_rate_std",
    })
    lebanese_anomaly = lebanese_anomaly.merge(p_has, on="dialect", how="left").merge(p_no, on="dialect", how="left")
    lebanese_anomaly.to_csv(out_dir / "lebanese_anomaly_diagnostics.csv", index=False, encoding="utf-8-sig")


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_df = load_data(Path(args.data), args.text_col, args.label_col)

    if args.dataset == "balanced":
        # Important: balance once using balance_seed, then split with experiment seeds.
        df = make_balanced(original_df, args.label_col, args.balance_seed)
    elif args.dataset == "full":
        df = original_df.copy()
    else:
        raise ValueError("--dataset must be full or balanced")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    metadata = {
        "dataset_mode": args.dataset,
        "rows_after_cleaning_original": len(original_df),
        "rows_used": len(df),
        "labels": sorted(df[args.label_col].unique().tolist()),
        "label_counts_used": df[args.label_col].value_counts().to_dict(),
        "seeds": seeds,
        "test_size": args.test_size,
        "model": "wordchar",
        "top_k": args.top_k,
    }
    with open(out_dir / "revision_diagnostics_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    results = []
    for seed in seeds:
        print(f"Running seed {seed} on {args.dataset} dataset...")
        results.append(run_one_seed(df, seed, args, out_dir))

    aggregate_and_save(results, out_dir)

    print("\nDone.")
    print(f"Results saved to: {out_dir.resolve()}")
    print("\nMost important files:")
    print(" - top_increased_confusions_averaged.csv")
    print(" - focus_confusion_shift_syrian_palestinian_lebanese.csv")
    print(" - punctuation_vulnerability_averaged.csv")
    print(" - feature_change_weighted_averaged.csv")
    print(" - lebanese_anomaly_diagnostics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="revision_diagnostics_results")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="dialect")
    parser.add_argument("--dataset", choices=["balanced", "full"], default="balanced")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--balance_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_word_features", type=int, default=120000)
    parser.add_argument("--max_char_features", type=int, default=200000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
