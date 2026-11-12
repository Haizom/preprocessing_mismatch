
import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
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


def count_punct(text: str) -> int:
    return len(PUNCT_RE.findall(str(text)))


def punct_chars(text: str) -> List[str]:
    return re.findall(rf"[{re.escape(PUNCT_CHARS)}]", str(text))


# -----------------------------
# Data
# -----------------------------

def load_data(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns {text_col!r} and {label_col!r}. Found: {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_spaces)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df.replace({text_col: {"": np.nan}, label_col: {"": np.nan}})
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

    # Remove ambiguous duplicate texts with multiple labels.
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
# Confusion analysis
# -----------------------------

def clean_cm_labels(labels: List[str]) -> List[str]:
    cleaned = []
    for label in labels:
        label = str(label)
        label = re.sub(r"^(true|pred)_", "", label)
        cleaned.append(label)
    return cleaned


def read_confusion(path: Path) -> pd.DataFrame:
    cm = pd.read_csv(path, index_col=0)
    cm.index = clean_cm_labels(list(cm.index))
    cm.columns = clean_cm_labels(list(cm.columns))
    return cm


def confusion_shift_analysis(raw_cm: pd.DataFrame, aggr_cm: pd.DataFrame, out_dir: Path) -> None:
    labels = list(raw_cm.index)
    aggr_cm = aggr_cm.loc[labels, labels]

    delta = aggr_cm - raw_cm
    delta.to_csv(out_dir / "confusion_delta_aggressive_minus_raw.csv", encoding="utf-8-sig")

    correct_rows = []
    for lab in labels:
        raw_correct = int(raw_cm.loc[lab, lab])
        aggr_correct = int(aggr_cm.loc[lab, lab])
        support = int(raw_cm.loc[lab].sum())
        correct_rows.append({
            "dialect": lab,
            "support": support,
            "raw_correct": raw_correct,
            "aggressive_correct": aggr_correct,
            "correct_drop_count": raw_correct - aggr_correct,
            "raw_recall": raw_correct / support if support else 0,
            "aggressive_recall": aggr_correct / support if support else 0,
            "recall_drop": (raw_correct - aggr_correct) / support if support else 0,
        })
    pd.DataFrame(correct_rows).sort_values("correct_drop_count", ascending=False).to_csv(
        out_dir / "confusion_correct_drop_by_dialect.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Top increased off-diagonal confusions.
    rows = []
    for true_lab in labels:
        for pred_lab in labels:
            if true_lab == pred_lab:
                continue
            raw_val = int(raw_cm.loc[true_lab, pred_lab])
            aggr_val = int(aggr_cm.loc[true_lab, pred_lab])
            rows.append({
                "true_dialect": true_lab,
                "predicted_as": pred_lab,
                "raw_count": raw_val,
                "aggressive_count": aggr_val,
                "increase": aggr_val - raw_val,
            })
    pd.DataFrame(rows).sort_values("increase", ascending=False).to_csv(
        out_dir / "top_increased_confusions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Syrian-focused table if label exists.
    if "Syrian" in labels:
        syrian = []
        for pred_lab in labels:
            syrian.append({
                "true_dialect": "Syrian",
                "predicted_as": pred_lab,
                "raw_count": int(raw_cm.loc["Syrian", pred_lab]),
                "aggressive_count": int(aggr_cm.loc["Syrian", pred_lab]),
                "change": int(aggr_cm.loc["Syrian", pred_lab] - raw_cm.loc["Syrian", pred_lab]),
            })
        pd.DataFrame(syrian).sort_values("change", ascending=False).to_csv(
            out_dir / "syrian_confusion_shift.csv",
            index=False,
            encoding="utf-8-sig",
        )


# -----------------------------
# Punctuation analysis
# -----------------------------

def punctuation_profile(df: pd.DataFrame, out_dir: Path, text_col: str, label_col: str) -> None:
    rows = []
    inv_rows = []

    for dialect, part in df.groupby(label_col):
        punct_counts = part[text_col].map(count_punct)
        has_punct = punct_counts > 0
        chars = []
        for t in part[text_col]:
            chars.extend(punct_chars(t))
        char_counter = Counter(chars)

        rows.append({
            "dialect": dialect,
            "rows": len(part),
            "rows_with_punctuation": int(has_punct.sum()),
            "pct_rows_with_punctuation": float(has_punct.mean()),
            "mean_punct_groups_per_row": float(punct_counts.mean()),
            "median_punct_groups_per_row": float(punct_counts.median()),
            "max_punct_groups_in_row": int(punct_counts.max()),
            "top_punctuation_chars": " ".join([f"{k}:{v}" for k, v in char_counter.most_common(10)]),
        })

        for ch, cnt in char_counter.most_common():
            inv_rows.append({
                "dialect": dialect,
                "punct_char": ch,
                "count": cnt,
                "count_per_1000_rows": cnt / len(part) * 1000,
            })

    pd.DataFrame(rows).sort_values("pct_rows_with_punctuation", ascending=False).to_csv(
        out_dir / "punctuation_profile_by_dialect.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(inv_rows).to_csv(
        out_dir / "punctuation_inventory_by_dialect.csv",
        index=False,
        encoding="utf-8-sig",
    )


def punctuation_vulnerability_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    text_col: str,
    label_col: str,
    seed: int,
    test_size: float,
    balanced: bool,
    max_word_features: int,
    max_char_features: int,
) -> None:
    """
    Tests whether rows containing punctuation are more likely to flip from correct under raw->raw
    to wrong under raw->aggressive.
    """
    work_df = make_balanced(df, label_col, seed=seed) if balanced else df.copy()
    labels = sorted(work_df[label_col].unique().tolist())

    train_df, test_df = train_test_split(
        work_df,
        test_size=test_size,
        random_state=seed,
        stratify=work_df[label_col],
    )

    y_train = train_df[label_col].tolist()
    y_test = np.array(test_df[label_col].tolist())

    model = build_wordchar_model(max_word_features, max_char_features)
    model.fit(train_df[text_col].map(raw).tolist(), y_train)

    pred_raw = np.array(model.predict(test_df[text_col].map(raw).tolist()))
    pred_aggr = np.array(model.predict(test_df[text_col].map(aggressive).tolist()))

    raw_correct = pred_raw == y_test
    aggr_correct = pred_aggr == y_test
    harmed = raw_correct & ~aggr_correct
    helped = ~raw_correct & aggr_correct

    punct_n = test_df[text_col].map(count_punct).to_numpy()
    has_punct = punct_n > 0

    rows = []
    for group_name, mask in [
        ("has_punctuation", has_punct),
        ("no_punctuation", ~has_punct),
        ("punct_count_ge_2", punct_n >= 2),
        ("punct_count_ge_4", punct_n >= 4),
    ]:
        if mask.sum() == 0:
            continue
        rows.append({
            "dataset": "balanced" if balanced else "full",
            "group": group_name,
            "rows": int(mask.sum()),
            "raw_accuracy": float((pred_raw[mask] == y_test[mask]).mean()),
            "aggressive_accuracy": float((pred_aggr[mask] == y_test[mask]).mean()),
            "accuracy_drop": float((pred_raw[mask] == y_test[mask]).mean() - (pred_aggr[mask] == y_test[mask]).mean()),
            "pct_harmed_raw_correct_aggr_wrong": float(harmed[mask].mean()),
            "pct_helped_raw_wrong_aggr_correct": float(helped[mask].mean()),
        })

    # By dialect and punctuation presence.
    for dialect in labels:
        dmask = test_df[label_col].to_numpy() == dialect
        for group_name, pmask in [("has_punctuation", has_punct), ("no_punctuation", ~has_punct)]:
            mask = dmask & pmask
            if mask.sum() == 0:
                continue
            rows.append({
                "dataset": "balanced" if balanced else "full",
                "group": f"{dialect}__{group_name}",
                "rows": int(mask.sum()),
                "raw_accuracy": float((pred_raw[mask] == y_test[mask]).mean()),
                "aggressive_accuracy": float((pred_aggr[mask] == y_test[mask]).mean()),
                "accuracy_drop": float((pred_raw[mask] == y_test[mask]).mean() - (pred_aggr[mask] == y_test[mask]).mean()),
                "pct_harmed_raw_correct_aggr_wrong": float(harmed[mask].mean()),
                "pct_helped_raw_wrong_aggr_correct": float(helped[mask].mean()),
            })

    pd.DataFrame(rows).to_csv(
        out_dir / ("punctuation_vulnerability_balanced.csv" if balanced else "punctuation_vulnerability_full.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Save harmed examples for qualitative inspection.
    examples = test_df.copy()
    examples["pred_raw"] = pred_raw
    examples["pred_aggressive"] = pred_aggr
    examples["raw_correct"] = raw_correct
    examples["aggressive_correct"] = aggr_correct
    examples["harmed_raw_correct_aggr_wrong"] = harmed
    examples["punct_count"] = punct_n
    examples["aggressive_text"] = examples[text_col].map(aggressive)
    examples[examples["harmed_raw_correct_aggr_wrong"]].sort_values("punct_count", ascending=False).head(200).to_csv(
        out_dir / ("harmed_examples_balanced_top200.csv" if balanced else "harmed_examples_full_top200.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# -----------------------------
# Feature analysis
# -----------------------------

def classify_feature_change(feature: str) -> Tuple[str, str]:
    """
    Return (normalized_feature, change_type).

    This is heuristic because feature strings are n-grams, not full texts.
    Still useful for summarizing which raw weighted features are affected.
    """
    norm = aggressive(feature)

    if norm == feature:
        return norm, "unchanged"

    if norm == "":
        return norm, "removed_empty"

    if PUNCT_RE.search(feature) and not PUNCT_RE.search(norm):
        return norm, "punctuation_removed_or_changed"

    if REPEAT_RE.search(feature):
        return norm, "repeat_collapsed_or_changed"

    # Specific Arabic mappings.
    if any(ch in feature for ch in ["ة", "ئ", "ؤ", "أ", "إ", "آ", "ٱ", "ى"]):
        return norm, "arabic_letter_normalized"

    return norm, "other_changed"


def feature_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    text_col: str,
    label_col: str,
    seed: int,
    test_size: float,
    balanced: bool,
    top_k: int,
    max_word_features: int,
    max_char_features: int,
) -> None:
    work_df = make_balanced(df, label_col, seed=seed) if balanced else df.copy()
    labels = sorted(work_df[label_col].unique().tolist())

    train_df, test_df = train_test_split(
        work_df,
        test_size=test_size,
        random_state=seed,
        stratify=work_df[label_col],
    )

    model = build_wordchar_model(max_word_features, max_char_features)
    model.fit(train_df[text_col].map(raw).tolist(), train_df[label_col].tolist())

    feature_names = model.named_steps["features"].get_feature_names_out()
    clf = model.named_steps["clf"]

    rows = []
    for class_idx, dialect in enumerate(labels):
        coefs = clf.coef_[class_idx]
        top_idx = np.argsort(coefs)[-top_k:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            feat = str(feature_names[idx])
            # FeatureUnion names appear as "word__..." and "char__..."
            if feat.startswith("word__"):
                feat_type = "word"
                clean_feat = feat.replace("word__", "", 1)
            elif feat.startswith("char__"):
                feat_type = "char"
                clean_feat = feat.replace("char__", "", 1)
            else:
                feat_type = "unknown"
                clean_feat = feat

            norm_feat, change_type = classify_feature_change(clean_feat)
            rows.append({
                "dataset": "balanced" if balanced else "full",
                "dialect": dialect,
                "rank": rank,
                "feature_type": feat_type,
                "feature": clean_feat,
                "weight": float(coefs[idx]),
                "aggressive_version": norm_feat,
                "change_type": change_type,
                "changed_by_aggressive": change_type != "unchanged",
            })

    top_features = pd.DataFrame(rows)
    top_features.to_csv(
        out_dir / ("top_raw_features_aggressive_change_balanced.csv" if balanced else "top_raw_features_aggressive_change_full.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        top_features
        .groupby(["dataset", "dialect", "feature_type", "change_type"], as_index=False)
        .agg(
            n_features=("feature", "count"),
            mean_weight=("weight", "mean"),
        )
        .sort_values(["dataset", "dialect", "feature_type", "n_features"], ascending=[True, True, True, False])
    )
    summary.to_csv(
        out_dir / ("feature_change_summary_balanced.csv" if balanced else "feature_change_summary_full.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Compact dialect-level percentage changed.
    pct = (
        top_features
        .groupby(["dataset", "dialect"], as_index=False)
        .agg(
            top_features=("feature", "count"),
            changed=("changed_by_aggressive", "sum"),
            pct_changed=("changed_by_aggressive", "mean"),
            punct_related=("change_type", lambda s: int((s == "punctuation_removed_or_changed").sum())),
            arabic_letter_related=("change_type", lambda s: int((s == "arabic_letter_normalized").sum())),
        )
        .sort_values("pct_changed", ascending=False)
    )
    pct.to_csv(
        out_dir / ("feature_loss_by_dialect_balanced.csv" if balanced else "feature_loss_by_dialect_full.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data), args.text_col, args.label_col)

    # Confusion analysis if matrices are provided.
    if args.raw_cm and args.aggr_cm:
        raw_cm = read_confusion(Path(args.raw_cm))
        aggr_cm = read_confusion(Path(args.aggr_cm))
        confusion_shift_analysis(raw_cm, aggr_cm, out_dir)

    # Punctuation profile for full and balanced data.
    punctuation_profile(df, out_dir / "full_punctuation", args.text_col, args.label_col) if False else None

    full_punct_dir = out_dir / "full"
    bal_punct_dir = out_dir / "balanced"
    full_punct_dir.mkdir(exist_ok=True)
    bal_punct_dir.mkdir(exist_ok=True)

    punctuation_profile(df, full_punct_dir, args.text_col, args.label_col)
    balanced_df = make_balanced(df, args.label_col, seed=args.balance_seed)
    punctuation_profile(balanced_df, bal_punct_dir, args.text_col, args.label_col)

    punctuation_vulnerability_analysis(
        df=df,
        out_dir=full_punct_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        seed=args.seed,
        test_size=args.test_size,
        balanced=False,
        max_word_features=args.max_word_features,
        max_char_features=args.max_char_features,
    )
    punctuation_vulnerability_analysis(
        df=df,
        out_dir=bal_punct_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        seed=args.seed,
        test_size=args.test_size,
        balanced=True,
        max_word_features=args.max_word_features,
        max_char_features=args.max_char_features,
    )

    feature_analysis(
        df=df,
        out_dir=full_punct_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        seed=args.seed,
        test_size=args.test_size,
        balanced=False,
        top_k=args.top_k,
        max_word_features=args.max_word_features,
        max_char_features=args.max_char_features,
    )
    feature_analysis(
        df=df,
        out_dir=bal_punct_dir,
        text_col=args.text_col,
        label_col=args.label_col,
        seed=args.seed,
        test_size=args.test_size,
        balanced=True,
        top_k=args.top_k,
        max_word_features=args.max_word_features,
        max_char_features=args.max_char_features,
    )

    metadata = {
        "data": str(args.data),
        "rows_after_cleaning": len(df),
        "seed": args.seed,
        "balance_seed": args.balance_seed,
        "test_size": args.test_size,
        "top_k_features": args.top_k,
        "notes": [
            "Feature-change analysis is heuristic: it checks whether top raw features are altered by aggressive normalization.",
            "Punctuation vulnerability is run for seed only; repeat with multiple seeds if you want paper-level statistics.",
        ],
    }
    with open(out_dir / "deep_analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to {out_dir.resolve()}")
    print("Important files:")
    print(" - confusion_delta_aggressive_minus_raw.csv")
    print(" - syrian_confusion_shift.csv")
    print(" - top_increased_confusions.csv")
    print(" - balanced/punctuation_profile_by_dialect.csv")
    print(" - balanced/punctuation_vulnerability_balanced.csv")
    print(" - balanced/feature_loss_by_dialect_balanced.csv")
    print(" - balanced/top_raw_features_aggressive_change_balanced.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="deep_analysis_results")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="dialect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_word_features", type=int, default=120000)
    parser.add_argument("--max_char_features", type=int, default=200000)
    parser.add_argument("--raw_cm", default=None, help="Path to raw->raw confusion matrix CSV.")
    parser.add_argument("--aggr_cm", default=None, help="Path to raw->aggressive confusion matrix CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

