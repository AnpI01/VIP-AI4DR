#!/usr/bin/env python3
"""
QProp: Decision Tree + Random Forest tuning + balanced vs unbalanced experiment
Also prints the ROOT NODE split for interpretability.

Run:
  python baselines\\qprop_tree_rf_tune.py --data qprop_data\\proppy_1.0.train.tsv --mode balanced
  python baselines\\qprop_tree_rf_tune.py --data qprop_data\\proppy_1.0.train.tsv --mode unbalanced
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def clean_text(s: str) -> str:
    # Simple clear preprocessing
    s = str(s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def make_balanced(df: pd.DataFrame, label_col: int, n_each: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    parts = []
    for cls in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cls]
        parts.append(sub.sample(n=min(n_each, len(sub)), random_state=rng))
    return pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)


def print_root_split(model: DecisionTreeClassifier, vectorizer: CountVectorizer) -> None:
    # For sklearn trees, root is node 0
    tree = model.tree_
    feat_id = tree.feature[0]
    thr = tree.threshold[0]

    if feat_id == -2:
        print("Root is a leaf (no split).")
        return

    feat_name = vectorizer.get_feature_names_out()[feat_id]
    print(f"ROOT SPLIT: if count('{feat_name}') <= {thr:.3f} go LEFT else RIGHT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="qprop_data/proppy_1.0.train.tsv",
                    help="Path to local QProp TSV (not committed to git)")
    ap.add_argument("--mode", choices=["balanced", "unbalanced"], default="balanced")
    ap.add_argument("--max_vocab", type=int, default=2000)
    ap.add_argument("--balanced_n_each", type=int, default=4021, help="4021 gives 8042 total like last week")
    args = ap.parse_args()

    p = Path(args.data)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Put the QProp TSV in qprop_data/ (local only; not committed), or pass --data path/to/file.tsv"
        )

    #QProp TSV has NO header row -> use header=None
    df = pd.read_csv(p, sep="\t", header=None, low_memory=False, on_bad_lines="skip")

    text_col = 0
    label_col = df.columns[-1]

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(clean_text)

    # Convert -1/1 -> 0/1 if needed
    uniq = set(df[label_col].unique())
    if uniq == {-1, 1}:
        df[label_col] = (df[label_col] == 1).astype(int)

    # Balanced vs unbalanced
    if args.mode == "balanced":
        df = make_balanced(df, label_col, n_each=args.balanced_n_each)
        print(f"MODE=balanced  total={len(df)}  (approx 50/50)")
    else:
        print(f"MODE=unbalanced total={len(df)}  (natural distribution)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col],
        test_size=0.2, random_state=42,
        stratify=df[label_col]
    )

    vec = CountVectorizer(max_features=args.max_vocab, stop_words="english")
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    # ---- Decision Tree tuning ----
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 10, 50],
        "min_samples_leaf": [1, 5, 20],
    }
    dt_search = GridSearchCV(dt, dt_grid, scoring="f1_macro", cv=3, n_jobs=-1)
    dt_search.fit(Xtr, y_train)
    best_dt = dt_search.best_estimator_

    # ---- Random Forest tuning ----
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 20, 40],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 5, 20],
    }
    rf_search = GridSearchCV(rf, rf_grid, scoring="f1_macro", cv=3, n_jobs=-1)
    rf_search.fit(Xtr, y_train)
    best_rf = rf_search.best_estimator_

    # Evaluate
    for name, model in [("DecisionTree", best_dt), ("RandomForest", best_rf)]:
        pred = model.predict(Xte)
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro")

        print("\n" + "=" * 70)
        print(f"{name} RESULTS | Accuracy={acc:.4f} | MacroF1={f1m:.4f}")
        print("Best hyperparameters (key ones):")
        if name == "DecisionTree":
            print({
                "max_depth": model.get_params()["max_depth"],
                "min_samples_split": model.get_params()["min_samples_split"],
                "min_samples_leaf": model.get_params()["min_samples_leaf"],
            })
        else:
            print({
                "n_estimators": model.get_params()["n_estimators"],
                "max_depth": model.get_params()["max_depth"],
                "max_features": model.get_params()["max_features"],
                "min_samples_leaf": model.get_params()["min_samples_leaf"],
            })
        print("=" * 70)
        print(classification_report(y_test, pred, digits=3))

    print("\n--- Decision Tree interpretability (root node) ---")
    print_root_split(best_dt, vec)


if __name__ == "__main__":
    main()