#!/usr/bin/env python3
"""
QProp evaluation add-on:
- TF-IDF + Logistic Regression baseline
- Precision/Recall/F1
- Precision–Recall curve
- Threshold sweep to pick best threshold by F1

Outputs saved to ./results
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\\S+", " <URL> ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def guess_columns(df: pd.DataFrame):
    cols_lower = [c.lower() for c in df.columns]

    # label column guesses
    label_candidates = ["label", "class", "y", "target", "is_propaganda", "propaganda"]
    label_col = None
    for cand in label_candidates:
        if cand in cols_lower:
            label_col = df.columns[cols_lower.index(cand)]
            break

    # text column guesses
    text_candidates = ["text", "content", "article", "body", "sentence", "headline", "title"]
    text_col = None
    for cand in text_candidates:
        if cand in cols_lower:
            text_col = df.columns[cols_lower.index(cand)]
            break

    return text_col, label_col

def f1_from_pr(p, r):
    denom = (p + r)
    return np.where(denom == 0, 0, 2 * p * r / denom)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="qprop_data/proppy_1.0.train.tsv",
                    help="Path to local QProp TSV/CSV (DO NOT upload to GitHub)")
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load TSV or CSV
    p = Path(args.data)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Put the QProp TSV on your computer in qprop_data/ (it is ignored by git),\n"
            "or pass --data path/to/file.tsv"
        )

    if p.suffix.lower() == ".tsv":
        df = pd.read_csv(p, sep="\\t", low_memory=False)
    else:
        df = pd.read_csv(p, low_memory=False)

    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    text_col, label_col = guess_columns(df)
    if args.text_col: text_col = args.text_col
    if args.label_col: label_col = args.label_col

    if not text_col or not label_col:
        print("Could not auto-detect text/label columns.")
        print("Re-run with: --text_col <TEXT> --label_col <LABEL>")
        return

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(clean_text)

    # Convert -1/1 -> 0/1 if needed
    uniq = set(pd.unique(df[label_col]))
    if uniq == {-1, 1}:
        df[label_col] = (df[label_col] == 1).astype(int)

    # Split (stratified so imbalance is preserved in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )

    # TF-IDF baseline
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    # Logistic Regression (balanced)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    # Probabilities for threshold tuning
    probs = clf.predict_proba(Xte)[:, 1]

    # PR curve + best threshold by F1
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = f1_from_pr(precision, recall)

    best_idx = int(np.argmax(f1))
    best_p = float(precision[best_idx])
    best_r = float(recall[best_idx])
    best_f1 = float(f1[best_idx])

    # threshold array is length-1 vs precision/recall; handle safely
    if best_idx == 0:
        best_thr = 0.5
    else:
        best_thr = float(thresholds[min(best_idx-1, len(thresholds)-1)])

    # Make predictions using best threshold
    pred_best = (probs >= best_thr).astype(int)

    report = classification_report(y_test, pred_best, digits=3)
    cm = confusion_matrix(y_test, pred_best)

    # Save report
    (out_dir / "report.txt").write_text(
        f"Best threshold: {best_thr:.3f}\\n"
        f"Precision: {best_p:.3f}  Recall: {best_r:.3f}  F1: {best_f1:.3f}\\n\\n"
        f"Confusion Matrix:\\n{cm}\\n\\n"
        f"Classification Report:\\n{report}\\n",
        encoding="utf-8"
    )

    # Save PR curve image
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (TF-IDF + Logistic Regression)")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=200)
    plt.close(fig)

    # Save threshold sweep table (sampled thresholds)
    sweep = []
    for thr in np.arange(0.10, 0.91, 0.05):
        pred = (probs >= thr).astype(int)
        # compute precision/recall/f1 manually
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        p_val = tp / (tp + fp) if (tp + fp) else 0
        r_val = tp / (tp + fn) if (tp + fn) else 0
        f1_val = (2*p_val*r_val/(p_val+r_val)) if (p_val+r_val) else 0
        sweep.append([thr, p_val, r_val, f1_val])

    pd.DataFrame(sweep, columns=["threshold", "precision", "recall", "f1"]).to_csv(
        out_dir / "threshold_sweep.csv", index=False
    )

    print("Saved:")
    print(" -", out_dir / "pr_curve.png")
    print(" -", out_dir / "threshold_sweep.csv")
    print(" -", out_dir / "report.txt")
    print("\\nBest threshold:", best_thr)

if __name__ == "__main__":
    main()
