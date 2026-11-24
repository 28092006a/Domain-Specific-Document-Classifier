"""
train_model.py
==============
Trains and compares two classifiers, then saves the better one.

Steps
-----
 1. Load and clean the dataset
 2. NLP preprocessing of every document
 3. Train/test split (80 / 20, stratified)
 4. Build two pipelines: TF-IDF + Multinomial Naive Bayes, TF-IDF + Logistic Regression
 5. Train, evaluate (Accuracy, Precision, Recall, F1, Classification Report, Confusion Matrix)
 6. Also run 5-fold cross-validation (more trustworthy on a small dataset)
 7. Pick the best model, re-train it on ALL data, save it with Joblib
 8. Save metrics (JSON) and charts (PNG) so the Streamlit dashboard can show them

Run:   python train_model.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import utils
from preprocessing import preprocess_text

RANDOM_STATE = 42   # fixed seed -> same results every run
TEST_SIZE = 0.20    # 20% of documents are held back for testing


# --------------------------------------------------------------------------
# STEP 1-2: DATASET PREPARATION AND CLEANING
# --------------------------------------------------------------------------
def load_and_clean_dataset():
    """Read the CSV, remove bad rows, and add a 'clean_text' column."""
    if not utils.DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {utils.DATASET_PATH}")

    df = pd.read_csv(utils.DATASET_PATH)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("The dataset must contain the columns 'text' and 'label'.")

    start_rows = len(df)
    df = df.dropna(subset=["text", "label"]).copy()              # remove missing values
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.title()  # 'medical ' -> 'Medical'
    df = df[df["text"] != ""]                                     # remove empty text
    df = df.drop_duplicates(subset="text")                        # remove duplicates
    df = df[df["label"].isin(utils.DOMAINS)]                      # keep only known domains

    df["clean_text"] = df["text"].apply(preprocess_text)          # NLP preprocessing
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    print(f"Dataset cleaned: {start_rows} rows -> {len(df)} usable rows")
    print("Class distribution:")
    print(df["label"].value_counts().to_string(), "\n")

    if df["label"].value_counts().min() < 5:
        raise ValueError("Each domain needs at least 5 documents for reliable training.")
    return df


# --------------------------------------------------------------------------
# STEP 4: MODEL DEFINITIONS (TF-IDF feature engineering + classifier)
# --------------------------------------------------------------------------
def build_models():
    """Return {model_name: sklearn Pipeline}. Each pipeline = TF-IDF -> classifier."""

    def tfidf():
        # ngram_range=(1,2): single words AND two-word phrases ("blood pressure")
        # sublinear_tf: dampens very frequent words;  max_df: ignore near-universal words
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.9)

    return {
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]
        ),
        "Logistic Regression": Pipeline(
            [("tfidf", tfidf()), ("clf", LogisticRegression(C=10, max_iter=1000))]
        ),
    }


# --------------------------------------------------------------------------
# STEP 5-6: EVALUATION
# --------------------------------------------------------------------------
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test, X_all, y_all, labels):
    """Train one pipeline and return all its metrics in a dictionary."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 5-fold cross-validation on the whole dataset (the pipeline is re-fitted
    # inside every fold, so there is no data leakage)
    n_splits = min(5, int(pd.Series(y_all).value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clone(pipeline), X_all, y_all, cv=cv, scoring="f1_weighted")

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "cv_folds": int(n_splits),
        "report": classification_report(y_test, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }


def plot_model_comparison(results, save_path):
    """Grouped bar chart: Accuracy / Precision / Recall / F1 for each model."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    names = list(results.keys())
    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        values = [results[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width - width / 2, values, width, label=name)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in metric_keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Naive Bayes vs Logistic Regression (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# --------------------------------------------------------------------------
def main():
    """Run the full training workflow and return the saved metrics dictionary."""
    utils.ensure_directories()
    df = load_and_clean_dataset()

    X, y = df["clean_text"], df["label"]
    labels = sorted(y.unique())

    # Train-test split. stratify=y keeps the same domain proportions in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training documents: {len(X_train)} | Testing documents: {len(X_test)}\n")

    models = build_models()
    results = {}
    for name, pipeline in models.items():
        print("=" * 64)
        print(f"MODEL: {name}")
        print("=" * 64)
        results[name] = evaluate_model(
            name, pipeline, X_train, X_test, y_train, y_test, X, y, labels
        )
        r = results[name]
        print(f"Accuracy : {r['accuracy']:.4f}")
        print(f"Precision: {r['precision']:.4f} (weighted)")
        print(f"Recall   : {r['recall']:.4f} (weighted)")
        print(f"F1-score : {r['f1']:.4f} (weighted)")
        print(f"{r['cv_folds']}-fold CV F1: {r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f}\n")
        print(r["report"])

        # Save the confusion-matrix picture for the dashboard / report
        slug = name.lower().replace(" ", "_")
        fig = utils.plot_confusion_matrix(
            np.array(r["confusion_matrix"]), labels, f"Confusion Matrix - {name}",
            save_path=utils.MODEL_DIR / f"confusion_matrix_{slug}.png",
        )
        plt.close(fig)

    # STEP 7: choose the winner (test F1 first, cross-validation F1 breaks ties)
    best_name = max(results, key=lambda n: (round(results[n]["f1"], 6), results[n]["cv_f1_mean"]))
    print("*" * 64)
    print(f"BEST MODEL: {best_name}")
    print("*" * 64)

    plot_model_comparison(results, utils.MODEL_DIR / "model_comparison.png")

    # Re-train the winning pipeline on ALL data: with a small dataset every
    # document helps. The reported metrics above come from the held-out test set.
    final_pipeline = clone(models[best_name]).fit(X, y)

    bundle = {
        "pipeline": final_pipeline,
        "model_name": best_name,
        "classes": list(final_pipeline.classes_),
        "n_training_samples": int(len(df)),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, utils.MODEL_PATH)  # Joblib: efficient saving of sklearn objects
    print(f"Model saved to: {utils.MODEL_PATH}")

    metrics = {
        "best_model": best_name,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "labels": labels,
        "models": results,
    }
    with open(utils.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plain-text accuracy report (handy to paste into the project report)
    with open(utils.MODEL_DIR / "accuracy_report.txt", "w", encoding="utf-8") as f:
        for name, r in results.items():
            f.write(f"{name}\n{'-' * len(name)}\n")
            f.write(f"Accuracy={r['accuracy']:.4f}  Precision={r['precision']:.4f}  "
                    f"Recall={r['recall']:.4f}  F1={r['f1']:.4f}\n")
            f.write(f"CV F1 = {r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f}\n\n{r['report']}\n\n")
        f.write(f"BEST MODEL: {best_name}\n")
    print(f"Metrics and charts saved in: {utils.MODEL_DIR}")
    return metrics


if __name__ == "__main__":
    main()
