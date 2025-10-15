# src/train_baseline.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.calibration import calibration_curve

from src.prepare_data import load_and_split

import joblib

def ensure_dirs():
    os.makedirs("results", exist_ok=True)


def plot_confusion_matrix(cm, out_path):
    plt.figure()
    # Pure matplotlib version for fewer deps
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Baseline)")
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ["Real (0)", "Fake (1)"]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(y_true, y_prob, out_path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Baseline")
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.title("Calibration Curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()

    # Load prepared splits (handles Headline/Body/Label via prepare_data.py)
    X_tr, X_te, y_tr, y_te = load_and_split(path="data/train.csv", test_size=0.2, seed=42)

    # TF-IDF + Logistic Regression baseline
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )

    pipe.fit(X_tr, y_tr)

    # Predictions & probabilities
    probs = pipe.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Text report
    print(classification_report(y_te, preds))

    # Confusion matrix
    cm = confusion_matrix(y_te, preds)
    print("Confusion matrix:\n", cm)
    plot_confusion_matrix(cm, "results/confusion_matrix.png")

    # Calibration
    plot_calibration_curve(y_te, probs, "results/calibration_curve.png")

    # Metrics JSON
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary")
    acc = accuracy_score(y_te, preds)
    metrics = {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_test": int(len(y_te)),
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to results/metrics.json")

    # Save model pipeline for API use
    joblib.dump(pipe, "results/baseline_pipeline.joblib")
    print("Saved model to results/baseline_pipeline.joblib")

    # Show top positive/negative features (global explanation)
    clf = pipe.named_steps["clf"]
    tfidf = pipe.named_steps["tfidf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    top_fake_idx = np.argsort(coefs)[-20:][::-1]   # highest weights → label 1 (fake)
    top_real_idx = np.argsort(coefs)[:20]          # lowest weights → label 0 (real)

    print("\nTop features indicating FAKE (label=1):")
    for i in top_fake_idx:
        print(f"{feature_names[i]}  ({coefs[i]:.4f})")

    print("\nTop features indicating REAL (label=0):")
    for i in top_real_idx:
        print(f"{feature_names[i]}  ({coefs[i]:.4f})")


if __name__ == "__main__":
    main()
