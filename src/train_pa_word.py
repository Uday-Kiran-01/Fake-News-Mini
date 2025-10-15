# src/train_pa_word.py
import os, json, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from src.prepare_data import load_and_split

os.makedirs("results", exist_ok=True)

def main():
    X_tr, X_te, y_tr, y_te = load_and_split(path="data/train.csv", test_size=0.2, seed=42)

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        sublinear_tf=True
    )

    base = PassiveAggressiveClassifier(max_iter=1000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)

    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    probs = pipe.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print(classification_report(y_te, preds))
    cm = confusion_matrix(y_te, preds)
    print("Confusion matrix:\n", cm)

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary")
    acc = accuracy_score(y_te, preds)
    metrics = {"model":"pa_word","accuracy":float(acc),"precision":float(p),"recall":float(r),"f1":float(f1),"n_test":int(len(y_te))}
    with open("results/metrics_pa_word.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    import joblib
    joblib.dump(pipe, "results/pa_word_pipeline.joblib")
    print("Saved model to results/pa_word_pipeline.joblib")

if __name__ == "__main__":
    main()
