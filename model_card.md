# 🧾 Model Card — Fake News Mini

**Version:** 1.0  
**Date:** 2025-10-14  
**Author:** Uday Kiran  

---

## 🧠 Model Overview

This project introduces lightweight and interpretable machine learning models for **fake news detection**, inspired by Refute’s mission to combat disinformation in real time.

The system includes multiple models trained on the same dataset, focusing on speed, transparency, and explainability rather than deep transformer complexity.

---

## 🧩 Models Included

| Model | Description | Algorithm | Features | File |
|--------|--------------|------------|-----------|------|
| **baseline_pipeline.joblib** | Baseline classifier using TF-IDF and Logistic Regression. | Logistic Regression | TF-IDF (word-level 1–2 grams) | `results/baseline_pipeline.joblib` |
| **pa_word_pipeline.joblib** | High-performance variant using online Passive-Aggressive updates. | Passive-Aggressive Classifier | TF-IDF (word-level 1–2 grams) | `results/pa_word_pipeline.joblib` |

---

## 📊 Performance Summary

| Metric | Baseline | Passive-Aggressive |
|--------|-----------|--------------------|
| Accuracy | 0.966 | **0.990** |
| Precision | 0.945 | **0.988** |
| Recall | 0.988 | **0.991** |
| F1-score | 0.966 | **0.990** |

> Test set size: 709 samples

---

## ⚙️ Intended Use

- **Purpose:** Identify and flag misleading or fake news text snippets for manual review or automated response workflows.  
- **Audience:** Developers, researchers, and analysts building disinformation detection pipelines.  
- **Scope:** English-language text from online articles and news headlines.

---

## 🚫 Limitations

- Trained on a single dataset — performance may vary on unseen domains (e.g., memes, satire).  
- English-only; multilingual or multimodal sources not supported.  
- Model decisions rely purely on text features (no source or network metadata).  
- Sensitive to very short text (< 5 words).

---

## 💡 Ethical & Practical Notes

- **Transparency:** Both models are interpretable — feature weights can be inspected for explainability.  
- **Bias:** Dataset bias (source imbalance, topic skew) may affect classification fairness.  
- **Deployment:** Recommended to combine with human verification for high-stakes scenarios.  
- **Privacy:** Does not log user identifiers; logs only anonymized text and scores for evaluation.

---

## 🔮 Future Enhancements

- Add **DistilBERT** fine-tuning for contextual robustness.  
- Extend coverage to **multilingual** and **cross-platform** disinformation.  
- Integrate **active learning** and **stream monitoring** for near real-time adaptation.  
- Deploy as a microservice with **Docker** or **FastAPI + Streamlit** UI.

---

## 🧰 Artifacts

| Artifact | Description |
|-----------|-------------|
| `results/baseline_pipeline.joblib` | Trained Logistic Regression model |
| `results/pa_word_pipeline.joblib` | Trained Passive-Aggressive model |
| `results/metrics.json` | Baseline metrics |
| `results/metrics_pa_word.json` | PA model metrics |
| `results/predictions_log.csv` | Inference logs |
| `src/api.py` | REST API (FastAPI) |

---

## 🏁 Citation

If you reference or reuse this work:
Kiran, Uday (2025). Fake News Mini — Lightweight Disinformation Detector.

---

✅ **Summary:**  
This model card documents the core components, metrics, and ethical scope of the Fake News Mini system — a fast, interpretable, and production-ready foundation for real-time fake news detection.
