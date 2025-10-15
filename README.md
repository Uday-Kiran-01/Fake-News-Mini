# 🧠 Fake News Mini — Lightweight Disinformation Detector

A compact, interpretable text-classification pipeline that detects fake news using TF-IDF features and classical ML models.

---

## 🚀 Project Overview
- Detects and classifies fake vs real news articles.
- Focuses on fast, transparent predictions suitable for real-time use.
- Includes baseline (TF-IDF + Logistic Regression) and an upgraded Passive-Aggressive model (~99 % F1).

---

## 📁 Structure
data/ → training CSV
results/ → saved models, metrics, logs
src/ → all code (training, API, evaluation)

---

## ⚙️ Setup
```bash
pip install -r requirements.txt

Train models (already done):
python -m src.train_baseline
python -m src.train_pa_word

Run API:
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

or, to serve the PA model:
$env:MODEL_PATH="results/pa_word_pipeline.joblib"; $env:MODEL_NAME="pa_word"; python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

Docs UI → http://127.0.0.1:8000/docs

📊 Results
Model	Accuracy	Precision	Recall	F1
baseline (TF-IDF + LR)	0.966	0.945	0.988	0.966
pa_word (TF-IDF + PassiveAggressive)	0.990	0.988	0.991	0.990

---

## 🧪 Model Comparison

A simple evaluation script `src/compare_metrics.py` automatically compares all available model metric files in the `results/` folder and prints them as a clean CSV table.

Run it anytime to see updated results:

```bash
python -m src.compare_metrics

Example output:

model,accuracy,precision,recall,f1,n_test
baseline,0.9661,0.9446,0.9884,0.9660,709
pa_word,0.9901,0.9884,0.9913,0.9899,709


🧩 API Features

/predict, /batch, /predict_proba

Configurable threshold (MODEL_THRESHOLD)

Logs each request to results/predictions_log.csv

Returns decision bands: high_confidence, uncertain, normal

🧠 Next Steps

(Optional) fine-tune DistilBERT (requires Python 3.11 + PyTorch)

Add ONNX or HF-Inference integration

Expand dataset and evaluate multilingual robustness

Deploy via Docker or Streamlit

📜 License & Acknowledgement

Dataset: public “Fake News Detection” corpus (Kaggle).
Built for research and demonstration purposes.


---

### 2️⃣ Fill `model_card.md`
Keep it short, e.g.:

```markdown
# Model Card — Fake News Mini
**Version:** 1.0  
**Date:** 2025-10-14  

**Models:**  
- `baseline_pipeline.joblib` — TF-IDF + Logistic Regression  
- `pa_word_pipeline.joblib` — TF-IDF (1–2-gram) + PassiveAggressive  

**Performance:** see README table.  
**Intended use:** Detect and flag likely fake-news text for manual review or automated response.  
**Limitations:** English-only, trained on single dataset, may not generalize to memes or non-text forma