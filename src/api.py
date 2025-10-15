# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib, json, os, time
from fastapi.middleware.cors import CORSMiddleware

# ===================== Config =====================
RESULTS_DIR = "results"

# Threshold for classifying as FAKE (1)
THRESHOLD: float = float(os.getenv("MODEL_THRESHOLD", "0.50"))

# Model path/name are env-switchable; defaults to baseline
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(RESULTS_DIR, "baseline_pipeline.joblib"))
MODEL_NAME = os.getenv("MODEL_NAME", os.path.basename(MODEL_PATH).replace(".joblib", ""))

# We try these metrics files in order and use the first one found
METRICS_CANDIDATES = [
    os.getenv("METRICS_PATH", ""),            # allow explicit override
    os.path.join(RESULTS_DIR, "metrics.json"),            # baseline
    os.path.join(RESULTS_DIR, "metrics_pa_word.json"),    # PA model
    os.path.join(RESULTS_DIR, "metrics_char_svm.json"),   # char SVM
]
LOG_PATH = os.path.join(RESULTS_DIR, "predictions_log.csv")

# ===================== App ========================
app = FastAPI(title="Fake News Mini API", version="1.2.0")

# CORS (useful if you attach a minimal frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class Item(BaseModel):
    text: str

class BatchItems(BaseModel):
    texts: List[str]

# ===================== Load model & metrics ========
pipe = joblib.load(MODEL_PATH)

api_metrics = {}
for candidate in METRICS_CANDIDATES:
    if candidate and os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                api_metrics = json.load(f)
            break
        except Exception:
            api_metrics = {}
            break  # stop trying others if the chosen one errors

# ===================== Helpers =====================
def band(prob: float) -> str:
    """Simple business-friendly decision banding."""
    if prob >= max(THRESHOLD, 0.8):
        return "high_confidence"
    if 0.4 <= prob <= 0.7:
        return "uncertain"
    return "normal"

def label_name(lbl: int) -> str:
    return "fake" if lbl == 1 else "real"

def maybe_log(text: str, prob: float, lbl: int):
    """Append lightweight CSV logs; safe to disable if you prefer."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        header_needed = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("ts,model,threshold,prob,label,label_name,text\n")
            ts = int(time.time())
            safe_text = text.replace('"', '""')
            f.write(f'{ts},{MODEL_NAME},{THRESHOLD},{prob:.6f},{lbl},{label_name(lbl)},"{safe_text}"\n')
    except Exception:
        # best-effort logging; ignore errors
        pass

# ===================== Routes ======================
@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME, "threshold": THRESHOLD}

@app.get("/version")
def version():
    return {
        "model": MODEL_NAME,
        "api_version": app.version,
        "threshold": THRESHOLD,
        "metrics": api_metrics or None,
    }

@app.post("/predict")
def predict(item: Item):
    prob = float(pipe.predict_proba([item.text])[0, 1])
    lbl = int(prob >= THRESHOLD)
    resp = {
        "model": MODEL_NAME,
        "label": lbl,
        "label_name": label_name(lbl),
        "score": prob,
        "threshold": THRESHOLD,
        "decision_band": band(prob),
    }
    maybe_log(item.text, prob, lbl)
    return resp

@app.post("/predict_proba")
def predict_proba(item: Item):
    prob = float(pipe.predict_proba([item.text])[0, 1])
    return {"model": MODEL_NAME, "score": prob}

@app.post("/batch")
def batch_predict(items: BatchItems):
    probs = pipe.predict_proba(items.texts)[:, 1].tolist()
    labels = [int(p >= THRESHOLD) for p in probs]
    results = [
        {
            "label": lbl,
            "label_name": label_name(lbl),
            "score": float(p),
            "decision_band": band(float(p)),
        }
        for lbl, p in zip(labels, probs)
    ]
    for t, p, l in zip(items.texts, probs, labels):
        maybe_log(t, float(p), l)
    return {"model": MODEL_NAME, "threshold": THRESHOLD, "results": results}
