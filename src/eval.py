# src/eval.py
import joblib
from src.robustness import add_spacing_noise, punct_noise, leetspeak

pipe = joblib.load("results/baseline_pipeline.joblib")

samples = [
    "Breaking: shocking claim spreads on social media about election fraud",
    "Report: central bank publishes measured update on inflation outlook",
]

for s in samples:
    base = float(pipe.predict_proba([s])[0,1])
    noisy = float(pipe.predict_proba([add_spacing_noise(punct_noise(leetspeak(s)))])[0,1])
    print(f"\nTEXT: {s}\nscore_base={base:.3f}  score_noisy={noisy:.3f}")
