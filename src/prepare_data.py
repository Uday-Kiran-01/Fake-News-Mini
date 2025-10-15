# src/prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import re, hashlib
from pathlib import Path

# force these names for your dataset
FORCE_TITLE_COL = "Headline"
FORCE_TEXT_COL = "Body"
FORCE_LABEL_COL = "Label"

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def text_hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load_and_split(path="data/train.csv", test_size=0.2, seed=42):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    # read CSV
    df = pd.read_csv(path)
    print("Columns detected:", list(df.columns))

    # verify expected columns
    for c in [FORCE_TITLE_COL, FORCE_TEXT_COL, FORCE_LABEL_COL]:
        if c not in df.columns:
            raise KeyError(
                f"Expected column '{c}' not found. Columns present: {list(df.columns)}"
            )

    # combine headline + body text
    df["text_norm"] = (
        df[FORCE_TITLE_COL].fillna("") + " " + df[FORCE_TEXT_COL].fillna("")
    ).apply(normalize_text)

    # map label column to 0/1
    if df[FORCE_LABEL_COL].dtype == object:
        df["label"] = df[FORCE_LABEL_COL].str.lower().map({"real": 0, "fake": 1})
    else:
        df["label"] = df[FORCE_LABEL_COL].astype(int)

    # remove missing text
    df = df[df["text_norm"].str.strip().str.len() > 0]

    # drop duplicates
    df["text_hash"] = df["text_norm"].apply(text_hash)
    df = df.drop_duplicates(subset=["text_hash"]).drop(columns=["text_hash"])

    X = df["text_norm"].values
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")
    return X_tr, X_te, y_tr, y_te

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_and_split()
