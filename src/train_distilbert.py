# src/train_distilbert.py
import os, sys, json, time
print(f"🐍 Python: {sys.version}")
print(f"📄 Running file: {__file__}")
print(f"📂 CWD: {os.getcwd()}")

# Ensure results dir
os.makedirs("results/bert", exist_ok=True)

# Imports (loud)
try:
    import numpy as np
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from src.prepare_data import load_and_split
    print("✅ Imports OK")
except Exception as e:
    print("❌ Import failure:", repr(e))
    raise

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    try:
        print("🚀 Starting DistilBERT fine-tuning...")
        print("CUDA available:", torch.cuda.is_available())

        # Load data
        X_tr, X_te, y_tr, y_te = load_and_split(path="data/train.csv", test_size=0.2, seed=42)
        print(f"📊 Split sizes — train: {len(X_tr)}, test: {len(X_te)}")

        # HF datasets
        train_ds = Dataset.from_dict({"text": X_tr, "label": y_tr})
        test_ds  = Dataset.from_dict({"text": X_te, "label": y_te})
        dsd = DatasetDict({"train": train_ds, "test": test_ds})

        # Tokenizer & tokenize
        model_name = "distilbert-base-uncased"
        print("🔡 Loading tokenizer:", model_name)
        tok = AutoTokenizer.from_pretrained(model_name)

        def tokenize(batch):
            return tok(batch["text"], truncation=True, padding="max_length", max_length=256)

        print("🧩 Tokenizing...")
        dsd = dsd.map(tokenize, batched=True).remove_columns(["text"])
        dsd.set_format(type="torch")

        # Model
        print("🧠 Loading model:", model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Training args
        args = TrainingArguments(
            output_dir="results/bert",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=2,
            weight_decay=0.01,
            seed=42,
            logging_steps=50,
            report_to=[],  # no wandb
            disable_tqdm=False,
        )

        # Trainer
        print("🏃 Training starts…")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dsd["train"],
            eval_dataset=dsd["test"],
            compute_metrics=compute_metrics,
        )

        trainer.train()
        print("✅ Training complete. Evaluating…")
        metrics = trainer.evaluate()
        print("📈 Metrics:", metrics)

        with open("results/bert/metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        print("💾 Saved metrics to results/bert/metrics.json")

        # Save model + tokenizer
        out_dir = "results/bert/model"
        print(f"💾 Saving model to {out_dir}")
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
        print("🎉 Done.")

    except Exception as e:
        print("❌ Fatal error:", repr(e))
        raise

if __name__ == "__main__":
    main()
