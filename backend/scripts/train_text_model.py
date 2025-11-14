"""
Training script for text-only model.

Expect a JSONL file with lines like:
{"text": "I love writing code.", "labels": [3.2, 4.0, 2.8, 1.9, 3.7]}

This script:
- loads dataset
- extracts features (SBERT + lexical)
- trains a model using train_text_model
- reports evaluation metrics (Pearson r & MAE per trait)
"""

import json
import numpy as np
import os
import sys
# ensure backend package root on sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.text_features import text_to_features
from backend.models.text_model import create_feature_vector, train_text_model, predict_text_model, load_text_model
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_labeled_text.jsonl")

def load_dataset(path):
    X_list = []
    Y_list = []
    lex_keys = None
    for line in open(path, "r", encoding="utf-8"):
        obj = json.loads(line)
        text = obj.get("text", "")
        labels = obj.get("labels", None)
        if labels is None:
            continue
        emb, lex = text_to_features(text)
        if lex_keys is None:
            lex_keys = sorted(list(lex.keys()))
        feat = create_feature_vector(emb, lex, lex_keys)
        X_list.append(feat)
        Y_list.append(labels)
    X = np.vstack(X_list).astype(np.float32)
    Y = np.vstack(Y_list).astype(np.float32)
    return X, Y, lex_keys

def evaluate(y_true, y_pred):
    # y_true, y_pred: n x 5
    metrics = {}
    trait_names = ["extraversion","agreeableness","conscientiousness","neuroticism","openness"]
    for i, name in enumerate(trait_names):
        r = pearsonr(y_true[:,i], y_pred[:,i])[0]
        mae = mean_absolute_error(y_true[:,i], y_pred[:,i])
        metrics[name] = {"pearson_r": float(r), "mae": float(mae)}
    return metrics

def main():
    print("Loading dataset:", DATA_FILE)
    X, Y, lex_keys = load_dataset(DATA_FILE)
    print("Dataset sizes:", X.shape, Y.shape)
    model = train_text_model(X, Y)
    preds = predict_text_model(model, X)
    metrics = evaluate(Y, preds)
    print("Training evaluation:")
    for t, m in metrics.items():
        print(f" - {t}: Pearson r = {m['pearson_r']:.3f}, MAE = {m['mae']:.3f}")
    print("Model saved. Lexical key order (store this alongside model):")
    print(lex_keys)
    # Save lex_keys next to model
    model_meta = {"lex_keys": lex_keys}
    meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "text_personality_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f)
    print("Saved meta to", meta_path)

if __name__ == "__main__":
    main()
