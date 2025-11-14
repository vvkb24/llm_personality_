"""
Text-only personality model utilities.

- Trains a simple regression model on concatenated [embedding | lexical features]
- Uses scikit-learn's Ridge (fast, stable) as default
- Saves model & feature metadata with joblib
"""

import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, List, Dict

MODEL_FILENAME = os.path.join(os.path.dirname(__file__), "..", "models", "text_personality_model.joblib")

def create_feature_vector(embedding: np.ndarray, lex_features: Dict[str, float], lex_order: List[str]):
    """
    Build a single feature vector from embedding and lexical dict.
    lex_order controls feature ordering for consistent training/inference.
    """
    lex_vec = np.array([float(lex_features.get(k, 0.0)) for k in lex_order], dtype=np.float32)
    return np.concatenate([embedding, lex_vec])

def train_text_model(X: np.ndarray, y: np.ndarray):
    """
    Train and persist a multi-output regressor.
    - X: n_samples x n_features
    - y: n_samples x 5  (columns: extraversion, agreeableness, conscientiousness, neuroticism, openness)
    Returns trained model.
    """
    # pipeline: scale features -> multioutput ridge
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("est", MultiOutputRegressor(Ridge(alpha=1.0)))
    ])
    pipeline.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_FILENAME), exist_ok=True)
    joblib.dump(pipeline, MODEL_FILENAME)
    return pipeline

def load_text_model():
    if os.path.exists(MODEL_FILENAME):
        return joblib.load(MODEL_FILENAME)
    return None

def predict_text_model(model, X: np.ndarray):
    """
    Predict; clamp outputs to 1.0 - 5.0 range.
    """
    preds = model.predict(X)
    preds = np.clip(preds, 1.0, 5.0)
    return preds
