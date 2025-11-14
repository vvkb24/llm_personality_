"""
FastAPI router for text-only analysis and training trigger.

Endpoints:
- POST /analyze/text   -> {"text": "..."}  returns personality scores + features
- POST /train          -> (admins) trains model from backend/data/sample_labeled_text.jsonl
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.models.text_features import text_to_features
from backend.models.text_model import load_text_model, predict_text_model
import numpy as np
import os
import json
from typing import Dict
from backend.models.text_model import create_feature_vector

router = APIRouter()

class TextIn(BaseModel):
    text: str

@router.post("/analyze/text")
def analyze_text(payload: TextIn):
    text = payload.text
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    emb, lex = text_to_features(text)
    # load meta keys to order lexical features
    meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "text_personality_meta.json")
    if not os.path.exists(meta_path):
        # Try to compute order from current lex
        lex_keys = sorted(list(lex.keys()))
    else:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        lex_keys = meta.get("lex_keys", sorted(list(lex.keys())))
    X = create_feature_vector(emb, lex, lex_keys).reshape(1, -1)
    model = load_text_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Call /train first.")
    preds = predict_text_model(model, X)[0]
    trait_names = ["extraversion","agreeableness","conscientiousness","neuroticism","openness"]
    return {
        "personality": {trait_names[i]: float(preds[i]) for i in range(5)},
        "lexical_features": lex
    }

@router.post("/train")
def train_from_data():
    # Warning: This is a local training endpoint for convenience.
    # In production secure or remove this endpoint.
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_text_model.py")
    if not os.path.exists(script_path):
        raise HTTPException(status_code=500, detail="Training script not found.")
    # Run training script as a subprocess (so it can print logs)
    import subprocess, sys
    try:
        p = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        return {"status": "trained", "stdout": p.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "stderr": e.stderr, "stdout": e.stdout}
