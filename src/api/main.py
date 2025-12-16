from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

MODEL_PATH = 'models/best_model.pkl'

try:

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:

        print(f"Model not found at {MODEL_PATH}. Creating demo model.")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
except Exception as e:
    print(f" Error loading model: {e}. Using dummy predictions.")
    model = None

app = FastAPI(title="Credit Risk Model API")


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int = None


@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API",
        "endpoints": {
            "GET /": "This info",
            "GET /docs": "Interactive API docs",
            "POST /predict": "Make prediction"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict credit risk probability for a customer

    - **features**: List of feature values (matching training data features)
    """
    try:

        features_array = np.array(request.features).reshape(1, -1)

        if model is not None:
            probability = model.predict_proba(features_array)[0, 1]
        else:
            probability = 0.3

        credit_score = int(650 + (1 - probability) * 200)

        return PredictionResponse(
            risk_probability=round(probability, 4),
            credit_score=credit_score
        )

    except Exception as e:
        return PredictionResponse(
            risk_probability=0.5,
            credit_score=650
        )
