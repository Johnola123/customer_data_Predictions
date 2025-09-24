from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List

# Load model bundle
bundle = joblib.load("models/churn_model.pkl")
pipe = bundle["pipeline"]
features = bundle["features"]

class Customer(BaseModel):
    gender: str
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    internet_service: str
    payment_method: str
    has_dependents: int
    has_partner: int
    num_support_tickets: int
    is_active_member: int

app = FastAPI(title="Customer Churn Predictor", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.dict()])[features]
    proba = pipe.predict_proba(data)[0, 1]
    pred = int(proba >= 0.5)
    return {"churn_prediction": pred, "churn_probability": float(proba)}

@app.post("/predict_batch")
def predict_batch(customers: List[Customer]):
    data = pd.DataFrame([c.dict() for c in customers])[features]
    probas = pipe.predict_proba(data)[:, 1].tolist()
    preds = (pd.Series(probas) >= 0.5).astype(int).tolist()
    return {"predictions": preds, "probabilities": probas}
