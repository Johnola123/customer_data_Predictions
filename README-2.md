# Customer Churn Predictor (Synthetic)

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs
```

## Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Request example
POST /predict
```json
{
  "gender": "Male",
  "age": 45,
  "tenure_months": 3,
  "monthly_charges": 75.5,
  "total_charges": 226.5,
  "contract_type": "month-to-month",
  "internet_service": "Fiber",
  "payment_method": "Electronic check",
  "has_dependents": 0,
  "has_partner": 0,
  "num_support_tickets": 2,
  "is_active_member": 1
}
```
