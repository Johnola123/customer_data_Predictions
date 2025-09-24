import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/synthetic_telco_churn.csv")

# Feature columns
features = [
    "gender",
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "internet_service",
    "payment_method",
    "has_dependents",
    "has_partner",
    "num_support_tickets",
    "is_active_member",
]

target = "churn"

X = df[features]
y = df[target]

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# -----------------------------
# 3. Preprocessing
# -----------------------------
categorical = ["gender", "contract_type", "internet_service", "payment_method"]
numerical = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "has_dependents",
    "has_partner",
    "num_support_tickets",
    "is_active_member",
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical),
    ]
)

# -----------------------------
# 4. Model
# -----------------------------
clf = LogisticRegression(max_iter=200, solver="lbfgs")

pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])

# -----------------------------
# 5. Train
# -----------------------------
pipe.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))



import joblib
joblib.dump(pipe, "models/churn_model.pkl")

