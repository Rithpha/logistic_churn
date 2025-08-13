from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load model, DictVectorizer, and scaler
with open("churn-model.bin", "rb") as f_in:
    dv, scaler, model = pickle.load(f_in)

app = FastAPI()

# Input model
class Customer(BaseModel):
    tenure: float
    monthlycharges: float
    totalcharges: float
    gender: str
    seniorcitizen: str
    partner: str
    dependents: str
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str

numerical = ['tenure', 'monthlycharges', 'totalcharges']

# Prediction function
def predict_single(customer: dict):
    df = pd.DataFrame([customer])
    df[numerical] = scaler.transform(df[numerical])
    X = dv.transform(df.to_dict(orient="records"))
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

@app.post("/predict")
def predict(customer: Customer):
    pred = predict_single(customer.dict())
    churn = pred >= 0.5
    return {
        "churn_probability": float(pred),
        "churn": bool(churn)
    }
