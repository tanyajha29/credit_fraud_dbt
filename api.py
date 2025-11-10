import mlflow
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel # Corrected import: should be pydantic
from datetime import date, datetime
import os

# Tell MLflow where the server is
mlflow.set_tracking_uri("http://localhost:5000")

# --- !! PASTE YOUR MLFLOW RUN ID HERE !! ---
MLFLOW_RUN_ID = "7576ee8e910843d59d1c65893f734df6"
# -------------------------------------------

MODEL_NAME = "fraud_detector_pipeline" # Matches the artifact name in train_model.py
MODEL_PATH = f"runs:/{MLFLOW_RUN_ID}/{MODEL_NAME}"

print(f"Loading model '{MODEL_NAME}' from MLflow Run ID: {MLFLOW_RUN_ID}...")
try:
    # This loads the *entire* Scikit-learn pipeline
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure MLflow UI is running, the Run ID is correct, and the model artifact exists.")
    exit()

app = FastAPI()

# Define Pydantic model for input data (matches raw inputs pipeline expects)
class Transaction(BaseModel):
    transaction_timestamp: datetime
    date_of_birth: date
    category: str
    amount: float
    gender: str
    job: str
    city_population: int
    customer_latitude: float
    customer_longitude: float
    merchant_latitude: float
    merchant_longitude: float

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # 1. Convert Pydantic model to dict
        data_dict = transaction.dict()

        # 2. Create the DataFrame *first*
        data_df = pd.DataFrame([data_dict])

        # 3. THEN convert date/time columns *within the DataFrame*
        data_df['transaction_timestamp'] = pd.to_datetime(data_df['transaction_timestamp'])
        data_df['date_of_birth'] = pd.to_datetime(data_df['date_of_birth'])

        # 4. Access the underlying scikit-learn pipeline
        sklearn_pipeline = model._model_impl

        # 5. Predict Probability using the scikit-learn pipeline and the prepared DataFrame
        fraud_probability = sklearn_pipeline.predict_proba(data_df)[:, 1][0]

        # 6. Apply Threshold
        threshold = 0.3 # Adjust as needed
        is_fraud = 1 if fraud_probability >= threshold else 0

        # 7. Return JSON result
        return {
            "is_fraud": is_fraud,
            "prediction": "Fraud" if is_fraud == 1 else "Not Fraud",
            "fraud_probability": round(fraud_probability, 4)
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        # Ensure error response is valid JSON
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "detail": "Prediction failed"
        }

@app.get("/")
def health_check():
    return {"status": "API is running", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)