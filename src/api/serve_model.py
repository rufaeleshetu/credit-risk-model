from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Adjust path based on where you run uvicorn from
model = joblib.load("../../models/credit_risk_model.pkl")

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}
