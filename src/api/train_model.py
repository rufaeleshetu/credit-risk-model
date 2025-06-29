# src/train.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import cloudpickle
import os

# Dummy example: replace with actual dataset and preprocessing
X = pd.DataFrame({
    "age": [25, 32, 47],
    "income": [3000, 4500, 6000],
    "loan_amount": [10000, 15000, 20000],
    "credit_score": [650, 700, 750]
})
y = [0, 1, 0]

# Define pipeline
preprocessor = StandardScaler()
model = Pipeline([
    ('scaler', preprocessor),
    ('clf', LogisticRegression())
])

model.fit(X, y)

# Save preprocessor and model
os.makedirs("models", exist_ok=True)
with open("models/preprocessor.pkl", "wb") as f:
    cloudpickle.dump(preprocessor, f)

with open("models/model.pkl", "wb") as f:
    cloudpickle.dump(model, f)
