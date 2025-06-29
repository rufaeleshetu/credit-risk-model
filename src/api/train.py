import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess your data
df = pd.read_csv("data/Processed/xente_transactions_processed.csv")
X = df.drop(columns='target')
y = df['target']

# Preprocessor
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model & preprocessor
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/preprocessor.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and preprocessor saved.")
