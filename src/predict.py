import pandas as pd
import joblib
from preprocessing import preprocess_data

# === Step 1: Load raw data ===
df = pd.read_csv("data/raw/xente_transactions.csv")
agg_df = preprocess_data(df)

# === Step 2: Load model artifacts ===
model = joblib.load("models/credit_risk_model.joblib")
woe = joblib.load("models/woe_transformer.joblib")
expected_columns = joblib.load("models/model_columns.joblib")  # Optional but recommended

# === Step 3: Apply WOE transformation ===
X_woe = woe.transform(agg_df)

# === Step 4: Ensure correct column order ===
X_woe = X_woe[expected_columns]

# === Step 5: Predict using model (includes internal preprocessor) ===
predictions = model.predict(X_woe)

# === Step 6: Output predictions ===
print("✅ Predictions:\n", predictions)

# Optional: add predictions to original DataFrame
agg_df['prediction'] = predictions

# Save to CSV
agg_df.to_csv("data/predictions/predicted_risks.csv", index=False)
print("📁 Predictions saved to data/predictions/predicted_risks.csv")
