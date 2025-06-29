import os
import joblib

# Get current file directory (src/api)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to project root
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# Point to the model file
MODEL_PATH = os.path.join(ROOT_DIR, "model", "fraud_model.pkl")

print("Resolved MODEL_PATH:", MODEL_PATH)

# Load the model
model = joblib.load(MODEL_PATH)
