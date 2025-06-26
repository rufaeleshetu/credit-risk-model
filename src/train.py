import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# === 1. Load & normalize data
df = pd.read_csv("data/raw/xente_transactions.csv")
df.columns = df.columns.str.strip().str.lower()

# === 2. Simulate binary target from `amount`
if "amount" not in df.columns:
    raise ValueError("Column 'amount' not found even after normalization!")
y = (df["amount"] > 100000).astype(int)

# === 3. Select only numerical features (drop IDs, timestamps, and categorical vars)
drop_cols = ["transactionid", "customerid", "transactionstarttime"]
X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include="number")

# === ✅ Handle missing values (drop rows with NaN)
X = X.dropna()
y = y[X.index]  # sync labels

# === 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_bal).value_counts())

# === 5. Oversample using SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_bal).value_counts())


# === 6. Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# === 7. Train model
pipeline.fit(X_train_bal, y_train_bal)

# === 8. Evaluate
y_pred = pipeline.predict(X_test)
print("\nModel Evaluation Report:\n")
print(classification_report(y_test, y_pred))

with open("reports/eval_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))


import joblib
joblib.dump(pipeline, "models/credit_risk_model.pkl")
