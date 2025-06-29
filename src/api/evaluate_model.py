import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os

# Load data and model
data_path = "data/processed/xente_transactions_engineered.csv"
model_path = "models/random_forest_model.pkl"

df = pd.read_csv(data_path)
X = df.drop(columns=["FraudResult"])
y = df["FraudResult"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ROC Curve
y_proba = model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.savefig("outputs/roc_curve.png")
plt.close()

# Confusion Matrix
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Feature Importance
importances = model.feature_importances_
features = X.columns
feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values("Importance", ascending=False)
sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(10))
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()
