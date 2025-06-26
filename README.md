## Credit Scoring Business Understanding

**1. Basel II & Interpretability**

The Basel II Accord requires financial institutions to assess credit risk using internal models that are both transparent and auditable. As such, models must be interpretable to justify decisions to regulators, reduce bias, and ensure fairness. This makes simple, explainable models preferable in regulated environments.

**2. Importance of Proxy Target**

Without a true default label in the dataset, we must create a proxy variable—such as labeling inactive customers as high-risk. This allows training a model to predict creditworthiness. However, this introduces risk: a poorly defined proxy can lead to biased predictions and incorrect decisions.

**3. Simple vs Complex Models**

Logistic Regression with WoE is highly interpretable, making it suitable for compliance-heavy environments. On the other hand, Gradient Boosting offers higher accuracy but lacks transparency. In regulated contexts, the safer trade-off often favors interpretability over raw performance.