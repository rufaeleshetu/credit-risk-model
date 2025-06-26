## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord mandates that financial institutions adopt internal rating systems to assess and manage credit risk responsibly. To comply, models must not only be accurate but also transparent, interpretable, and auditable. This ensures that decisions about loan approvals or rejections can be justified to regulators and stakeholders. As a result, our credit risk model must be well-documented and based on explainable techniques to build trust, ensure fairness, and meet regulatory standards.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In our dataset, there is no explicit label indicating whether a customer defaulted on a loan. To proceed with model training, we must create a proxy variable — for example, using disengagement or spending behavior (like RFM: Recency, Frequency, Monetary) to label customers as high or low risk. However, using a proxy introduces risks: if poorly designed, it may misclassify customers, leading to unfair credit denials or approvals, reputational damage, and potential financial losses. Care must be taken to define this proxy accurately and validate it against business realities.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

Simple models like Logistic Regression with Weight of Evidence (WoE) offer high interpretability, which is crucial for explaining decisions to regulators and complying with financial laws. They allow for transparent credit scoring that aligns well with the risk governance structure of banks. However, these models may not capture complex patterns in the data as effectively as models like Gradient Boosting Machines (GBM), which often provide higher predictive accuracy. The trade-off in regulated contexts is between compliance and performance: while GBMs may perform better, they lack explainability, making simpler models more practical and acceptable in highly regulated environments.
