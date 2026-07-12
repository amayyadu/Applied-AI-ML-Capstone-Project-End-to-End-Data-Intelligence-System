import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("cleaned_data.csv")

# 1. Map Ordinal & Class columns
df["class"] = df["class"].map({"good": 1, "bad": 0})
df["checking_status"] = df["checking_status"].map({"no checking": 0, "<0": 1, "0<=X<200": 2, ">=200": 3})
df["savings_status"] = df["savings_status"].map({"no known savings": 0, "<100": 1, "100<=X<500": 2, "500<=X<1000": 3, ">=1000": 4})
df["employment"] = df["employment"].map({"unemployed": 0, "<1": 1, "1<=X<4": 2, "4<=X<7": 3, ">=7": 4})

# --- SEPARATE TARGETS BEFORE DUMMIES TO PREVENT LEAKAGE ---
y_reg = df["credit_amount"]
y_clf = df["class"]

# 2. Extract feature sets *before* calling get_dummies
X_reg_raw = df.drop(columns=["credit_amount"])
X_clf_raw = df.drop(columns=["class"])

nominal_cols = [
    "purpose", "credit_history", "personal_status", "other_parties",
    "property_magnitude", "other_payment_plans", "housing", "job",
    "own_telephone", "foreign_worker"
]

# 3. Apply get_dummies cleanly to separate feature matrices
X_reg = pd.get_dummies(X_reg_raw, columns=nominal_cols, drop_first=True)
X_clf = pd.get_dummies(X_clf_raw, columns=nominal_cols, drop_first=True)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

mse=round(mean_squared_error(y_test_reg, y_pred_reg), 4)
r2= round(r2_score(y_test_reg, y_pred_reg), 4)
print("Mean Squared Error (MSE):",mse)
print("R² Score:",r2)

coef_df = pd.DataFrame({
    "Feature": X_reg.columns,
    "Coefficient": reg_model.coef_
})
print("Model Coefficients:")
print(coef_df.sort_values(by="Coefficient", key=abs, ascending=False).head(3))

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Train Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_reg, y_train_reg)

# Predict
y_pred_ridge = ridge_model.predict(X_test_reg)

# Evaluation
ridge_mse = round(mean_squared_error(y_test_reg, y_pred_ridge),4)
ridge_r2 = round(r2_score(y_test_reg, y_pred_ridge),4)
print("Ridge Regression")
print("MSE:", ridge_mse)
print("R² Score:",ridge_r2)

# Comparison Table
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MSE": [mse, ridge_mse],
    "R² Score": [r2, ridge_r2]
})
print(comparison.to_string(float_format=lambda x: f'{x:.4f}'))

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

print("Class Distribution:")
print(y_train_clf.value_counts())
print("\nPercentage Distribution:")
print((y_train_clf.value_counts(normalize=True) * 100).round(2))

# Logistic Regression with class imbalance handling
from sklearn.linear_model import LogisticRegression

scaler_clf = StandardScaler()

X_train_clf = scaler_clf.fit_transform(X_train_clf)
X_test_clf = scaler_clf.transform(X_test_clf)

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

log_model.fit(X_train_clf, y_train_clf)

y_pred_clf = log_model.predict(X_test_clf)

y_prob_clf = log_model.predict_proba(X_test_clf)[:, 1]

print("Predicted Labels:")
print(y_pred_clf[:10])

print("\nPredicted Probabilities:")
print(y_prob_clf[:10])

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print("\nAccuracy:", round(accuracy, 4))

# Precision, Recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test_clf, y_pred_clf))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Compute ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_clf, y_prob_clf)

# Compute AUC
auc = roc_auc_score(y_test_clf, y_prob_clf)

print("AUC Score:", round(auc, 4))

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")

# Annotate AUC on the plot
plt.text(0.60, 0.10, f"AUC = {auc:.3f}", fontsize=11,
         bbox=dict(facecolor="white", alpha=0.8))

plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

results = []

for threshold in thresholds:
    y_pred_threshold = (y_prob_clf >= threshold).astype(int)

    precision = precision_score(y_test_clf, y_pred_threshold)
    recall = recall_score(y_test_clf, y_pred_threshold)
    f1 = f1_score(y_test_clf, y_pred_threshold)

    results.append({
        "Threshold": threshold,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    })

threshold_df = pd.DataFrame(results)

print(threshold_df)

log_model_c = LogisticRegression(
    C=0.01,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

log_model_c.fit(X_train_clf, y_train_clf)

# Predictions
y_pred_c = log_model_c.predict(X_test_clf)
y_prob_c = log_model_c.predict_proba(X_test_clf)[:, 1]

# Metrics
comparison = pd.DataFrame({
    "Model": ["Logistic Regression (C=1.0)", "Logistic Regression (C=0.01)"],
    "Precision": [
        precision_score(y_test_clf, y_pred_clf),
        precision_score(y_test_clf, y_pred_c)
    ],
    "Recall": [
        recall_score(y_test_clf, y_pred_clf),
        recall_score(y_test_clf, y_pred_c)
    ],
    "AUC": [
        roc_auc_score(y_test_clf, y_prob_clf),
        roc_auc_score(y_test_clf, y_prob_c)
    ]
})

print(comparison.round(4))

y_true = np.array(y_test_clf)
prob_c1 = np.array(y_prob_clf)
prob_c = np.array(y_prob_c)

n_bootstrap = 500
auc_diffs = []

np.random.seed(42)

for _ in range(n_bootstrap):
    idx = np.random.choice(len(y_true), size=len(y_true), replace=True)

    # Skip samples containing only one class
    if len(np.unique(y_true[idx])) < 2:
        continue

    auc_c1 = roc_auc_score(y_true[idx], prob_c1[idx])
    auc_c = roc_auc_score(y_true[idx], prob_c[idx])

    auc_diffs.append(auc_c1 - auc_c)

auc_diffs = np.array(auc_diffs)

mean_diff = auc_diffs.mean()
lower_ci = np.percentile(auc_diffs, 2.5)
upper_ci = np.percentile(auc_diffs, 97.5)

print(f"Mean AUC Difference: {mean_diff:.4f}")
print(f"95% Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")

if lower_ci > 0 or upper_ci < 0:
    print("The 95% confidence interval excludes zero.")
else:
    print("The 95% confidence interval includes zero.")
