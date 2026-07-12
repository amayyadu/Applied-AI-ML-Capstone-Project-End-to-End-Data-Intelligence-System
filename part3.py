from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train_clf, y_train_clf)

# Predictions
y_train_pred_dt = dt_model.predict(X_train_clf)
y_test_pred_dt = dt_model.predict(X_test_clf)

# Accuracy
train_acc = accuracy_score(y_train_clf, y_train_pred_dt)
test_acc = accuracy_score(y_test_clf, y_test_pred_dt)

print("Decision Tree Classifier")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

dt_controlled = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

dt_controlled.fit(X_train_clf, y_train_clf)

# Predictions
y_train_pred_controlled = dt_controlled.predict(X_train_clf)
y_test_pred_controlled = dt_controlled.predict(X_test_clf)

# Accuracy
train_acc_controlled = accuracy_score(y_train_clf, y_train_pred_controlled)
test_acc_controlled = accuracy_score(y_test_clf, y_test_pred_controlled)

print("Controlled Decision Tree")
print(f"Training Accuracy: {train_acc_controlled:.4f}")
print(f"Test Accuracy: {test_acc_controlled:.4f}")

dt_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

dt_gini.fit(X_train_clf, y_train_clf)
y_pred_gini = dt_gini.predict(X_test_clf)
gini_acc = accuracy_score(y_test_clf, y_pred_gini)

# Entropy Decision Tree
dt_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

dt_entropy.fit(X_train_clf, y_train_clf)
y_pred_entropy = dt_entropy.predict(X_test_clf)
entropy_acc = accuracy_score(y_test_clf, y_pred_entropy)

comparison = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Test Accuracy": [gini_acc, entropy_acc]
})

print(comparison.round(4))

from sklearn.ensemble import RandomForestClassifier
# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train_clf, y_train_clf)

# Predictions
y_train_pred_rf = rf_model.predict(X_train_clf)
y_test_pred_rf = rf_model.predict(X_test_clf)
y_prob_rf = rf_model.predict_proba(X_test_clf)[:, 1]

# Metrics
train_acc_rf = accuracy_score(y_train_clf, y_train_pred_rf)
test_acc_rf = accuracy_score(y_test_clf, y_test_pred_rf)
auc_rf = roc_auc_score(y_test_clf, y_prob_rf)

print("Random Forest Classifier")
print(f"Training Accuracy: {train_acc_rf:.4f}")
print(f"Test Accuracy: {test_acc_rf:.4f}")
print(f"ROC-AUC: {auc_rf:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X_clf.columns,
    "Importance": rf_model.feature_importances_
})

top5_features = feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(5)

print("Top 5 Important Features:")
print(top5_features)

from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train_clf, y_train_clf)

# Predictions
y_train_pred_gb = gb_model.predict(X_train_clf)
y_test_pred_gb = gb_model.predict(X_test_clf)
y_prob_gb = gb_model.predict_proba(X_test_clf)[:, 1]

# Metrics
train_acc_gb = accuracy_score(y_train_clf, y_train_pred_gb)
test_acc_gb = accuracy_score(y_test_clf, y_test_pred_gb)
auc_gb = roc_auc_score(y_test_clf, y_prob_gb)

print("Gradient Boosting Classifier")
print(f"Training Accuracy: {train_acc_gb:.4f}")
print(f"Test Accuracy: {test_acc_gb:.4f}")
print(f"ROC-AUC: {auc_gb:.4f}")

# Find 5 least important features
importance_df = pd.DataFrame({
    "Feature": X_clf.columns,
    "Importance": rf_model.feature_importances_
})

least5 = importance_df.sort_values(
    by="Importance",
    ascending=True
).head(5)

print("5 Least Important Features:")
print(least5)

least_features = least5["Feature"].tolist()

# Remove them from train and test sets
X_train_reduced = pd.DataFrame(
    X_train_clf,
    columns=X_clf.columns
).drop(columns=least_features)

X_test_reduced = pd.DataFrame(
    X_test_clf,
    columns=X_clf.columns
).drop(columns=least_features)

# Train reduced Random Forest
rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_reduced.fit(X_train_reduced, y_train_clf)

# Predict probabilities
y_prob_reduced = rf_reduced.predict_proba(X_test_reduced)[:, 1]

# Compute AUC
auc_reduced = roc_auc_score(y_test_clf, y_prob_reduced)

print(f"\nFull Model ROC-AUC: {auc_rf:.4f}")
print(f"Reduced Model ROC-AUC: {auc_reduced:.4f}")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Logistic Regression Pipeline
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", log_model)
])

# Cross-validation
log_scores = cross_val_score(
    log_pipeline,
    X_clf,
    y_clf,
    cv=cv,
    scoring="roc_auc"
)

dt_scores = cross_val_score(
    dt_controlled,
    X_clf,
    y_clf,
    cv=cv,
    scoring="roc_auc"
)

rf_scores = cross_val_score(
    rf_model,
    X_clf,
    y_clf,
    cv=cv,
    scoring="roc_auc"
)

gb_scores = cross_val_score(
    gb_model,
    X_clf,
    y_clf,
    cv=cv,
    scoring="roc_auc"
)

cv_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree (Depth=5)",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Mean AUC": [
        log_scores.mean(),
        dt_scores.mean(),
        rf_scores.mean(),
        gb_scores.mean()
    ],
    "Std AUC": [
        log_scores.std(),
        dt_scores.std(),
        rf_scores.std(),
        gb_scores.std()
    ]
})

print(cv_results.round(4))

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split

X_train_clf_raw, X_test_clf_raw, y_train_clf, y_test_clf = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42
)

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# Parameter Grid
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]
}

# 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

# Fit on the ORIGINAL (unscaled) training data
grid_search.fit(X_train_clf_raw, y_train_clf)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Mean ROC-AUC:")
print(round(grid_search.best_score_, 4))

from sklearn.metrics import roc_auc_score
import pandas as pd

best_pipeline = grid_search.best_estimator_

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
results = []

for f in fractions:

    n = int(f * len(X_train_clf_raw))

    X_train_subset = X_train_clf_raw.iloc[:n]
    y_train_subset = y_train_clf.iloc[:n]

    # Train on subset
    best_pipeline.fit(X_train_subset, y_train_subset)

    # Training AUC
    train_prob = best_pipeline.predict_proba(X_train_subset)[:, 1]
    train_auc = roc_auc_score(y_train_subset, train_prob)

    # Test AUC
    test_prob = best_pipeline.predict_proba(X_test_clf_raw)[:, 1]
    test_auc = roc_auc_score(y_test_clf, test_prob)

    results.append({
        "Training Fraction": f,
        "Training AUC": round(train_auc, 4),
        "Test AUC": round(test_auc, 4)
    })
learning_curve = pd.DataFrame(results)
print(learning_curve)

import joblib
joblib.dump(best_pipeline, "best_model.pkl")
print("Model saved successfully as best_model.pkl")
# Load saved model
loaded_model = joblib.load("best_model.pkl")
# Create two handcrafted test rows
sample_data = X_test_clf_raw.iloc[:2].copy()
# Predict
predictions = loaded_model.predict(sample_data)
print("Predictions:")
print(predictions)
