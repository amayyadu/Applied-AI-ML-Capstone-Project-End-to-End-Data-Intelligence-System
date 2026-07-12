# Applied AI & ML Capstone Project: End-to-End Data Intelligence System

## Dataset Justification
The German Credit (`credit-g`) dataset is a widely used benchmark dataset for machine learning. It contains both numerical and categorical attributes, making it suitable for exploratory data analysis, outlier detection, skewness analysis, correlation analysis, and predictive modeling.

There are no duplicate as well as null values in the data set.

Some features have very high positive skew others have mediocre negative or normal skew.

*   **High Positive Skew:** Some very high instances value (outliers) of that schema which deviated the mean and projects greater mean.

### Outlier Detection with IQR
*   Number of rows that fall outside these bounds for credit amount = 72
*   Number of rows that fall outside these bounds for duration = 70

---

## Direction and Approximate Strength of the Relationship of Scatterplot
The relationship moves in a positive (upward) direction. As the duration of the loan increases on the x-axis, the `credit_amount` on the y-axis generally rises as well. This confirms that longer payment windows are granted for larger credit pools. 

The strength of the relationship is moderate. While there is a clear upward trend, the data points do not form a perfectly tight line. The moderate positive relationship suggests duration may be an informative predictor of credit amount.

---

## Boxplot Analysis
*   Customers owning a telephone generally have a slightly higher median credit amount.
*   However, both groups contain several high-value outliers.
*   The overlap between boxes indicates that telephone ownership alone is not a strong predictor of loan amount.

---

## Heatmap Analysis
The heat map summarizes pairwise correlations among numerical variables.
*   Most correlations are weak to moderate.
*   Duration and `credit_amount` exhibit the strongest positive relationship.
*   No pair shows extremely high correlation, suggesting multicollinearity is limited.

### Top 3 pairs of $|\text{Spearman} - \text{Pearson}|$ values are:
1.  **age & num_dependents:** relationship is monotonic but non-linear.
2.  **duration & installment_commitment:** relationship is monotonic but non-linear.
3.  **credit_amount & installment_commitment:** relationship is approximately linear.

> Since several variable pairs exhibit monotonic but non-linear relationships, Spearman correlation will be used to guide feature selection in Part 2, as it is more robust to skewed distributions and non-linear monotonic relationships than Pearson correlation.

---

## Grouped Aggregation Analysis
Using `own_telephone` as the categorical variable (where 0 = No and 1 = Yes) and `credit_amount` as the numeric variable, the data reveals the following insights:

*   **(a) Highest Group Metrics:** The group with the highest mean credit amount and the highest standard deviation is Group 1 (Individuals who own a telephone).
*   **(b) Within-Group Variance Concern:** Yes, the high within-group standard deviation is a significant concern for a predictive model. Because the standard deviation within both groups is almost as large (or larger) than the group means themselves, it proves that the data points are highly scattered. This means the `own_telephone` feature alone is completely insufficient to predict the credit amount reliably for a specific individual.
*   **(c) Predictive Signal Ratio:** The ratio of the highest group mean to the lowest group mean is approximately 1.22. This ratio is not large enough to suggest that this feature carries a strong, independent predictive signal. While it shows a slight baseline trend (telephone owners request slightly higher loans on average), the variance is too massive and the mean difference is too small for a model to rely heavily on this column by itself.

# Credit Analysis: Regression & Classification Modeling

## 1. Feature Engineering & Preprocessing

### Target Labels
*   **Regression Label:** `credit_amount`
*   **Classification Label:** `class`

### Encoding Strategy
*   **Label Encoded Features:** `checking_status`, `savings_status`, and `employment` *(chosen because they have a natural order)*.
*   **One-Hot Encoded Features:** `purpose`, `housing`, `job`, `personal_status`, and `credit_history` *(chosen because they do not have any natural order)*.

> **Encoding Note:** Label encoding introduces a false ordinal relationship because it assigns integers (1, 2, 3), causing algorithms to mistakenly assume an order or rank where none exists. One-hot encoding avoids this by creating separate binary columns (0 or 1) for each category, treating them as completely independent and equal.

### Train-Test Split & Scaling
The dataset was split into 80% training and 20% testing data using `random_state=42`. `StandardScaler` was fitted only on the training set and then applied to both training and testing sets. Fitting on the full dataset would cause data leakage by exposing test-set statistics during training.

---

## 2. Regression Modeling (`credit_amount`)

### Model Performance
| Model | MSE | R² Score |
| :--- | :--- | :--- |
| **Linear Regression** | 2755317.5255 | 0.5471 |
| **Ridge Regression** | 2753983.4869 | 0.5473 |

### Key Concepts
*   **Model Coefficients:** The three features with the largest absolute coefficients have the greatest influence on the predicted credit amount. A large positive coefficient means a one-unit increase in the scaled feature increases the predicted credit amount by that coefficient value. A large negative coefficient means a one-unit increase in the scaled feature decreases the predicted credit amount by that coefficient value.
*   **Ridge Regression:** Ridge adds an L2 penalty that shrinks coefficient values, reducing overfitting and multicollinearity. The `alpha` parameter controls the strength of this penalty; a larger alpha results in greater coefficient shrinkage.

---

## 3. Classification Modeling (`class`)

### Class Imbalance & Evaluation Metrics
*   **Class Imbalance:** The minority class contained less than 35% of the training samples. Therefore, `class_weight='balanced'` was used in Logistic Regression to assign higher weight to the minority class without modifying the dataset.
*   **Precision:** TP / (TP + FP)
*   **Recall:** TP / (TP + FN)
*   **Metric Choice:** Recall is more important because misclassifying a bad credit applicant as good (false negative) is riskier than rejecting a good applicant (false positive).
*   **AUC:** An AUC of 0.7293 means the model has good ability to distinguish between good and bad credit applicants. A value closer to 1 indicates better class separation.
*   **Best Threshold:** The highest F1-score (0.8224) was obtained at a threshold of 0.30.

### Model Performance Comparison
| Model | Precision | Recall | AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (C=1.0)** | 0.8120 | 0.6738 | 0.7293 |
| **Logistic Regression (C=0.01)** | 0.8083 | 0.6879 | 0.7200 |

### Insights & Hyperparameters
*   **C Parameter:** C controls the strength of L2 regularization. A smaller C applies stronger regularization, shrinking model coefficients more to reduce overfitting.
*   **Comparison:** Reducing C from 1.0 to 0.01 slightly improved recall (0.6738 → 0.6879) but reduced precision (0.8120 → 0.8083) and lowered AUC (0.7293 → 0.7200). Overall, the baseline model (C=1.0) performed better on this dataset because it achieved higher precision and better class separation.

### Statistical Significance
*   **Bootstrap AUC Comparison:** Mean AUC Difference = 0.0091. 95% Confidence Interval = [-0.0086, 0.0287].
*   **Interpretation:** The 95% confidence interval includes 0, so the performance difference between the C=1.0 and C=0.01 models is not statistically reliable. Although the C=1.0 model has a slightly higher average AUC, its advantage may be due to sampling variation rather than a consistent improvement.


# Model Evaluation and Analysis Report

## 1. Decision Tree Performance & Overfitting

### Overfitting Analysis
* **Overfitting:** Yes. The Decision Tree achieved 100% training accuracy but only 70% test accuracy, indicating overfitting. The model memorized the training data and failed to generalize well to unseen data.
* **Model Variance:** Decision Trees are high-variance models because they greedily optimize each split without revisiting earlier decisions.

### Hyperparameter Tuning
* `max_depth`: Limits how deep the tree can grow, reducing model variance and overfitting.
* `min_samples_split`: Prevents splitting nodes with too few samples, reducing splits caused by noise.

### Model Comparison
* **Unconstrained Tree:** The unconstrained tree achieved 100% training accuracy but only 70% test accuracy, showing clear overfitting.
* **Controlled Tree:** The controlled tree achieved 78.25% training accuracy and 75.00% test accuracy, resulting in a much smaller train-test gap and better generalization to unseen data.

---

## 2. Splitting Criteria

### Mathematical Formulas
* **Gini Impurity:** 1 - Σ(pi²)
* **Entropy:** -Σ(pi log₂(pi))

### Pure Nodes (Gini = 0)
A Gini impurity of 0 means the node is completely pure. Every sample in that node belongs to the same class (either all good or all bad), so there is no uncertainty or mixing of classes. Such a node does not need to be split further because it already makes a perfect classification.

### Criterion Comparison
The Entropy criterion achieved a test accuracy of 0.7600, while the Gini criterion achieved 0.7550. Entropy performed slightly better on this dataset, but the difference is very small, indicating that both splitting criteria performed similarly.

---

## 3. Feature Importance & Random Forest Mechanics

### Top 5 Important Features
1. `credit_amount` – 0.1239
2. `checking_status` – 0.1044
3. `duration` – 0.0943
4. `age` – 0.0933
5. `employment` – 0.0526

### Feature Importance Definition
Random Forest computes feature importance by averaging the reduction in Gini impurity produced by each feature across all splits in all trees. Features that contribute more to creating purer nodes receive higher importance scores.

### Difference from Linear Regression
Feature importance indicates how useful a feature is for making tree splits, while a linear regression coefficient indicates the direction and magnitude of a feature's effect on the predicted value assuming a linear relationship.

### Bagging in Random Forest
Random Forest uses bootstrap sampling, where each tree is trained on a random sample of the training data selected with replacement, so different trees see slightly different datasets. At each split, only a random subset of approximately √(number of features) is considered instead of all features. Since each tree makes different errors, combining their predictions by voting averages out individual mistakes, reducing variance and overfitting compared to a single deep Decision Tree while improving generalization.

---

## 4. Dimensionality Reduction & Production Impact

### Results
* **Full Model ROC-AUC:** 0.7815
* **Reduced Model ROC-AUC:** 0.7954

### Interpretation
The reduced model achieved a slightly higher ROC-AUC than the full model, indicating that the removed features were largely uninformative and may have introduced noise. This suggests they were not contributing meaningful predictive information.

### Production Impact
A lower-dimensional model requires less memory, faster inference, and simpler maintenance. Such a model is preferable in production as long as the AUC does not decrease beyond an acceptable threshold. In this case, the AUC improved, so removing these features is beneficial.

---

## 5. Model Selection & Cross-Validation

### Cross-Validation Results

| | Model | Mean AUC | Std AUC |
|---|---|---|---|
| 0 | Logistic Regression | 0.7469 | 0.0295 |
| 1 | Decision Tree (Depth=5) | 0.7018 | 0.0203 |
| 2 | Random Forest | 0.7884 | 0.0307 |
| 3 | Gradient Boosting | 0.7755 | 0.0184 |

### Interpretation
Cross-validation provides a more reliable estimate of generalization performance because the model is evaluated on five different train-test splits instead of one. This reduces the effect of a lucky or unlucky split. The mean ROC-AUC measures the model's average performance across all folds, while the standard deviation indicates how consistent the model's performance is across different splits.

---

## 6. Hyperparameter Optimization & Learning Curves

### Configurations Evaluated
* **Total Parameter Combinations:** 3 × 3 × 2 = 18 parameter combinations. 
* **Total Models Trained:** With 5-fold cross-validation, a total of 90 models were trained.

### Grid Search vs Randomized Search
* **Grid Search:** Tests every parameter combination and finds the best model but is computationally expensive.
* **Randomized Search:** Evaluates only a random subset of combinations, making it faster for large search spaces.

### Learning Curve Insights
* **Training AUC:** The training AUC decreases slightly as the training set grows, indicating the model overfits less when more data is available.
* **Test AUC:** The test AUC consistently increases from 0.7170 to 0.8001 as the training data grows, showing that the model generalizes better with more data.
* **Conclusion:** The test AUC is still improving at 100% of the training data, so the model appears to be data-limited. Collecting more training data would likely improve performance further.

---

## 7. Model Comparison & Recommendation

### Model Comparison Matrix

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.7469 | 0.0295 | 0.7293 |
| Decision Tree (Depth=5) | 0.7018 | 0.0203 | 0.7600 |
| Random Forest | 0.7884 | 0.0307 | 0.7815 |
| Gradient Boosting | 0.7755 | 0.0184 | 0.8188 |

### Recommendation
I recommend the Gradient Boosting Classifier. It achieved the highest test ROC-AUC (0.8188), indicating the best ability to distinguish between good and bad credit risks on unseen data. Although Random Forest had the highest cross-validation mean AUC, Gradient Boosting performed better on the held-out test set while also having a lower cross-validation standard deviation, indicating more consistent performance. Overall, Gradient Boosting provides the best balance of predictive accuracy and generalization for this dataset.


# Track C: Model Prediction Explanation Pipeline

Load the best-performing model from Part 3 using `joblib.load('best_model.pkl')`. For three hand-crafted feature-vector inputs, call `.predict()` and `.predict_proba()`. For each input, pass the feature values, predicted class, and predicted probability to the LLM API as a structured user prompt and request a JSON explanation with at least 5 required scalar fields (e.g., `{"prediction_label": "string", "confidence_level": "low|medium|high", "top_reason": "string", "second_reason": "string", "next_step": "string"}`). Validate each JSON response against a schema and apply the PII guardrail before each LLM call.

## System Prompt

```text
You are an AI assistant that explains credit risk model predictions.
You will receive:

- Customer feature values.
- The predicted class (Good Credit or Bad Credit).
- The predicted probability.

Your task is to explain the prediction and return ONLY a valid JSON object.

The JSON must contain exactly these fields:
{
  "prediction_label": "Good Credit | Bad Credit",
  "confidence_level": "High | Medium | Low",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}

Rules:
- Return only valid JSON.
- Do not include markdown or any extra text.
- Base the explanation only on the provided feature values and prediction.
- Keep each explanation concise and professional.
- If the predicted probability is greater than or equal to 0.80, set "confidence_level" to "High".
- If the predicted probability is between 0.60 and 0.79, set "confidence_level" to "Medium".
- Otherwise, set "confidence_level" to "Low".
```

## User Prompt Template

```text
Feature Values:
{json.dumps(important_features, indent=2)}

Predicted Class:
{label}

Predicted Probability:
{probability:.4f}

Explain this prediction and return ONLY a valid JSON object with the following fields:

{
"prediction_label": "...",
"confidence_level": "...",
"top_reason": "...",
"second_reason": "...",
"next_step": "..."
}
```

## Temperature

The LLM was run with **temperature = 0** to produce deterministic and consistent outputs. A low temperature minimizes randomness, ensuring the same input generates the same structured JSON response. This is appropriate for structured prediction explanations where reproducibility and valid JSON output are more important than creativity.

## Results

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
|---------------|-----------------|------------:|------------------|-------------------|
| checking_status=2, duration=24, credit_amount=4500, savings_status=2, employment=4, age=35, existing_credits=1, num_dependents=1 | Good Credit | 0.5942 | `{"prediction_label":"Good Credit","confidence_level":"Medium","top_reason":"Stable employment and moderate credit amount suggest reliable repayment capacity.","second_reason":"Positive checking and savings status indicate good financial habits.","next_step":"Proceed with a standard credit offer while monitoring repayment behavior."}` | Valid |
| checking_status=0, duration=48, credit_amount=9000, savings_status=2, employment=1, age=23, existing_credits=1, num_dependents=1 | Good Credit | 0.5551 | `{"prediction_label":"Good Credit","confidence_level":"Low","top_reason":"Moderate credit amount and short loan duration are viewed favorably by the model.","second_reason":"Employment status and savings indicate financial stability, while few existing credits and dependents reduce risk.","next_step":"Proceed with approval but monitor repayment history closely."}` | Valid |
| checking_status=3, duration=6, credit_amount=1500, savings_status=2, employment=4, age=45, existing_credits=1, num_dependents=1 | Good Credit | 0.7601 | `{"prediction_label":"Good Credit","confidence_level":"Medium","top_reason":"Short loan duration (6 months) and moderate credit amount indicate low risk.","second_reason":"Stable employment and few dependents support repayment capacity.","next_step":"Approve the loan and offer a competitive interest rate."}` | Valid |
