Regression label is "credit_amount" and Classification label is "class"
Label Encoded features : checking_status, savings_status, and employment, as they have a natural order
one-hot encoded features : purpose, housing, job, personal_status, and credit_history as they do not have any natural order
Label encoding introduces a false ordinal relationship because it assigns integers (1, 2, 3), causing algorithms to mistakenly assume an order or rank where none exists.One-hot encoding avoids this by creating separate binary columns (0 or 1) for each category, treating them as completely independent and equal.
Train-Test Split & Scaling:
The dataset was split into 80% training and 20% testing data using random_state=42. StandardScaler was fitted only on the training set and then applied to both training and testing sets. Fitting on the full dataset would cause data leakage by exposing test-set statistics during training.
Model Coefficients: The three features with the largest absolute coefficients have the greatest influence on the predicted credit amount. A large positive coefficient means a one-unit increase in the scaled feature increases the predicted credit amount by that coefficient value. A large negative coefficient means a one-unit increase in the scaled feature decreases the predicted credit amount by that coefficient value.
               Model          MSE  R² Score
0  Linear Regression 2755317.5255    0.5471
1   Ridge Regression 2753983.4869    0.5473
Ridge Regression: Ridge adds an L2 penalty that shrinks coefficient values, reducing overfitting and multicollinearity. The alpha parameter controls the strength of this penalty; a larger alpha results in greater coefficient shrinkage.
Classification model — Logistic Regression:
Class Imbalance: The minority class contained less than 35% of the training samples. Therefore, class_weight='balanced' was used in Logistic Regression to assign higher weight to the minority class without modifying the dataset.
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
Metric Choice: Recall is more important because misclassifying a bad credit applicant as good (false negative) is riskier than rejecting a good applicant (false positive).
AUC: An AUC of 0.7293 means the model has good ability to distinguish between good and bad credit applicants. A value closer to 1 indicates better class separation.
Best Threshold: The highest F1-score (0.8224) was obtained at a threshold of 0.30 .
                          Model  Precision  Recall     AUC
0   Logistic Regression (C=1.0)     0.8120  0.6738  0.7293
1  Logistic Regression (C=0.01)     0.8083  0.6879  0.7200
C Parameter: C controls the strength of L2 regularization. A smaller C applies stronger regularization, shrinking model coefficients more to reduce overfitting.
Comparison: Reducing C from 1.0 to 0.01 slightly improved recall (0.6738 → 0.6879) but reduced precision (0.8120 → 0.8083) and lowered AUC (0.7293 → 0.7200). Overall, the baseline model (C=1.0) performed better on this dataset because it achieved higher precision and better class separation.
Bootstrap AUC Comparison: Mean AUC Difference = 0.0091. 95% Confidence Interval = [-0.0086, 0.0287].
Interpretation: The 95% confidence interval includes 0, so the performance difference between the C=1.0 and C=0.01 models is not statistically reliable. Although the C=1.0 model has a slightly higher average AUC, its advantage may be due to sampling variation rather than a consistent improvement.
