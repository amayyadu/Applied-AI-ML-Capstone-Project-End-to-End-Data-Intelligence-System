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
