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
