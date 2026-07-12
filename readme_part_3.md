Overfitting: Yes. The Decision Tree achieved 100% training accuracy but only 70% test accuracy, indicating overfitting. The model memorized the training data and failed to generalize well to unseen data. Decision Trees are high-variance models because they greedily optimize each split without revisiting earlier decisions.
max_depth: Limits how deep the tree can grow, reducing model variance and overfitting.
min_samples_split: Prevents splitting nodes with too few samples, reducing splits caused by noise.
Comparison: The unconstrained tree achieved 100% training accuracy but only 70% test accuracy, showing clear overfitting. The controlled tree achieved 78.25% training accuracy and 75.00% test accuracy, resulting in a much smaller train-test gap and better generalization to unseen data.
Gini Impurity: 1 - Σ(pi²)
Entropy: -Σ(pi log₂(pi))
Gini = 0: A Gini impurity of 0 means the node is completely pure. Every sample in that node belongs to the same class (either all good or all bad), so there is no uncertainty or mixing of classes. Such a node does not need to be split further because it already makes a perfect classification.
Comparison: The Entropy criterion achieved a test accuracy of 0.7600, while the Gini criterion achieved 0.7550. Entropy performed slightly better on this dataset, but the difference is very small, indicating that both splitting criteria performed similarly.
