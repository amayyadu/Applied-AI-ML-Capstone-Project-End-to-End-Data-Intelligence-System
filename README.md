# Applied-AI-ML-Capstone-Project-End-to-End-Data-Intelligence-System
 A complete, end-to-end data intelligence system — from raw data ingestion and exploratory analysis through supervised machine learning, advanced ensemble modeling and pipelines, to a final LLM-powered feature that adds a structured-data JSON analysis or model explanation capability on top of the insights already produced.
I have loaded 'credit-g' dataset from sklearn.datasets itself, The objective of the dataset is to determine whether an applicant represents a good or bad credit risk.
There are no dupplicate as well as null values in the data set
Some features have very high posive skew others have mediocre negative or normal skew
HIGH POSITIVE SKEW : Some very high instances value (outliers) of that scheema which deviated the mean and projects greater mean 
Outlier detection with IQR: 
Number of rows that fall outside these bounds for credit amount = 72 
Number of rows that fall outside these bounds for duration : 70
Direction and approximate strength of the relationship of scatterplot : 
The relationship moves in a positive (upward) direction. As the duration of the loan increases on the x-axis, the credit_amount on the y-axis generally rises as well. This confirms that longer payment windows are granted for larger credit pools.The strength of the relationship is moderate. While there is a clear upward trend, the data points do not form a perfectly tight line.The moderate positive relationship suggests duration may be an informative predictor of credit amount.
BOXPLOT:
Customers owning a telephone generally have a slightly higher median credit amount.However, both groups contain several high-value outliers.The overlap between boxes indicates that telephone ownership alone is not a strong predictor of loan amount.

HEATMAP:
The heat map summarizes pairwise correlations among numerical variables.
Most correlations are weak to moderate.
Duration and credit_amount exhibit the strongest positive relationship.
No pair shows extremely high correlation, suggesting multicollinearity is limited.

Top 3 pairs of  |Spearman − Pearson| values are :
i) age & num_dependents : relationship is monotonic but non-linear, 
which relation to be used in PART 2
ii) duration & installment_commitment : relationship is monotonic but non-linear, 
which relation to be used in PART 2
iii) credit_amount & installment_commitment : relationship is approximately linear, 
which relation to be used in PART 2

Grouped Aggregation Analysis
Using own_telephone as the categorical variable (where 0 = No and 1 = Yes) and credit_amount as the numeric variable, the data reveals the following insights:
(a) Highest Group Metrics: The group with the highest mean credit amount and the highest standard deviation is Group 1 (Individuals who own a telephone).
(b) Within-Group Variance Concern: Yes, the high within-group standard deviation is a significant concern for a predictive model. Because the standard deviation within both groups is almost as large (or larger) than the group means themselves, it proves that the data points are highly scattered. This means the own_telephone feature alone is completely insufficient to predict the credit amount reliably for a specific individual.
(c) Predictive Signal Ratio: The ratio of the highest group mean to the lowest group mean is approximately 1.22. This ratio is not large enough to suggest that this feature carries a strong, independent predictive signal. While it shows a slight baseline trend (telephone owners request slightly higher loans on average), the variance is too massive and the mean difference is too small for a model to rely heavily on this column by itself.
