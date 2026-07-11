# Applied-AI-ML-Capstone-Project-End-to-End-Data-Intelligence-System
 A complete, end-to-end data intelligence system — from raw data ingestion and exploratory analysis through supervised machine learning, advanced ensemble modeling and pipelines, to a final LLM-powered feature that adds a structured-data JSON analysis or model explanation capability on top of the insights already produced.
I have loaded 'credit-g' dataset from sklearn.datasets itself
There are no dupplicate as well as null values in the data set
Some features have very high posive skew others have mediocre negative or normal skew
HIGH POSITIVE SKEW : Some very high instances value (outliers) of that scheema which deviated the mean and projects greater mean 
Outlier detection with IQR: 
Number of rows that fall outside these bounds for credit amount = 72 
Number of rows that fall outside these bounds for duration : 70
Direction and approximate strength of the relationship of scatterplot : 
The relationship moves in a positive (upward) direction. As the duration of the loan increases on the x-axis, the credit_amount on the y-axis generally rises as well. This confirms that longer payment windows are granted for larger credit pools.
The strength of the relationship is moderate. While there is a clear upward trend, the data points do not form a perfectly tight line.
Top 3 pairs of  |Spearman − Pearson| values are :
i) age & num_dependents : relationship is monotonic but non-linear, 
which relation to be used in PART 2
ii) duration & installment_commitment : relationship is monotonic but non-linear, 
which relation to be used in PART 2
iii) credit_amount & installment_commitment : relationship is approximately linear, 
which relation to be used in PART 2

