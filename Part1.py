import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
credit = fetch_openml(name="credit-g",as_frame=True)
df = credit.frame

print(df.head(5))

df.info()

df.shape

null_table = pd.DataFrame({
    "Null Count": df.isnull().sum(),
    "Null Percentage": (df.isnull().sum()/len(df))*100
})
print(null_table)

#0 dupliates
df.duplicated().sum()

#converting own_telephone as type int
print(df['own_telephone'].unique())
df['own_telephone'] = (df['own_telephone'] == 'yes').astype(int)
df['own_telephone'].unique()

df.info()

desc=df.describe()
numeric_cols = desc.columns
skd=pd.DataFrame([df[numeric_cols].skew()], index=['skew'])
print(skd)
sort_val_skd=skd.sort_values(by='skew', axis=1, ascending=False)
print(sort_val_skd)


#credit_amount is most skewed in terms of mean as it has very high values which shifts the actual mean

Q1=df['credit_amount'].quantile(0.25)
Q3=df['credit_amount'].quantile(0.75)
IQR=Q3-Q1
lower_bound= Q1 - (1.5 * IQR)
upper_bound= Q3 + (1.5 * IQR)
print("Upper bounds for outliers:\n", upper_bound)
print("Lower bounds for outliers:\n", lower_bound)
count=((df['credit_amount'] < lower_bound) | (df['credit_amount'] > upper_bound)).sum()
print(f'number of rows that fall outside these bounds : {count}')

Q1=df['duration'].quantile(0.25)
Q3=df['duration'].quantile(0.75)
IQR=Q3-Q1
lower_bound= Q1 - (1.5 * IQR)
upper_bound= Q3 + (1.5 * IQR)
print("Upper bounds for outliers:\n", upper_bound)
print("Lower bounds for outliers:\n", lower_bound)
count=((df['duration'] < lower_bound) | (df['duration'] > upper_bound)).sum()
print(f'number of rows that fall outside these bounds : {count}')

plt.plot(df.index, df['credit_amount'], linewidth=1.5, label='Credit Amount')
plt.title('Credit Amount Distribution Over Row Index', fontsize=14, fontweight='bold')
plt.xlabel(' Index ', fontsize=12)
plt.ylabel('Credit Amount ', fontsize=12)

grouped_data = df.groupby('own_telephone')['credit_amount'].mean()
grouped_data.index

plt.bar(grouped_data.index.astype(str), grouped_data.values, width=0.6)
plt.title('Average Credit Amount by Telephone Ownership Status', fontsize=14, fontweight='bold')
plt.xlabel('Owns Telephone? (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Mean Credit Amount ', fontsize=12)


sns.histplot(data=df, x='credit_amount', bins=20)
plt.title('Distribution of Credit Amount (Highly Skewed)', fontsize=14, fontweight='bold')

sns.scatterplot(data=df, x='duration', y='credit_amount', color='#2c3e50', alpha=0.6, edgecolor='w', s=50)
plt.title("Relationship Between Loan duration and Credit Amount", fontsize=14, fontweight='bold')

sns.boxplot(data=df, x='own_telephone', y='credit_amount', palette='Set2', width=0.5)
plt.title('Comparison of Credit Amount by Telephone Ownership', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Owns Telephone (0 = No, 1 = Yes)', fontsize=12, labelpad=10)
plt.ylabel('Credit Amount ($)', fontsize=12, labelpad=10)
plt.show()


sns.heatmap(df[desc.columns].corr())
plt.title("Correlation Matrix")
plt.show()

skew_cols = ["credit_amount", "num_dependents"]
for col in skew_cols:
    col_mean = df[col].mean()
    col_median = df[col].median()
    print(f"{col} -> Mean: {col_mean:.2f} | Median: {col_median:.2f}")


numeric_cols = desc.columns
pearson_mat = df[numeric_cols].corr(method="pearson")
spearman_mat = df[numeric_cols].corr(method="spearman")
print("\n--- Pearson Correlation Matrix ---")
print(pearson_mat.round(3))

print("\n--- Spearman Correlation Matrix ---")
print(spearman_mat.round(3))

# Compute absolute difference matrix
diff_matrix = (spearman_mat - pearson_mat).round(3)

pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1, col2 = numeric_cols[i], numeric_cols[j]
        pairs.append(
            {
                "Pair": f"{col1} & {col2}",
                "Pearson": pearson_mat.loc[col1, col2],
                "Spearman": spearman_mat.loc[col1, col2],
                "Abs_Diff": diff_matrix.loc[col1, col2],
            }
        )
diff_table = pd.DataFrame(pairs).sort_values(by="Abs_Diff", ascending=False)
print(diff_table.to_string(index=False))



grouped_res = df.groupby("own_telephone")["credit_amount"].agg(
    ["mean", "std", "count"]
)
print(grouped_res.round(2))


highest_mean = grouped_res["mean"].max()
lowest_mean = grouped_res["mean"].min()
mean_ratio = highest_mean / lowest_mean
print(f"\nRatio of Highest Mean to Lowest Mean: {mean_ratio:.2f}")


df.to_csv("cleaned_data.csv", index=False)
