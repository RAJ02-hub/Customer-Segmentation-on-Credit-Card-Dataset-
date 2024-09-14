# Credit Card Customer Segmentation Project

## Overview
This project focuses on customer segmentation using credit card transaction data from around 9000 active credit card users. The objective is to group customers into segments based on their credit card usage patterns over the last six months. By clustering customers with similar behaviors, businesses can tailor marketing strategies to meet the needs of each customer group, thereby improving customer satisfaction and business efficiency.

## Problem Statement
In the financial sector, understanding customer behavior is crucial for delivering personalized services, reducing churn, and improving customer satisfaction. The goal of this project is to identify different segments of credit card users based on their transaction and payment behavior. This segmentation will help financial institutions develop more effective strategies for targeting each group, improving product offerings, and creating more personalized marketing campaigns.

## Dataset
The dataset used in this project contains 18 variables summarizing the behavior of each customer. Here’s the data dictionary:

### Data Dictionary:
- **CUSTID**: Unique identification of the credit card holder.
- **BALANCE**: Current balance on the credit card.
- **BALANCEFREQUENCY**: How frequently the balance is updated (score between 0 and 1, where 1 = frequently updated).
- **PURCHASES**: Total value of purchases made using the credit card.
- **ONEOFFPURCHASES**: Maximum value of a single purchase made using the credit card.
- **INSTALLMENTSPURCHASES**: Total value of purchases made in installments.
- **CASHADVANCE**: Cash advances taken by the user.
- **PURCHASESFREQUENCY**: Frequency of purchases made (score between 0 and 1).
- **ONEOFFPURCHASESFREQUENCY**: Frequency of one-off purchases made.
- **PURCHASESINSTALLMENTSFREQUENCY**: Frequency of purchases made in installments.
- **CASHADVANCEFREQUENCY**: Frequency of cash advances taken.
- **CASHADVANCETRX**: Number of cash advance transactions.
- **PURCHASESTRX**: Number of purchase transactions.
- **CREDITLIMIT**: The credit limit of the card holder.
- **PAYMENTS**: Total payments made by the customer.
- **MINIMUM_PAYMENTS**: Minimum payments made by the customer.
- **PRCFULLPAYMENT**: Percentage of months in which the customer made a full payment.
- **TENURE**: Tenure of the credit card service in months.

## Technologies Used
- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Array and numerical computation
- **Matplotlib & Seaborn**: Data visualization
- **scikit-learn**: Machine learning (K-Means clustering, PCA)

## Project Workflow

### 1. Data Preprocessing
#### a. Missing Value Imputation
The dataset contained missing values in the `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` columns. To handle this, we used the mean of the respective columns to fill these missing values. This ensures that the analysis is not biased by missing data.

```python
customer.loc[(customer['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=customer['MINIMUM_PAYMENTS'].mean()
customer.loc[(customer['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=customer['CREDIT_LIMIT'].mean()
```

#### b. Outlier Detection and Treatment
Many of the columns, such as `BALANCE`, `PURCHASES`, and `CREDIT_LIMIT`, contained outliers. To preserve the integrity of the dataset, instead of removing outliers, we binned continuous variables into discrete ranges. This minimizes the influence of extreme values while maintaining the dataset's size.

Example of binning `BALANCE` into ranges:
```python
df.loc[((df['BALANCE']>0)&(df['BALANCE']<=500)), 'BALANCE_RANGE'] = 1
df.loc[((df['BALANCE']>500)&(df['BALANCE']<=1000)), 'BALANCE_RANGE'] = 2
```

#### c. Feature Engineering
Several features were transformed into ranges or bins to simplify analysis and reduce the impact of outliers. Frequency-based features like `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASESFREQUENCY`, and `CASHADVANCE_FREQUENCY` were converted to ranges. 

For instance, `PURCHASES_FREQUENCY` was categorized into discrete bins ranging from low to high frequency. This approach allows us to classify customer behaviors in a more interpretable manner.

### 2. Data Normalization
All features were standardized using `StandardScaler` to ensure that each feature contributed equally to the clustering algorithm. This was essential because the raw data had varying magnitudes (e.g., `BALANCE` vs `PRCFULLPAYMENT`).

```python
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(df)
```

### 3. Clustering with K-Means
K-Means clustering was selected for segmenting customers into different groups. To find the optimal number of clusters, the **elbow method** was used, which indicated that six clusters were optimal.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=44, n_init=15, max_iter=300)
kmeans.fit(X)
labels = kmeans.labels_
```

After assigning labels to each customer, the dataset was augmented with a new `cluster` column, which indicates the cluster to which each customer belongs.

### 4. Dimensionality Reduction using PCA
To visualize the clusters, **Principal Component Analysis (PCA)** was applied to reduce the dimensions to 2D. Before applying PCA, cosine similarity was used to calculate the distance between customers, as it focuses on the relationship between features rather than their magnitude.

```python
from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)
```

### 5. Cluster Interpretation
Each cluster was analyzed to understand the behavior patterns of the customers within it. Below is a brief interpretation of each cluster:

- **Cluster 0**: Customers who don’t spend much but have an average to high credit limit.
- **Cluster 1**: High spenders with large credit limits, often making expensive purchases.
- **Cluster 2**: Customers with high credit limits who often take cash advances.
- **Cluster 3**: Customers with average to high credit limits who purchase mostly via installments.
- **Cluster 4**: Customers who frequently take cash advances and have pending payments.
- **Cluster 5**: Customers who make various types of purchases, from one-off to installments.

### 6. Visualization of Clusters
The resulting clusters were plotted in a 2D space using PCA, with different colors representing different clusters. This visual representation helped us understand the relationship between clusters and how customers group together based on their credit card behavior.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 13))
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name], label=names[name], mec='none')
ax.legend()
ax.set_title("Customer Segmentation based on Credit Card Usage")
plt.show()
```

## Results and Insights
After analyzing the clusters, we derived the following insights:
- **Cluster 1** consists of high-value customers who spend a lot and have a high credit limit. They are prime candidates for premium services or products.
- **Cluster 2** includes customers who frequently take cash advances. Financial institutions may want to offer special cash advance offers or target them with interest rate reductions.
- **Cluster 3** contains customers who rely on installment purchases, suggesting they may be more price-sensitive.
- **Cluster 5** is made up of customers who exhibit varied purchase behaviors, indicating a more dynamic spending pattern.

These insights can be leveraged by credit card companies to:
1. **Tailor marketing campaigns** to specific customer groups.
2. **Provide personalized product offerings**, such as installment-based payment plans or cash advance promotions.
3. **Improve customer retention** by offering loyalty programs to high spenders.

## Conclusion
This project demonstrates how clustering techniques can be applied to segment customers based on their credit card usage behavior. By understanding the patterns and traits of different customer segments, businesses can improve their marketing strategy and customer service offerings.
