# Importing the Libraries
import numpy as np
import pandas as pd
import torch
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans as SKLearnKMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
from sklearn.metrics import silhouette_score, v_measure_score
from models import KMeans  # Importing the custom KMeans class
from models.ContrastiveCluster import ContrastiveCluster  # Importing the custom ContrastiveCluster class

if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

data = pd.read_csv("..\data\marketing_campaign.csv", sep="\t")

"""Data Cleaning"""
data = data.dropna()

data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)  
# Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in the records:", max(dates))
print("The oldest customer's enrolment date in the records:", min(dates))
days = []
d1 = max(dates)  # taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["Education"].value_counts())

# Feature Engineering
# Age of customer today 
data["Age"] = 2021 - data["Year_Birth"]

# Total spendings on various items
data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]

# Deriving living situation by marital status "Alone"
data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone",})

# Feature indicating total children living in the household
data["Children"] = data["Kidhome"] + data["Teenhome"]

# Feature for total members in the household
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]

# Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

# Segmenting education levels in three groups
data["Education"] = data["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})

# For clarity
data = data.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

# Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)

# To plot some selected features 
# Setting up colors preferences
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
# Plotting following features
To_Plot = ["Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
print("Relative Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(data[To_Plot], hue="Is_Parent", palette=(["#682F2F", "#F3AB60"]))
# Taking hue 
plt.show()

# Dropping the outliers by setting a cap on Age and income. 
data = data[(data["Age"] < 90)]
data = data[(data["Income"] < 600000)]
print("The total number of data-points after removing the outliers are:", len(data))

# Correlation matrix
numeric_data = data.select_dtypes(include=[np.number])
corrmat = numeric_data.corr()
plt.figure(figsize=(20, 20))  
sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)

"""Data Preprocessing"""
# Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

# Label Encoding the object dtypes.
LE = LabelEncoder()
for i in object_cols:
    data[i] = data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")

# Creating a copy of data
ds = data.copy()
# Creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
# Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
print("All features are now scaled")

"""DIMENSIONALITY REDUCTION"""
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1", "col2", "col3"]))
PCA_ds.describe().T

# A 3D Projection Of Data In The Reduced Dimension
x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]
# To plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="maroon", marker="o")
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()


"""CLUSTERING"""
# Quick examination of elbow method to find numbers of clusters to make.
"""可以作为 KMeans 聚类的提前演示。通过肘部法的可视化，你可以在实际进行 KMeans 聚类之前，确定一个合理的簇数，从而提高聚类效果。"""
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(SKLearnKMeans(), k=10)  # KElbowVisualizer is a class that visualizes the elbow method
Elbow_M.fit(PCA_ds)
Elbow_M.show()
"""如果不需要动态确定簇数，或者已经确定了簇数，那么这段代码可以省略。但在大多数情况下，使用肘部法来确定簇数是一个推荐的步骤。"""


"""KMeans Clustering,簇的数量由上述代码验证为4,并且使用自定义的KMeans类"""
"""""
# Using custom KMeans class
kmeans = KMeans.KMeans(n_clusters=4,X=PCA_ds)
kmeans.fit(PCA_ds)
PCA_ds["Clusters"] = kmeans.predict(PCA_ds)
# Adding the Clusters feature to the original dataframe.
data["Clusters"] = PCA_ds["Clusters"]

#对KMeans聚类结果使用silhouette_score进行评估
silhouette = silhouette_score(PCA_ds, PCA_ds["Clusters"])
print("The silhouette score of the KMeans clustering model is:", silhouette)

# Plotting the clusters
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
ax.set_title("The Plot Of The Clusters")
plt.show()
"""""


"""Contrastive Clustering,簇的数量由上述代码验证为4,并且使用自定义的ContrastiveCluster类"""
input_dim = PCA_ds.shape[1]  # Input dimension is 3 (since we did PCA with 3 components)

# Initialize the ContrastiveClustering model
contrastive_cluster = ContrastiveCluster(input_dim=input_dim, n_clusters=4, n_features=3, alpha=1.0)

# Set optimizer (SGD or Adam)
optimizer = torch.optim.SGD([contrastive_cluster.centroids], lr=0.1)  # or use Adam optimizer
# optimizer = torch.optim.Adam([contrastive_cluster.centroids], lr=0.01)  # Optionally use Adam

x_data = torch.tensor(PCA_ds.values, dtype=torch.float32)
# Training loop
prev_loss = float('inf')  # Initialize previous loss for comparison
for epoch in range(100):
    optimizer.zero_grad()  # Clear gradients before backpropagation
    
    # Calculate the loss
    q = contrastive_cluster.loss(x_data)
    loss = q.sum()  # Sum of the loss

    loss.backward()  # Backpropagate the gradients
    optimizer.step()  # Update the centroids
    
    # Calculate the percentage change in loss
    loss_change_percentage = ((prev_loss - loss.item()) / prev_loss) * 100 if prev_loss != float('inf') else 0
    prev_loss = loss.item()

    # Print loss and percentage change
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Change = {loss_change_percentage:.2f}%")


# Predict cluster assignments
# Get cluster assignments
cluster_assignments = contrastive_cluster.predict(x_data).numpy()

# Calculate silhouette score
silhouette = silhouette_score(PCA_ds, cluster_assignments)
print(f"Silhouette Score after training: {silhouette:.4f}")

# Plotting (Optional)
plt.figure(figsize=(8, 6))
plt.scatter(PCA_ds["col1"], PCA_ds["col2"], c=cluster_assignments, cmap='viridis')
plt.title("Clustered Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


"""Evaluation of the clustering model"""
""""
# Plotting countplot of clusters
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
pl = sns.countplot(x=data["Clusters"], palette=pal)
pl.set_title("Distribution Of The Clusters")
plt.show()
pl = sns.scatterplot(data=data, x=data["Spent"], y=data["Income"], hue=data["Clusters"], palette=pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()
plt.figure()
pl = sns.swarmplot(x=data["Clusters"], y=data["Spent"], color="#CBEDDD", alpha=0.5)
pl = sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal)
plt.show()
# Creating a feature to get a sum of accepted promotions 
data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]
# Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()
# Plotting the number of deals purchased
plt.figure()
pl = sns.boxenplot(y=data["NumDealsPurchases"], x=data["Clusters"], palette=pal)
pl.set_title("Number of Deals Purchased")
plt.show()
"""""


"""Profile of the clusters"""
""""
Personal = ["Kidhome", "Teenhome", "Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education", "Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data[i], y=data["Spent"], hue=data["Clusters"], kind="kde", palette=pal)
    plt.show()
"""""