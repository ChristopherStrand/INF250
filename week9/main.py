from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
pca = PCA(n_components=3)
X_pca = pca.fit(df)




df["species_id"] = iris.target
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df)


sns.scatterplot(data=df, x="sepal width (cm)", y="sepal length (cm)", hue="species")
plt.show()
sns.histplot(data=df, x="petal length (cm)", bins=20, hue="species", multiple="stack")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[iris.feature_names])
for x in X_scaled:
    print(x)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled) 


print("Forklart varians:", pca.explained_variance_ratio_)
sns.scatterplot(x=X_pca[:, 0],y=X_pca[:, 1],hue=df["species"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca[:, :2])

x = X_pca[:, 0]
y = X_pca[:, 1]
plt.scatter(x, y, c=labels)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, c='black')
plt.show()