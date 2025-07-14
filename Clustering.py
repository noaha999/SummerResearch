import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv('/Users/daddy/Downloads/Banglore_traffic_Dataset.csv')

features = [
    "Traffic Volume", "Average Speed", "Travel Time Index", "Congestion Level",
    "Road Capacity Utilization", "Incident Reports", "Environmental Impact",
    "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
    "Pedestrian and Cyclist Count"
]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

cluster_centers_df = pd.DataFrame(centers_original, columns=features)
print(cluster_centers_df)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Traffic Signal Compliance",
    y="Traffic Volume",
    hue="Cluster",
    palette="viridis",
    s=100
)
plt.title("2D Clustering of Traffic Intersections")
plt.grid(True)
plt.tight_layout()
plt.show()

