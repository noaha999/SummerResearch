import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

df = pd.read_csv('/Users/daddy/Downloads/smart_traffic_management_dataset.csv')

le_light = LabelEncoder()
df['TrafficLightEncoded'] = le_light.fit_transform(df['signal_status'])

le_weather = LabelEncoder()
df['WeatherEncoded'] = le_weather.fit_transform(df['weather_condition'])

features = [
    'location_id', 'traffic_volume', 'avg_vehicle_speed', 'vehicle_count_cars', 'vehicle_count_trucks', 'vehicle_count_bikes',
    'WeatherEncoded', 'temperature', 'humidity', 'accident_reported', 'TrafficLightEncoded'
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
    x="TrafficLightEncoded",
    y="avg_vehicle_speed",
    hue="Cluster",
    palette="viridis",
    s=100
)
plt.title("2D Clustering of Traffic Intersections")
plt.grid(True)
plt.tight_layout()
plt.show()

