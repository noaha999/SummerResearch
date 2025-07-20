import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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


iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(X_scaled)

# Step 5: Label anomalies
# -1 means anomaly, 1 means normal
df["Anomaly Label"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

# Step 6: Visualize (2D plot example)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=range(len(df)), y="accident_reported", hue="Anomaly Label", palette={"Normal": "green", "Anomaly": "red"})
plt.title("Anomaly Detection")
plt.grid(True)
plt.show()

# Step 7: Show only anomalies
print("\nDetected Anomalies:")
print(df[df["Anomaly Label"] == "Anomaly"][["Date", "Area Name", "Road/Intersection Name"] + features])
