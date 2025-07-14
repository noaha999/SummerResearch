import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load your dataset
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


iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(X_scaled)

# Step 5: Label anomalies
# -1 means anomaly, 1 means normal
df["Anomaly Label"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

# Step 6: Visualize (2D plot example)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Traffic Volume", y="Average Speed", hue="Anomaly Label", palette={"Normal": "green", "Anomaly": "red"})
plt.title("Anomaly Detection: Traffic Volume vs Speed")
plt.grid(True)
plt.show()

# Step 7: Show only anomalies
print("\nDetected Anomalies:")
print(df[df["Anomaly Label"] == "Anomaly"][["Date", "Area Name", "Road/Intersection Name"] + features])
