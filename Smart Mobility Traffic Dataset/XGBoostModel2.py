import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('/Users/daddy/Downloads/smart_mobility_dataset.csv')

le_light = LabelEncoder()
df['TrafficLightEncoded'] = le_light.fit_transform(df['Traffic_Light_State'])

le_weather = LabelEncoder()
df['WeatherEncoded'] = le_weather.fit_transform(df['Weather_Condition'])

le_traffic = LabelEncoder()
df['TrafficEncoded'] = le_traffic.fit_transform(df['Traffic_Condition'])

X = df[['Latitude', 'Longitude', 'Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%', 'WeatherEncoded',
        'Accident_Report', 'Sentiment_Score', 'Ride_Sharing_Demand', 'Parking_Availability',
        'Emission_Levels_g_km', 'Energy_Consumption_L_h', 'TrafficEncoded']]
y = df['TrafficLightEncoded']

le_target = LabelEncoder()
y_fit = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = xgb.XGBClassifier(eta = .001,
                          n_estimators= 200,
                          max_depth= 5,
                          use_label_encoder = False,
                          eval_metric='mlogloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

xgb.plot_importance(model, importance_type='gain')
plt.show()