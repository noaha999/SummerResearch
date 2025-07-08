import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap


df = pd.read_csv('/Users/daddy/Downloads/smart_traffic_management_dataset.csv')

le_light = LabelEncoder()
df['TrafficLightEncoded'] = le_light.fit_transform(df['signal_status'])

le_weather = LabelEncoder()
df['WeatherEncoded'] = le_weather.fit_transform(df['weather_condition'])

X = df[['location_id', 'traffic_volume', 'avg_vehicle_speed', 'vehicle_count_cars', 'vehicle_count_trucks', 'vehicle_count_bikes',
        'WeatherEncoded', 'temperature', 'humidity', 'accident_reported']]
y = df['TrafficLightEncoded']

le_target = LabelEncoder()
y_fit = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

model = xgb.XGBClassifier(eta = .001,
                          n_estimators= 200,
                          max_depth= 5,
                          use_label_encoder = False,
                          eval_metric='mlogloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=X_test.columns)
