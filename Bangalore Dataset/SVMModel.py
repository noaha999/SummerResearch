import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from CatFunc import categorize_congestion, categorize_signals
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv('/Users/daddy/Downloads/Banglore_traffic_Dataset.csv')
print(df.head())

#df['CongestionCategory'] = df['Congestion Level'].apply(categorize_congestion)
df['TrafficCategory'] = df['Traffic Signal Compliance'].apply(categorize_signals)
le_weather = LabelEncoder()
df['Weather_Encoded'] = le_weather.fit_transform(df['Weather Conditions'])

#le_roadwork = LabelEncoder()
#df['Roadwork_Encoded'] = le_roadwork.fit_transform(df['Roadwork and Construction Activity'])

X = df[['Traffic Volume', 'Average Speed', 'Congestion Level', 'Weather_Encoded', 'Incident Reports', 'Pedestrian and Cyclist Count']]
y = df['TrafficCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

