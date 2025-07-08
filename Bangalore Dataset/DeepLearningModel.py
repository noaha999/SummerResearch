import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from CatFunc import categorize_congestion, categorize_signals

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

le_target = LabelEncoder()
y_fit = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}")