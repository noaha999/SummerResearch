import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from CatFunc import categorize_signals, categorize_congestion

df = pd.read_csv('/Users/daddy/Downloads/Banglore_traffic_Dataset.csv')
print(df.head())

df['CongestionCategory'] = df['Congestion Level'].apply(categorize_congestion)
#df['TrafficCategory'] = df['Traffic Signal Compliance'].apply(categorize_signals)
le_weather = LabelEncoder()
df['Weather_Encoded'] = le_weather.fit_transform(df['Weather Conditions'])

#le_roadwork = LabelEncoder()
#df['Roadwork_Encoded'] = le_roadwork.fit_transform(df['Roadwork and Construction Activity'])

X = df[['Traffic Volume', 'Average Speed', 'Travel Time Index', 'Road Capacity Utilization', 'Incident Reports', 'Public Transport Usage',
'Traffic Signal Compliance', 'Parking Usage', 'Pedestrian and Cyclist Count', 'Weather_Encoded']]
y = df['CongestionCategory']

le_target = LabelEncoder()
y_fit = le_target.fit_transform(y)
y_cat = to_categorical(y_fit)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}")