import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

data = pd.read_csv('winequality-white.csv', sep=';')

X = data.drop('quality', axis=1)
y = data['quality']

def remove_outliers_iqr(X_df, y_series):
    df = X_df.copy()
    y = y_series.copy()
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        df = df[mask]
        y = y[mask]
    return df, y

X_clean, y_clean = remove_outliers_iqr(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

test_loss, test_mse = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test MSE: {test_mse}')

test = model.predict(X_test_scaled[8].reshape(1, -1))
print("Predykcja:", np.round(test))
print(y[8])