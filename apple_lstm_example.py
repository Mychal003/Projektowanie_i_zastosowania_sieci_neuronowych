
# Przykład użycia danych Apple w modelu LSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Załaduj dane
X_train = np.load('apple_X_train.npy')
y_train = np.load('apple_y_train.npy')
X_val = np.load('apple_X_val.npy')
y_val = np.load('apple_y_val.npy')
X_test = np.load('apple_X_test.npy')
y_test = np.load('apple_y_test.npy')

print(f"Dane treningowe: {X_train.shape}, {y_train.shape}")
print(f"Dane walidacyjne: {X_val.shape}, {y_val.shape}")
print(f"Dane testowe: {X_test.shape}, {y_test.shape}")

# Stwórz model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Kompiluj model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Trenuj model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Ewaluacja
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Predykcje
predictions = model.predict(X_test)

# Zapisz model
model.save('apple_lstm_model.h5')
print("Model zapisany jako apple_lstm_model.h5")
