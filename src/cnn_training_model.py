
    # Trening modelu CNN
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from cnn_model_improvedv2 import TimeSeriesCNNImprovedV2
from cnn_trainer_improved import CNNTrainerImproved

# === 1. Wczytanie danych ===
file_path = "../notebooks/data/processed/apple_stock_data.csv"  # Zmień na 'raw' jeśli potrzebujesz
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.dropna().sort_index()

# Feature engineering
df['Returns'] = df['Close/Last'].pct_change()
df['MA_5'] = df['Close/Last'].rolling(window=5).mean()
df['MA_20'] = df['Close/Last'].rolling(window=20).mean()
df['Volatility'] = df['Returns'].rolling(window=20).std()
df = df.dropna()

feature_columns = ['Close/Last', 'Returns', 'MA_5', 'MA_20', 'Volatility']
data = df[feature_columns].values

# Skalowanie danych
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
price_scaler = StandardScaler()
price_scaler.fit(df['Close/Last'].values.reshape(-1, 1))

# Tworzenie sekwencji
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y).reshape(-1, 1)

seq_length = 30
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Tensory i DataLoadery
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=64)

# === 2. Model CNN ===
model = TimeSeriesCNNImprovedV2(input_channels=X_train.shape[2], seq_length=seq_length)
trainer = CNNTrainerImproved(model)
history = trainer.fit(train_loader, val_loader, epochs=100)

# === 3. Predykcje i denormalizacja ===
y_train_pred = trainer.predict(X_train)
y_test_pred = trainer.predict(X_test)
y_train_pred_denorm = price_scaler.inverse_transform(y_train_pred)
y_test_pred_denorm = price_scaler.inverse_transform(y_test_pred)
y_train_real_denorm = price_scaler.inverse_transform(y_train)
y_test_real_denorm = price_scaler.inverse_transform(y_test)

# Metryki
train_mae = mean_absolute_error(y_train_real_denorm, y_train_pred_denorm)
test_mae = mean_absolute_error(y_test_real_denorm, y_test_pred_denorm)
train_r2 = r2_score(y_train_real_denorm, y_train_pred_denorm)
test_r2 = r2_score(y_test_real_denorm, y_test_pred_denorm)
def directional_accuracy(y_true, y_pred):
    return np.mean((np.diff(y_true.flatten()) > 0) == (np.diff(y_pred.flatten()) > 0))
train_dir_acc = directional_accuracy(y_train_real_denorm, y_train_pred_denorm)
test_dir_acc = directional_accuracy(y_test_real_denorm, y_test_pred_denorm)

# === 4. Zapis modelu ===
os.makedirs("../notebooks/models/cnn", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': trainer.optimizer.state_dict(),
    'price_scaler': price_scaler,
    'scaler': scaler,
    'seq_length': seq_length,
    'feature_columns': feature_columns,
    'train_losses': history['train_loss'],
    'val_losses': history['val_loss'],
    'metrics': {
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_dir_acc': train_dir_acc,
        'test_mae': test_mae,
        'test_mse': mean_squared_error(y_test_real_denorm, y_test_pred_denorm),
        'test_r2': test_r2,
        'test_dir_acc': test_dir_acc
    },
    'train_predictions': y_train_pred_denorm,
    'test_predictions': y_test_pred_denorm,
    'y_train_real': y_train_real_denorm,
    'y_test_real': y_test_real_denorm,
    'train_dates': df.index[seq_length:train_size],
    'test_dates': df.index[train_size + seq_length:]
}, '../notebooks/models/cnn/best_model.pth')

print("✅ Model CNN został wytrenowany i zapisany do analizy.")