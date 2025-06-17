import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Sprawdzenie dostępności CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")
if torch.cuda.is_available():
    print(f"Nazwa GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pamięć GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Ustawienie seedów dla reprodukowalności
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Wczytanie danych Apple
file_path = r'C:\Users\pawli\OneDrive\Pulpit\sieci\Projektowanie_i_zastosowania_sieci_neuronowych\apple_stock_data.csv'

print("📊 Wczytywanie danych Apple...")
df = pd.read_csv(file_path)
print(f"✅ Wczytano {len(df)} rekordów")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

if 'Close/Last' not in df.columns and 'Close' in df.columns:
    df['Close/Last'] = df['Close']

# Czyszczenie i sortowanie danych
df = df.dropna()
df = df.sort_index()

print(f"Liczba dni w zbiorze danych: {len(df)}")
print(f"Zakres dat: {df.index[0]} - {df.index[-1]}")

# Feature engineering
df['Returns'] = df['Close/Last'].pct_change()
df['MA_5'] = df['Close/Last'].rolling(window=5).mean()
df['MA_20'] = df['Close/Last'].rolling(window=20).mean()
df['MA_50'] = df['Close/Last'].rolling(window=50).mean()
df['Volatility'] = df['Returns'].rolling(window=20).std()
df['RSI'] = calculate_rsi(df['Close/Last'], 14)

# Usuwamy NaN powstałe z rolling
df = df.dropna()

# Używamy wielu cech zamiast tylko ceny zamknięcia
feature_columns = ['Close/Last', 'Returns', 'MA_5', 'MA_20', 'Volatility']
data = df[feature_columns].values

# Używamy StandardScaler dla lepszej normalizacji
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Zapisujemy scaler dla ceny (pierwsza kolumna)
price_scaler = StandardScaler()
price_scaler.fit(df['Close/Last'].values.reshape(-1, 1))

# Podział na dane treningowe i testowe (80/20)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Funkcja do tworzenia sekwencji czasowych
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Przewidujemy tylko cenę zamknięcia
    return np.array(X), np.array(y).reshape(-1, 1)

# Krótsze okno czasowe dla lepszej generalizacji
seq_length = 30  # 30 dni

# Tworzenie sekwencji
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

print(f"Kształt danych treningowych: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Kształt danych testowych: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Konwersja na tensory PyTorch
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# DataLoadery
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ULEPSZONA ARCHITEKTURA
class ImprovedStockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=1, dropout=0.3):
        super(ImprovedStockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM dla lepszego kontekstu
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Głębsza sieć fully connected z batch normalization
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        out = self.relu(self.bn1(self.fc1(context)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.relu(self.bn3(self.fc3(out)))
        out = self.fc4(out)
        
        return out

# Model z większą pojemnością
model = ImprovedStockLSTM(input_size=len(feature_columns)).to(device)

print(f"Architektura modelu:")
print(f"Liczba parametrów: {sum(p.numel() for p in model.parameters()):,}")

# Huber Loss zamiast MSE dla odporności na outliers
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.best_model = model.state_dict().copy()
        self.val_loss_min = val_loss

# Funkcje trenowania
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Trenowanie z early stopping
epochs = 100  # Zmniejszone dla szybszego treningu
train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=15, verbose=True)

print("🚀 Rozpoczynanie trenowania modelu LSTM dla Apple...")
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = evaluate(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    early_stopping(val_loss, model)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoka [{epoch+1}/{epochs}], Strata treningowa: {train_loss:.6f}, Strata walidacyjna: {val_loss:.6f}')
    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        model.load_state_dict(early_stopping.best_model)
        break

# Przewidywania z najlepszym modelem
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor).cpu().numpy()
    test_predictions = model(X_test_tensor).cpu().numpy()

# Denormalizacja tylko cen (pierwsza kolumna)
train_predictions_denorm = price_scaler.inverse_transform(train_predictions)
test_predictions_denorm = price_scaler.inverse_transform(test_predictions)
y_train_denorm = price_scaler.inverse_transform(y_train)
y_test_denorm = price_scaler.inverse_transform(y_test)

# Metryki
train_mse = mean_squared_error(y_train_denorm, train_predictions_denorm)
test_mse = mean_squared_error(y_test_denorm, test_predictions_denorm)
train_mae = mean_absolute_error(y_train_denorm, train_predictions_denorm)
test_mae = mean_absolute_error(y_test_denorm, test_predictions_denorm)
train_r2 = r2_score(y_train_denorm, train_predictions_denorm)
test_r2 = r2_score(y_test_denorm, test_predictions_denorm)

print(f"\n📊 Metryki modelu:")
print(f"Trening - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
print(f"Test - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# Analiza kierunkowości
def directional_accuracy(y_true, y_pred):
    y_true_diff = np.diff(y_true.flatten())
    y_pred_diff = np.diff(y_pred.flatten())
    return np.mean((y_true_diff > 0) == (y_pred_diff > 0))

train_dir_acc = directional_accuracy(y_train_denorm, train_predictions_denorm)
test_dir_acc = directional_accuracy(y_test_denorm, test_predictions_denorm)

print(f"\nDokładność kierunkowa:")
print(f"Trening: {train_dir_acc:.2%}")
print(f"Test: {test_dir_acc:.2%}")

# Zapisz model z wszystkimi potrzebnymi danymi
model_save_path = r'C:\Users\pawli\OneDrive\Pulpit\sieci\Projektowanie_i_zastosowania_sieci_neuronowych\improved_spy_lstm_model.pth'

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'price_scaler': price_scaler,
    'scaler': scaler,
    'seq_length': seq_length,
    'feature_columns': feature_columns,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'metrics': {
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'test_dir_acc': test_dir_acc
    },
    # Dodajemy rzeczywiste predykcje do późniejszego użycia
    'train_predictions': train_predictions_denorm,
    'test_predictions': test_predictions_denorm,
    'y_train_real': y_train_denorm,
    'y_test_real': y_test_denorm,
    'train_dates': df.index[seq_length:train_size],
    'test_dates': df.index[train_size+seq_length:]
}, model_save_path)

print(f"\n✅ Model zapisany jako: {model_save_path}")
print("🎉 Trening zakończony! Teraz możesz uruchomić notebook z prawdziwymi wynikami.")
print(f"\n📈 Podsumowanie:")
print(f"   • Wytrenowano na {len(X_train)} próbkach treningowych")
print(f"   • Przetestowano na {len(X_test)} próbkach testowych") 
print(f"   • MAE na testach: ${test_mae:.2f}")
print(f"   • R² na testach: {test_r2:.3f}")
print(f"   • Dokładność kierunkowa: {test_dir_acc:.1%}")