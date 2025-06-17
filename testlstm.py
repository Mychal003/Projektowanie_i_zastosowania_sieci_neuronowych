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

# Wczytanie danych
file_path = r'C:\Users\pawli\OneDrive\Pulpit\sieci\Projektowanie_i_zastosowania_sieci_neuronowych\apple_stock_data.csv'

try:
    df = pd.read_csv(file_path)
    print(f"\nPomyślnie wczytano dane z pliku")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    if 'Close/Last' not in df.columns and 'Close' in df.columns:
        df['Close/Last'] = df['Close']
        
except:
    print("Używam przykładowych danych...")
    np.random.seed(42)
    dates = pd.date_range(start='2015-01-01', end='2025-06-06', freq='D')
    prices = 200 + np.cumsum(np.random.randn(len(dates)) * 2)
    df = pd.DataFrame({'Close/Last': prices}, index=dates)

# Czyszczenie i sortowanie danych
df = df.dropna()
df = df.sort_index()

print(f"\nLiczba dni w zbiorze danych: {len(df)}")
print(f"Zakres dat: {df.index[0]} - {df.index[-1]}")

# ULEPSZONE: Dodanie feature engineering
df['Returns'] = df['Close/Last'].pct_change()
df['MA_5'] = df['Close/Last'].rolling(window=5).mean()
df['MA_20'] = df['Close/Last'].rolling(window=20).mean()
df['MA_50'] = df['Close/Last'].rolling(window=50).mean()
df['Volatility'] = df['Returns'].rolling(window=20).std()
df['RSI'] = calculate_rsi(df['Close/Last'], 14)

# Usuwamy NaN powstałe z rolling
df = df.dropna()

# Funkcja do obliczania RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ULEPSZONE: Używamy wielu cech zamiast tylko ceny zamknięcia
feature_columns = ['Close/Last', 'Returns', 'MA_5', 'MA_20', 'Volatility']
data = df[feature_columns].values

# ULEPSZONE: Używamy StandardScaler dla lepszej normalizacji
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

# ULEPSZONE: Krótsze okno czasowe dla lepszej generalizacji
seq_length = 30  # 30 dni zamiast 60

# Tworzenie sekwencji
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

print(f"\nKształt danych treningowych: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Kształt danych testowych: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Konwersja na tensory PyTorch
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# DataLoadery
batch_size = 64  # Większy batch size
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

print(f"\nArchitektura modelu:")
print(f"Liczba parametrów: {sum(p.numel() for p in model.parameters()):,}")

# ULEPSZONE: Huber Loss zamiast MSE dla odporności na outliers
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
epochs = 200
train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=30, verbose=True)

print("\nRozpoczynanie trenowania...")
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

print(f"\nMetryki modelu:")
print(f"Trening - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
print(f"Test - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# Analiza kierunkowości (czy przewidujemy kierunek zmiany)
def directional_accuracy(y_true, y_pred):
    y_true_diff = np.diff(y_true.flatten())
    y_pred_diff = np.diff(y_pred.flatten())
    return np.mean((y_true_diff > 0) == (y_pred_diff > 0))

train_dir_acc = directional_accuracy(y_train_denorm, train_predictions_denorm)
test_dir_acc = directional_accuracy(y_test_denorm, test_predictions_denorm)

print(f"\nDokładność kierunkowa:")
print(f"Trening: {train_dir_acc:.2%}")
print(f"Test: {test_dir_acc:.2%}")

# Wykresy
plt.style.use('seaborn-v0_8-darkgrid')

# Wykres 1: Historia treningu
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses[:len(train_losses)], label='Strata treningowa', alpha=0.8)
plt.plot(val_losses[:len(val_losses)], label='Strata walidacyjna', alpha=0.8)
plt.title('Historia treningu modelu')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.grid(True, alpha=0.3)

# Wykres 2: Rozkład błędów z normalnym rozkładem
plt.subplot(1, 2, 2)
test_errors = y_test_denorm.flatten() - test_predictions_denorm.flatten()
plt.hist(test_errors, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)

# Dodaj krzywą normalną
from scipy import stats
mu, std = stats.norm.fit(test_errors)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'r-', linewidth=2, label=f'Normal(μ={mu:.1f}, σ={std:.1f})')

plt.title('Rozkład błędów przewidywań')
plt.xlabel('Błąd przewidywania ($)')
plt.ylabel('Gęstość')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Wykres 3: Scatter plot przewidywań
plt.figure(figsize=(10, 8))
plt.scatter(y_test_denorm, test_predictions_denorm, alpha=0.5, s=20)
plt.plot([y_test_denorm.min(), y_test_denorm.max()], 
         [y_test_denorm.min(), y_test_denorm.max()], 'r--', lw=2)
plt.xlabel('Rzeczywiste ceny ($)')
plt.ylabel('Przewidywane ceny ($)')
plt.title(f'Przewidywania vs Rzeczywistość (R² = {test_r2:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Wykres 4: Przewidywania na danych testowych
plt.figure(figsize=(15, 8))
test_dates = df.index[train_size+seq_length:]

plt.plot(test_dates, y_test_denorm, label='Rzeczywiste ceny', color='black', linewidth=2)
plt.plot(test_dates, test_predictions_denorm, label='Przewidywania', color='red', linewidth=2, alpha=0.8)

# Przedział ufności
plt.fill_between(test_dates, 
                 test_predictions_denorm.flatten() - test_mae, 
                 test_predictions_denorm.flatten() + test_mae, 
                 alpha=0.2, color='red', label=f'±MAE (${test_mae:.2f})')

plt.title('Przewidywania vs rzeczywiste ceny (dane testowe)', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Ulepszone przewidywanie przyszłości z Monte Carlo
def predict_future_monte_carlo(model, last_sequence, n_days, scaler, price_scaler, device, n_simulations=100):
    model.eval()
    all_predictions = []
    
    for sim in range(n_simulations):
        predictions = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for day in range(n_days):
                seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
                pred = model(seq_tensor).cpu().numpy()[0]
                
                # Dodaj szum do przewidywania
                noise = np.random.normal(0, 0.01)
                pred_with_noise = pred + noise
                predictions.append(pred_with_noise[0])
                
                # Aktualizacja features dla następnego kroku
                new_features = current_sequence[-1].copy()
                new_features[0] = pred_with_noise[0]  # Nowa cena
                
                # Przelicz pozostałe features (uproszczone)
                if len(current_sequence) > 1:
                    new_features[1] = (pred_with_noise[0] - current_sequence[-1, 0]) / current_sequence[-1, 0]  # Return
                
                current_sequence = np.vstack([current_sequence[1:], new_features])
        
        all_predictions.append(predictions)
    
    all_predictions = np.array(all_predictions)
    mean_predictions = np.mean(all_predictions, axis=0).reshape(-1, 1)
    std_predictions = np.std(all_predictions, axis=0).reshape(-1, 1)
    
    mean_denorm = price_scaler.inverse_transform(mean_predictions)
    std_denorm = std_predictions * price_scaler.scale_[0]
    
    return mean_denorm, std_denorm

# Przewidywanie z niepewnością
last_sequence = scaled_data[-seq_length:]
future_mean, future_std = predict_future_monte_carlo(model, last_sequence, 30, scaler, price_scaler, device)

# Wykres 5: Przewidywania z niepewnością
plt.figure(figsize=(15, 8))
last_60_days = df.index[-60:]
last_60_prices = df['Close/Last'].iloc[-60:]

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

plt.plot(last_60_days, last_60_prices, label='Dane historyczne', color='black', linewidth=2)
plt.plot(future_dates, future_mean, label='Średnie przewidywanie', 
         color='green', linewidth=2, linestyle='-')

# Przedziały ufności
plt.fill_between(future_dates, 
                 (future_mean - 2*future_std).flatten(), 
                 (future_mean + 2*future_std).flatten(), 
                 alpha=0.3, color='green', label='95% przedział ufności')

plt.axvline(x=df.index[-1], color='red', linestyle=':', alpha=0.5, label='Koniec danych')
plt.title('Przewidywania cen SPY z niepewnością (Monte Carlo)', fontsize=16)
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Podsumowanie
print(f"\nPodsumowanie ulepszonego modelu:")
print(f"- Architektura: Bidirectional LSTM z attention mechanism")
print(f"- Features: {len(feature_columns)} (cena, zwroty, średnie kroczące, zmienność)")
print(f"- Okno czasowe: {seq_length} dni")
print(f"- Early stopping na epoce: {len(train_losses)}")
print(f"- Średni błąd na danych testowych: ${test_mae:.2f} ({test_mae/df['Close/Last'].iloc[-1]*100:.1f}%)")
print(f"- Dokładność kierunkowa: {test_dir_acc:.1%}")
print(f"- Ostatnia cena: ${df['Close/Last'].iloc[-1]:.2f}")
print(f"- Przewidywanie za 30 dni: ${future_mean[-1][0]:.2f} (±${2*future_std[-1][0]:.2f})")

# Zapisz model
save_model = input("\nCzy zapisać ulepszony model? (t/n): ")
if save_model.lower() == 't':
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
        }
    }, 'improved_spy_lstm_model.pth')
    print("Model zapisany jako 'improved_spy_lstm_model.pth'")