'''Ulepszona implementacja modelu LSTM z lepszą wydajnością'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional

class TimeSeriesDataset(Dataset):
    """Dataset dla danych szeregów czasowych."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class ImprovedLSTMModel(nn.Module):
    """Ulepszona architektura LSTM z dodatkową normalizacją."""
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int = 128,  # Zwiększony rozmiar
                 num_layers: int = 3,     # Więcej warstw
                 output_size: int = 1,
                 dropout: float = 0.3):   # Zwiększony dropout
        super(ImprovedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM dla lepszego kontekstu
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Kluczowa zmiana
        )
        
        # Warstwa normalizacji
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Głębsza sieć fully connected
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Bierzemy ostatni output (z obu kierunków)
        last_output = lstm_out[:, -1, :]
        
        # Normalizacja batch (jeśli batch_size > 1)
        if batch_size > 1:
            last_output = self.batch_norm(last_output)
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class ImprovedLSTMTrainer:
    """Ulepszona klasa trenera z lepszymi strategiami optymalizacji."""
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.0005,  # Niższa początkowa wartość
                 device: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Huber Loss zamiast MSE - mniej wrażliwy na wartości odstające
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # AdamW z lepszą regularyzacją
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            
            # Gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(self, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            epochs: int = 150,  # Więcej epok
            early_stopping_patience: int = 20,
            verbose: bool = True) -> Dict[str, List[float]]:
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            if verbose and epoch % 10 == 0:
                print(f'Epoka {epoch+1}/{epochs}, '
                      f'Strata treningowa: {train_loss:.6f}, '
                      f'Strata walidacyjna: {val_loss:.6f}')
            
            # Early stopping z zapisem najlepszego modelu
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict().copy()
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f'Early stopping po {epoch+1} epokach')
                break
        
        # Przywróć najlepszy model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy zaawansowane cechy techniczne."""
    df = df.copy()
    
    # Podstawowe cechy techniczne
    df['Returns'] = df['Close/Last'].pct_change()
    df['Log_Returns'] = np.log(df['Close/Last'] / df['Close/Last'].shift(1))
    
    # Średnie kroczące
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['Close/Last'].rolling(window=window).mean()
        df[f'MA_ratio_{window}'] = df['Close/Last'] / df[f'MA_{window}']
    
    # Volatility (odchylenie standardowe zwrotów)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close/Last'])
    
    # Bollinger Bands
    df['BB_middle'] = df['Close/Last'].rolling(window=20).mean()
    bb_std = df['Close/Last'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    df['BB_position'] = (df['Close/Last'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price position within day's range
    df['Price_position'] = (df['Close/Last'] - df['Low']) / (df['High'] - df['Low'])
    
    # Usuń wiersze z NaN
    df = df.dropna()
    
    return df

def prepare_lstm_data_improved(df: pd.DataFrame,
                              seq_length: int = 30,  # Dłuższa sekwencja
                              target_days: int = 1,   # Predykcja na 1 dzień
                              test_size: float = 0.2,
                              val_size: float = 0.1,
                              batch_size: int = 64) -> Dict[str, Any]:
    """Ulepszona funkcja przygotowania danych."""
    
    # Dodaj zaawansowane cechy
    df = create_advanced_features(df)
    
    # Wybierz cechy do modelu
    feature_cols = [
        'Open', 'High', 'Low', 'Close/Last', 'Volume',
        'Returns', 'MA_5', 'MA_10', 'MA_20',
        'MA_ratio_5', 'MA_ratio_10', 'MA_ratio_20',
        'Volatility', 'RSI', 'BB_position',
        'Volume_ratio', 'Price_position'
    ]
    
    # Przygotuj dane
    features = df[feature_cols].values
    target = df['Close/Last'].values
    
    # Normalizacja z RobustScaler (lepszy dla danych z outlierami)
    from sklearn.preprocessing import RobustScaler
    
    feature_scaler = RobustScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1))
    
    # Tworzenie sekwencji
    X, y = [], []
    for i in range(seq_length, len(features_scaled) - target_days + 1):
        X.append(features_scaled[i-seq_length:i])
        y.append(target_scaled[i + target_days - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Podział chronologiczny (ważne dla danych czasowych!)
    train_size = int(len(X) * (1 - test_size - val_size))
    val_size = int(len(X) * val_size)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'input_size': len(feature_cols),
        'feature_names': feature_cols
    }