'''Definicja architektury sieci LSTM'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

class TimeSeriesDataset(Dataset):
    """
    Dataset do przygotowania danych dla modelu LSTM.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """
    Model sieci neuronowej LSTM do przewidywania cen akcji.
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int, 
                 dropout: float = 0.2):
        """
        Inicjalizacja modelu LSTM.
        
        Args:
            input_size: Liczba cech wejściowych (liczba zmiennych)
            hidden_size: Rozmiar warstw ukrytych LSTM
            num_layers: Liczba warstw LSTM
            output_size: Rozmiar wyjścia (np. 1 dla przewidywania jednej wartości)
            dropout: Współczynnik dropout (domyślnie 0.2)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Warstwa LSTM
        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        
        # Warstwa liniowa do predykcji
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass modelu LSTM.
        
        Args:
            x: Tensor wejściowy o kształcie [batch_size, seq_length, input_size]
            
        Returns:
            Tensor wyjściowy o kształcie [batch_size, output_size]
        """
        # Inicjalizacja stanu ukrytego
        batch_size = x.size(0)
        device = x.device
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # Forward pass przez LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Bierzemy tylko ostatni krok sekwencji
        out = out[:, -1, :]
        
        # Predykcja przez warstwę liniową
        out = self.fc(out)
        
        return out

class LSTMTrainer:
    """
    Klasa do trenowania i ewaluacji modelu LSTM.
    """
    def __init__(self, 
                 model: LSTMModel, 
                 learning_rate: float = 0.001, 
                 weight_decay: float = 1e-5,
                 device: Optional[str] = None):
        """
        Inicjalizacja trenera modelu.
        
        Args:
            model: Model LSTM do trenowania
            learning_rate: Szybkość uczenia (domyślnie 0.001)
            weight_decay: Współczynnik regularyzacji L2 (domyślnie 1e-5)
            device: Urządzenie do obliczeń ('cuda' lub 'cpu'), jeśli None to automatycznie wykrywa
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Trenowanie modelu przez jedną epokę.
        
        Args:
            train_loader: DataLoader z danymi treningowymi
            
        Returns:
            Średnia strata na epoce
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Walidacja modelu.
        
        Args:
            val_loader: DataLoader z danymi walidacyjnymi
            
        Returns:
            Średnia strata na walidacji
        """
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
            epochs: int, 
            early_stopping_patience: int = 10,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Trenowanie modelu przez określoną liczbę epok z opcją early stopping.
        
        Args:
            train_loader: DataLoader z danymi treningowymi
            val_loader: DataLoader z danymi walidacyjnymi
            epochs: Liczba epok
            early_stopping_patience: Liczba epok bez poprawy przed zatrzymaniem
            verbose: Czy wyświetlać postęp treningu
            
        Returns:
            Słownik z historią treningu
        """
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose:
                print(f'Epoka {epoch+1}/{epochs}, Strata treningowa: {train_loss:.6f}, Strata walidacyjna: {val_loss:.6f}')
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Zapisujemy najlepszy model
                torch.save(self.model.state_dict(), 'models/lstm/best_model.pth')
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f'Early stopping po {epoch+1} epokach')
                break
                
        # Ładujemy najlepszy model
        self.model.load_state_dict(torch.load('models/lstm/best_model.pth'))
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Wykonuje predykcję modelu na nowych danych.
        
        Args:
            X: Dane wejściowe jako numpy array
            
        Returns:
            Predykcje jako numpy array
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Ewaluacja modelu na danych testowych.
        
        Args:
            X_test: Dane testowe
            y_test: Etykiety testowe
            
        Returns:
            Słownik z metrykami
        """
        y_pred = self.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }

def create_sequence_data(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tworzy sekwencje danych dla modelu LSTM.
    
    Args:
        data: Dane wejściowe
        seq_length: Długość sekwencji
        
    Returns:
        Tuple (X, y) gdzie X to sekwencje danych, a y to odpowiadające etykiety
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)

def prepare_data_for_lstm(df, 
                          feature_cols, 
                          target_col, 
                          seq_length, 
                          test_size=0.2, 
                          val_size=0.1,
                          batch_size=32, 
                          shuffle=True) -> Dict[str, Any]:
    """
    Przygotowuje dane do trenowania modelu LSTM.
    
    Args:
        df: DataFrame z danymi
        feature_cols: Lista kolumn cech
        target_col: Nazwa kolumny celu
        seq_length: Długość sekwencji
        test_size: Proporcja danych testowych
        val_size: Proporcja danych walidacyjnych
        batch_size: Rozmiar batcha
        shuffle: Czy mieszać dane
        
    Returns:
        Słownik z przygotowanymi DataLoaderami i metadanymi
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    
    # Przygotowanie danych
    features = df[feature_cols].values
    target = df[target_col].values.reshape(-1, 1)
    
    # Normalizacja
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)
    
    # Tworzenie sekwencji
    X, y = create_sequence_data(features_scaled, seq_length)
    y_target = target_scaled[seq_length:]
    
    # Podział na zbiory treningowy, walidacyjny i testowy
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_target, test_size=test_size, shuffle=False)
    
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Korekta proporcji
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, shuffle=False)
    else:
        X_train, X_val, y_train, y_val = X_temp, np.empty((0, X_temp.shape[1], X_temp.shape[2])), y_temp, np.empty((0, 1))
    
    # Tworzenie DataLoaderów
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'input_size': X_train.shape[2]  # Liczba cech wejściowych
    }