'''Definicja architektury sieci konwolucyjnej'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

class TimeSeriesCNN(nn.Module):
    """
    Model sieci konwolucyjnej do prognozowania cen akcji.
    """
    def __init__(self, 
                 input_channels: int, 
                 seq_length: int,
                 output_size: int = 1,
                 kernel_sizes: List[int] = [3, 5, 7],
                 n_filters: List[int] = [64, 128, 256],
                 dropout: float = 0.3):
        """
        Inicjalizacja modelu CNN.
        
        Args:
            input_channels: Liczba kanałów wejściowych (liczba cech)
            seq_length: Długość sekwencji czasowej
            output_size: Rozmiar wyjścia (zwykle 1 dla przewidywania jednej wartości)
            kernel_sizes: Lista rozmiarów jąder konwolucyjnych
            n_filters: Lista liczby filtrów dla każdej warstwy konwolucyjnej
            dropout: Współczynnik dropout
        """
        super(TimeSeriesCNN, self).__init__()
        
        assert len(kernel_sizes) == len(n_filters), "kernel_sizes i n_filters muszą mieć taką samą długość"
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        
        # Warstwy konwolucyjne
        conv_layers = []
        in_channels = input_channels
        
        for i, (kernel_size, n_filter) in enumerate(zip(kernel_sizes, n_filters)):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, 
                              out_channels=n_filter, 
                              kernel_size=kernel_size,
                              padding=(kernel_size // 2)),  # Zachowanie długości sekwencji
                    nn.BatchNorm1d(n_filter),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = n_filter
            seq_length = seq_length // 2  # Zmniejszenie długości po MaxPool
        
        self.conv_blocks = nn.ModuleList(conv_layers)
        
        # Warstwa spłaszczająca
        self.flatten = nn.Flatten()
        
        # Obliczenie rozmiaru po przejściu przez warstwy konwolucyjne
        flattened_size = n_filters[-1] * seq_length
        
        # Warstwy w pełni połączone
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass przez model CNN.
        
        Args:
            x: Tensor wejściowy o kształcie [batch_size, seq_length, input_channels]
            
        Returns:
            Tensor wyjściowy o kształcie [batch_size, output_size]
        """
        # Zamiana wymiarów dla konwolucji 1D (oczekuje [batch, channels, seq_length])
        x = x.permute(0, 2, 1)
        
        # Przejście przez bloki konwolucyjne
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        # Spłaszczenie
        x = self.flatten(x)
        
        # Przejście przez warstwy w pełni połączone
        x = self.fc_layers(x)
        
        return x

class CNNTrainer:
    """
    Klasa do trenowania i ewaluacji modelu CNN.
    """
    def __init__(self, 
                 model: TimeSeriesCNN, 
                 learning_rate: float = 0.001, 
                 weight_decay: float = 1e-5,
                 device: Optional[str] = None):
        """
        Inicjalizacja trenera modelu.
        
        Args:
            model: Model CNN do trenowania
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
                torch.save(self.model.state_dict(), 'models/cnn/best_model.pth')
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f'Early stopping po {epoch+1} epokach')
                break
                
        # Ładujemy najlepszy model
        self.model.load_state_dict(torch.load('models/cnn/best_model.pth'))
        
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

def prepare_data_for_cnn(df, 
                         feature_cols, 
                         target_col, 
                         seq_length, 
                         test_size=0.2, 
                         val_size=0.1,
                         batch_size=32, 
                         shuffle=True) -> Dict[str, Any]:
    """
    Przygotowuje dane do trenowania modelu CNN.
    
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
    from src.lstm_model import TimeSeriesDataset, create_sequence_data
    
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
        'input_channels': X_train.shape[2],  # Liczba cech wejściowych
        'seq_length': X_train.shape[1]  # Długość sekwencji
    }