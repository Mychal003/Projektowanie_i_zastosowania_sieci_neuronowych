"""
Model GRU do prognozowania kierunku zmian cen akcji

Autor: Student
Data: 2025
Projekt: Sieci neuronowe w finansach
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class GRUModel(nn.Module):
    """
    Model GRU do klasyfikacji kierunku zmian cen akcji
    
    Parametry:
    - input_size: liczba cech wejściowych
    - hidden_size: rozmiar warstwy ukrytej GRU
    - num_layers: liczba warstw GRU
    - dropout: współczynnik dropout
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Warstwa GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Warstwa wyjściowa (2 klasy: spadek=0, wzrost=1)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Przejście przez GRU
        gru_out, _ = self.gru(x)
        
        # Bierzemy ostatni timestep
        last_output = gru_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # Warstwa wyjściowa
        output = self.fc(last_output)
        
        return output

class GRUTrainer:
    """
    Klasa do trenowania modelu GRU
    """
    
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader):
        """Trenuj jeden epok"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.long().to(self.device)
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statystyki
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Walidacja modelu"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader, epochs=100, patience=10, verbose=True):
        """
        Trenuj model z early stopping
        
        Parametry:
        - train_loader: DataLoader z danymi treningowymi
        - val_loader: DataLoader z danymi walidacyjnymi
        - epochs: maksymalna liczba epok
        - patience: cierpliwość dla early stopping
        - verbose: czy wyświetlać postęp
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Trenowanie
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Walidacja
            val_loss, val_acc = self.validate(val_loader)
            
            # Zapisz historię
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Wyświetl postęp
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoka {epoch+1:3d}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            
            # Zatrzymaj jeśli brak poprawy
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping po {epoch+1} epokach")
                break
        
        return history
    
    def predict(self, X_test):
        """Predykcje na danych testowych"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        # Konwertuj na tensor jeśli potrzeba
        if isinstance(X_test, np.ndarray):
            X_test = torch.FloatTensor(X_test)
        
        with torch.no_grad():
            # Przetwarzaj w batch'ach
            batch_size = 64
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size].to(self.device)
                outputs = self.model(batch)
                
                # Prawdopodobieństwa
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
                
                # Predykcje
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)

def prepare_stock_data(df, feature_cols, target_col='Target_Return', seq_length=30, 
                      test_size=0.2, val_size=0.1):
    """
    Przygotuj dane giełdowe do trenowania modelu GRU
    
    Parametry:
    - df: DataFrame z danymi
    - feature_cols: lista nazw kolumn z cechami
    - target_col: nazwa kolumny z targetem
    - seq_length: długość sekwencji czasowej
    - test_size: procent danych testowych
    - val_size: procent danych walidacyjnych
    
    Zwraca:
    - dict z przygotowanymi danymi
    """
    
    # Sortuj chronologicznie
    df = df.sort_values('Date' if 'Date' in df.columns else df.index).reset_index(drop=True)
    
    # Usuń NaN
    df_clean = df[feature_cols + [target_col]].dropna()
    
    # Przygotuj target jako klasyfikację (0=spadek, 1=wzrost)
    target = (df_clean[target_col] > 0).astype(int).values
    features = df_clean[feature_cols].values
    
    # Normalizacja cech
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Stwórz sekwencje czasowe
    X, y = [], []
    for i in range(seq_length, len(features_scaled)):
        X.append(features_scaled[i-seq_length:i])
        y.append(target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Utworzono {len(X)} sekwencji, każda o długości {seq_length}")
    print(f"Rozkład klas: {np.bincount(y)}")
    
    # Podział danych (chronologicznie)
    total_samples = len(X)
    test_samples = int(total_samples * test_size)
    val_samples = int(total_samples * val_size)
    train_samples = total_samples - test_samples - val_samples
    
    X_train = X[:train_samples]
    y_train = y[:train_samples]
    
    X_val = X[train_samples:train_samples + val_samples]
    y_val = y[train_samples:train_samples + val_samples]
    
    X_test = X[train_samples + val_samples:]
    y_test = y[train_samples + val_samples:]
    
    # Sprawdź balance klas
    print(f"\nPodział danych:")
    print(f"Train: {len(X_train)} próbek, klasa 1: {np.mean(y_train):.1%}")
    print(f"Val:   {len(X_val)} próbek, klasa 1: {np.mean(y_val):.1%}")
    print(f"Test:  {len(X_test)} próbek, klasa 1: {np.mean(y_test):.1%}")
    
    # Stwórz DataLoader'y
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'input_size': len(feature_cols),
        'feature_cols': feature_cols
    }

def add_technical_indicators(df):
    """
    Dodaj wskaźniki techniczne do DataFrame
    
    Parametry:
    - df: DataFrame z kolumnami OHLCV
    
    Zwraca:
    - DataFrame z dodanymi wskaźnikami
    """
    
    df = df.copy()
    
    # Podstawowe zwroty
    df['Returns'] = df['Close/Last'].pct_change()
    
    # Wskaźniki cenowe
    df['MA_5'] = df['Close/Last'].rolling(window=5).mean()
    df['MA_20'] = df['Close/Last'].rolling(window=20).mean()
    df['Price_MA5_Ratio'] = df['Close/Last'] / df['MA_5']
    df['Price_MA20_Ratio'] = df['Close/Last'] / df['MA_20']
    
    # Volatility
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Range i pozycja
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close/Last']
    df['Price_Position'] = (df['Close/Last'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    
    # Momentum
    df['Momentum_5'] = df['Close/Last'] / df['Close/Last'].shift(5) - 1
    df['Momentum_10'] = df['Close/Last'] / df['Close/Last'].shift(10) - 1
    
    # Target (przyszły zwrot)
    df['Target_Return'] = df['Returns'].shift(-1)
    
    return df

def plot_training_history(history):
    """Wykres historii trenowania"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Oceń model klasyfikacyjny
    
    Parametry:
    - y_true: prawdziwe etykiety
    - y_pred: przewidywane etykiety
    - y_proba: prawdopodobieństwa (opcjonalne)
    """
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Baseline (zawsze większość): {max(np.bincount(y_true)) / len(y_true):.1%}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Spadek', 'Wzrost']))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Spadek', 'Wzrost']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Dodaj wartości
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), 
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    
    plt.ylabel('Prawdziwe')
    plt.xlabel('Przewidywane')
    plt.tight_layout()
    plt.show()
    
    return accuracy

def simple_backtest(predictions, actual_returns, transaction_cost=0.001):
    """
    Prosty backtest strategii
    
    Parametry:
    - predictions: przewidywane kierunki (0/1)
    - actual_returns: rzeczywiste zwroty
    - transaction_cost: koszt transakcji
    """
    
    portfolio_value = 1.0  # Startujemy z 1
    position = 0  # 0 = cash, 1 = long
    
    portfolio_values = [portfolio_value]
    
    for pred, ret in zip(predictions, actual_returns):
        # Decyzja inwestycyjna
        if pred == 1 and position == 0:  # Kup
            portfolio_value *= (1 - transaction_cost)
            position = 1
        elif pred == 0 and position == 1:  # Sprzedaj
            portfolio_value *= (1 + ret - transaction_cost)
            position = 0
        
        # Aktualizuj wartość jeśli jesteś w pozycji
        if position == 1:
            portfolio_value *= (1 + ret)
        
        portfolio_values.append(portfolio_value)
    
    # Buy & hold porównanie
    buy_hold = np.cumprod(1 + actual_returns)
    
    total_return = portfolio_values[-1] - 1
    buy_hold_return = buy_hold[-1] - 1
    
    print(f"Strategia AI: {total_return:.1%}")
    print(f"Buy & Hold: {buy_hold_return:.1%}")
    print(f"Outperformance: {total_return - buy_hold_return:.1%}")
    
    # Wykres
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Strategia AI', linewidth=2)
    plt.plot([1] + list(buy_hold), label='Buy & Hold', linewidth=2)
    plt.title('Porównanie strategii')
    plt.xlabel('Dni')
    plt.ylabel('Wartość portfela')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return total_return, buy_hold_return