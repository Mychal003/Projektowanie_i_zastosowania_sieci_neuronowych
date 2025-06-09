'''Narzędzia pomocnicze (np. wizualizacje, metryki)'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from matplotlib.figure import Figure

def create_directories():
    """
    Tworzy niezbędne katalogi dla projektu, jeśli nie istnieją.
    """
    dirs = [
        './data/raw',
        './data/processed',
        './models/lstm',
        './models/cnn'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        
    print("Struktura katalogów utworzona pomyślnie.")

def plot_training_history(history: Dict[str, List[float]], title: str = "Historia treningu") -> Figure:
    """
    Tworzy wykres historii treningu (straty).
    
    Args:
        history: Słownik zawierający 'train_loss' i 'val_loss'
        title: Tytuł wykresu
        
    Returns:
        Obiekt Figure z wykresem
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Strata treningu')
    plt.plot(history['val_loss'], label='Strata walidacji')
    plt.title(title)
    plt.xlabel('Epoki')
    plt.ylabel('Strata (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig = plt.gcf()
    plt.close()
    return fig

def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    title: str = "Porównanie predykcji z wartościami rzeczywistymi",
                    scaler = None) -> Figure:
    """
    Tworzy wykres porównujący predykcje modelu z wartościami rzeczywistymi.
    
    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje modelu
        title: Tytuł wykresu
        scaler: Opcjonalny obiekt skalujący do denormalizacji danych
        
    Returns:
        Obiekt Figure z wykresem
    """
    # Jeśli przekazano skaler, odwracamy normalizację
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Wartości rzeczywiste', color='blue')
    plt.plot(y_pred, label='Predykcje', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Próbki')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig = plt.gcf()
    plt.close()
    return fig

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, scaler=None) -> Dict[str, float]:
    """
    Oblicza metryki wydajności dla modelu.
    
    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje modelu
        scaler: Opcjonalny obiekt skalujący do denormalizacji danych
        
    Returns:
        Słownik z metrykami (MSE, RMSE, MAE, R²)
    """
    # Jeśli przekazano skaler, odwracamy normalizację
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def print_metrics(metrics: Dict[str, float]):
    """
    Wyświetla metryki w czytelnym formacie.
    
    Args:
        metrics: Słownik z metrykami
    """
    print("\n=== Metryki wydajności modelu ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

def plot_training_comparison(histories: Dict[str, Dict[str, List[float]]]) -> Figure:
    """
    Tworzy wykres porównujący historię treningu różnych modeli.
    
    Args:
        histories: Słownik zawierający historię treningu dla różnych modeli
                  (klucz: nazwa modelu, wartość: słownik z 'train_loss' i 'val_loss')
                  
    Returns:
        Obiekt Figure z wykresem
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, history in histories.items():
        plt.plot(history['val_loss'], label=f'{model_name} - walidacja')
    
    plt.title('Porównanie straty walidacyjnej różnych modeli')
    plt.xlabel('Epoki')
    plt.ylabel('Strata (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig = plt.gcf()
    plt.close()
    return fig

def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]]) -> Figure:
    """
    Tworzy wykres porównujący metryki różnych modeli.
    
    Args:
        metrics: Słownik zawierający metryki dla różnych modeli
                (klucz: nazwa modelu, wartość: słownik z metrykami)
                
    Returns:
        Obiekt Figure z wykresem
    """
    # Przekształcenie danych do formatu dla seaborn
    df_metrics = []
    
    for model_name, model_metrics in metrics.items():
        for metric_name, metric_value in model_metrics.items():
            df_metrics.append({
                'Model': model_name,
                'Metryka': metric_name,
                'Wartość': metric_value
            })
    
    df_metrics = pd.DataFrame(df_metrics)
    
    # Tworzenie wykresu
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df_metrics, x='Metryka', y='Wartość', hue='Model')
    
    plt.title('Porównanie metryk dla różnych modeli')
    plt.xlabel('Metryka')
    plt.ylabel('Wartość')
    plt.legend(title='Model')
    plt.grid(True, alpha=0.3)
    
    # Dodanie wartości na słupkach
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f')
    
    fig = plt.gcf()
    plt.close()
    return fig

def save_model(model: torch.nn.Module, 
               path: str, 
               metadata: Optional[Dict[str, Any]] = None):
    """
    Zapisuje model wraz z metadanymi.
    
    Args:
        model: Model PyTorch do zapisania
        path: Ścieżka do zapisania modelu
        metadata: Opcjonalne metadane (parametry, wydajność itp.)
    """
    # Tworzenie katalogu jeśli nie istnieje
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Zapisywanie modelu
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, path)
    
    print(f"Model zapisany w {path}")

def load_model(model: torch.nn.Module, path: str) -> Tuple[torch.nn.Module, Optional[Dict[str, Any]]]:
    """
    Wczytuje model wraz z metadanymi.
    
    Args:
        model: Niezainicjalizowany model PyTorch
        path: Ścieżka do wczytania modelu
        
    Returns:
        Tuple (model, metadata)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    metadata = checkpoint.get('metadata', None)
    
    return model, metadata