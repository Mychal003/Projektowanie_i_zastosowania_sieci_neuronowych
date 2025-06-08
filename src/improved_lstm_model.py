# Ulepszona implementacja LSTM dla prognozowania cen akcji

## 1. Import bibliotek
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Union, Any

# Dodanie katalogu głównego do ścieżki
sys.path.append('..')

# Import z nowego modułu (zakładając, że zapisałeś ulepszoną wersję)
from src.improved_lstm_model import (
    ImprovedLSTMModel, 
    ImprovedLSTMTrainer, 
    prepare_lstm_data_improved,
    create_advanced_features
)
from src.utils import (
    create_directories, 
    plot_training_history, 
    plot_predictions, 
    evaluate_model, 
    print_metrics,
    save_model
)

# Ustawienie stylu dla wykresów
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
sns.set_context('talk')

# Sprawdzenie dostępności CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

# Tworzenie potrzebnych katalogów
create_directories()

## 2. Wczytanie i wstępna analiza danych
# Wczytanie danych
data_path = "../data/processed/cleaned_data.csv"

if not os.path.exists(data_path):
    from src.data_preprocessing import preprocess
    df = preprocess()
else:
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])

print(f"Liczba wierszy w danych: {len(df)}")
print(f"Zakres dat: {df['Date'].min()} - {df['Date'].max()}")
df.head()

## 3. Inżynieria cech - KLUCZOWA ZMIANA!
# Dodanie zaawansowanych cech technicznych
df_enhanced = create_advanced_features(df)

print(f"Liczba cech przed: {len(df.columns)}")
print(f"Liczba cech po: {len(df_enhanced.columns)}")
print("\nNowe cechy:")
print([col for col in df_enhanced.columns if col not in df.columns])

# Wizualizacja niektórych nowych cech
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# RSI
axes[0, 0].plot(df_enhanced['Date'], df_enhanced['RSI'])
axes[0, 0].axhline(y=70, color='r', linestyle='--', alpha=0.5)
axes[0, 0].axhline(y=30, color='g', linestyle='--', alpha=0.5)
axes[0, 0].set_title('RSI (Relative Strength Index)')
axes[0, 0].set_ylabel('RSI')

# Volatility
axes[0, 1].plot(df_enhanced['Date'], df_enhanced['Volatility'])
axes[0, 1].set_title('Zmienność (20-dniowa)')
axes[0, 1].set_ylabel('Volatility')

# Bollinger Bands Position
axes[1, 0].plot(df_enhanced['Date'], df_enhanced['BB_position'])
axes[1, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Pozycja w Bollinger Bands')
axes[1, 0].set_ylabel('BB Position')

# Volume Ratio
axes[1, 1].plot(df_enhanced['Date'], df_enhanced['Volume_ratio'])
axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Stosunek wolumenu do średniej')
axes[1, 1].set_ylabel('Volume Ratio')

plt.tight_layout()
plt.show()

## 4. Przygotowanie danych z ulepszoną metodą
# Parametry
seq_length = 30  # Dłuższa sekwencja dla lepszego kontekstu
batch_size = 64  # Większy batch size

# Przygotowanie danych
data = prepare_lstm_data_improved(
    df=df_enhanced,
    seq_length=seq_length,
    target_days=1,  # Predykcja na 1 dzień do przodu
    test_size=0.2,
    val_size=0.1,
    batch_size=batch_size
)

print(f"Liczba cech wejściowych: {data['input_size']}")
print(f"Cechy użyte w modelu: {data['feature_names']}")
print(f"\nLiczba próbek:")
print(f"  Treningowych: {len(data['X_train'])}")
print(f"  Walidacyjnych: {len(data['X_val'])}")
print(f"  Testowych: {len(data['X_test'])}")
print(f"\nKształt danych: {data['X_train'].shape}")

## 5. Definicja ulepszonego modelu LSTM
# Parametry modelu
model_params = {
    'input_size': data['input_size'],
    'hidden_size': 128,      # Zwiększony rozmiar
    'num_layers': 3,         # Więcej warstw
    'output_size': 1,
    'dropout': 0.3           # Większy dropout
}

# Inicjalizacja modelu
lstm_model = ImprovedLSTMModel(**model_params)
lstm_model = lstm_model.to(device)

# Wyświetlenie struktury modelu
print(f"Parametry modelu: {sum(p.numel() for p in lstm_model.parameters()):,}")
print("\nArchitektura:")
print(lstm_model)

## 6. Trenowanie z ulepszonymi strategiami
# Inicjalizacja trenera z nowymi ustawieniami
trainer = ImprovedLSTMTrainer(
    model=lstm_model,
    learning_rate=0.0005,  # Niższy learning rate
    device=device
)

# Parametry trenowania
epochs = 150  # Więcej epok
early_stopping_patience = 20

print("Rozpoczynam trenowanie z następującymi ulepszeniami:")
print("- Bidirectional LSTM")
print("- Huber Loss (odporna na outliers)")
print("- AdamW optimizer z learning rate scheduler")
print("- Gradient clipping")
print("- Batch normalization\n")

# Trenowanie modelu
history = trainer.fit(
    train_loader=data['train_loader'],
    val_loader=data['val_loader'],
    epochs=epochs,
    early_stopping_patience=early_stopping_patience,
    verbose=True
)

## 7. Analiza wyników treningu
# Wykres historii treningu z lepszą wizualizacją
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
ax1.plot(history['train_loss'], label='Strata treningowa', alpha=0.8)
ax1.plot(history['val_loss'], label='Strata walidacyjna', alpha=0.8)
ax1.set_title('Historia treningu modelu LSTM')
ax1.set_xlabel('Epoki')
ax1.set_ylabel('Strata (Huber Loss)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Zoom na ostatnie epoki
last_epochs = 50
if len(history['train_loss']) > last_epochs:
    ax2.plot(history['train_loss'][-last_epochs:], label='Strata treningowa')
    ax2.plot(history['val_loss'][-last_epochs:], label='Strata walidacyjna')
    ax2.set_title(f'Ostatnie {last_epochs} epok')
    ax2.set_xlabel('Epoki')
    ax2.set_ylabel('Strata (Huber Loss)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## 8. Ewaluacja modelu na zbiorze testowym
# Predykcja
y_pred = trainer.predict(data['X_test'])

# Obliczenie metryk
metrics = evaluate_model(data['y_test'], y_pred, scaler=data['target_scaler'])
print("\n=== WYNIKI PO ULEPSZENIACH ===")
print_metrics(metrics)

# Porównanie z baseline (prosta średnia)
baseline_pred = np.full_like(y_pred, np.mean(data['y_train']))
baseline_metrics = evaluate_model(data['y_test'], baseline_pred, scaler=data['target_scaler'])
print("\n=== BASELINE (średnia) ===")
print_metrics(baseline_metrics)

print(f"\nPoprawa względem baseline:")
print(f"RMSE: {((baseline_metrics['RMSE'] - metrics['RMSE']) / baseline_metrics['RMSE'] * 100):.2f}%")
print(f"R²: z {baseline_metrics['R²']:.3f} do {metrics['R²']:.3f}")

## 9. Zaawansowana wizualizacja predykcji
# Denormalizacja
y_test_denorm = data['target_scaler'].inverse_transform(data['y_test'])
y_pred_denorm = data['target_scaler'].inverse_transform(y_pred)

# Wykres z różnymi perspektywami
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Pełne porównanie
axes[0, 0].plot(y_test_denorm, label='Wartości rzeczywiste', color='blue', alpha=0.7)
axes[0, 0].plot(y_pred_denorm, label='Predykcje LSTM', color='red', alpha=0.7)
axes[0, 0].set_title('Porównanie predykcji z wartościami rzeczywistymi')
axes[0, 0].set_xlabel('Próbki')
axes[0, 0].set_ylabel('Cena akcji ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Ostatnie 100 próbek (zoom)
n_last = min(100, len(y_test_denorm))
axes[0, 1].plot(y_test_denorm[-n_last:], label='Rzeczywiste', color='blue', marker='o', markersize=3)
axes[0, 1].plot(y_pred_denorm[-n_last:], label='Predykcje', color='red', marker='x', markersize=3)
axes[0, 1].set_title(f'Ostatnie {n_last} predykcji (szczegóły)')
axes[0, 1].set_xlabel('Próbki')
axes[0, 1].set_ylabel('Cena akcji ($)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Scatter plot
axes[1, 0].scatter(y_test_denorm, y_pred_denorm, alpha=0.5, s=10)
axes[1, 0].plot([y_test_denorm.min(), y_test_denorm.max()], 
                [y_test_denorm.min(), y_test_denorm.max()], 
                'r--', lw=2)
axes[1, 0].set_title('Rzeczywiste vs Predykcje')
axes[1, 0].set_xlabel('Wartości rzeczywiste ($)')
axes[1, 0].set_ylabel('Predykcje ($)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Histogram błędów
errors = y_test_denorm.flatten() - y_pred_denorm.flatten()
axes[1, 1].hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title('Rozkład błędów predykcji')
axes[1, 1].set_xlabel('Błąd ($)')
axes[1, 1].set_ylabel('Częstość')
axes[1, 1].grid(True, alpha=0.3)

# Dodaj statystyki błędów
mean_error = np.mean(errors)
std_error = np.std(errors)
axes[1, 1].text(0.02, 0.98, f'Średni błąd: ${mean_error:.2f}\nOdch. std.: ${std_error:.2f}', 
                transform=axes[1, 1].transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

## 10. Analiza ważności cech
# Sprawdzenie które cechy są najważniejsze
# Możemy to zrobić przez permutation importance lub przez analizę wag

# Prosty sposób - sprawdzenie jak zmienia się wynik gdy usuniemy cechę
feature_importance = {}

for i, feature_name in enumerate(data['feature_names']):
    # Kopia danych testowych
    X_test_copy = data['X_test'].copy()
    
    # "Wyłączenie" cechy przez ustawienie na średnią
    X_test_copy[:, :, i] = np.mean(X_test_copy[:, :, i])
    
    # Predykcja
    y_pred_modified = trainer.predict(X_test_copy)
    
    # Metryka
    mse_modified = np.mean((y_pred_modified - data['y_test']) ** 2)
    mse_original = np.mean((y_pred - data['y_test']) ** 2)
    
    # Ważność = wzrost błędu po usunięciu cechy
    feature_importance[feature_name] = (mse_modified - mse_original) / mse_original * 100

# Sortowanie i wyświetlenie
importance_df = pd.DataFrame(list(feature_importance.items()), 
                           columns=['Feature', 'Importance (%)'])
importance_df = importance_df.sort_values('Importance (%)', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance (%)'])
plt.xlabel('Wzrost błędu po usunięciu cechy (%)')
plt.title('Ważność cech w modelu LSTM')
plt.tight_layout()
plt.show()

print("\nTop 5 najważniejszych cech:")
print(importance_df.head())

## 11. Zapisanie ulepszonego modelu
# Przygotowanie rozszerzonych metadanych
metadata = {
    'model_params': model_params,
    'feature_names': data['feature_names'],
    'seq_length': seq_length,
    'metrics': metrics,
    'baseline_metrics': baseline_metrics,
    'training_history': {
        'train_loss': [float(loss) for loss in history['train_loss']],
        'val_loss': [float(loss) for loss in history['val_loss']]
    },
    'feature_importance': feature_importance,
    'model_type': 'ImprovedLSTM_Bidirectional',
    'improvements': [
        'Bidirectional LSTM',
        'Huber Loss',
        'AdamW optimizer',
        'Learning rate scheduler',
        'Advanced feature engineering',
        'Gradient clipping',
        'Batch normalization'
    ]
}

# Zapisanie modelu
save_model(
    model=lstm_model,
    path='../models/lstm/improved_lstm_model.pth',
    metadata=metadata
)

print("\nModel zapisany pomyślnie!")
print(f"Osiągnięte R²: {metrics['R²']:.3f} (poprzednio: -0.733)")

## 12. Podsumowanie i wnioski
print("\n" + "="*50)
print("PODSUMOWANIE ULEPSZEŃ")
print("="*50)

improvements = [
    ("Architektura", "Bidirectional LSTM z 3 warstwami"),
    ("Loss function", "Huber Loss zamiast MSE"),
    ("Optymalizator", "AdamW z learning rate scheduler"),
    ("Feature engineering", f"{len(data['feature_names'])} cech technicznych"),
    ("Sekwencja", f"{seq_length} dni (poprzednio 20)"),
    ("Batch size", f"{batch_size} (poprzednio 32)")
]

for name, value in improvements:
    print(f"{name:.<25} {value}")

print(f"\nWynik końcowy:")
print(f"  R² Score: {metrics['R²']:.3f} (poprawa o {metrics['R²'] - (-0.733):.3f})")
print(f"  RMSE: ${metrics['RMSE']:.2f}")

# Sugestie dalszych ulepszeń
print("\n" + "="*50)
print("SUGESTIE DALSZYCH ULEPSZEŃ")
print("="*50)
print("1. Ensemble - połączenie LSTM z innymi modelami (Random Forest, XGBoost)")
print("2. Attention mechanism - dodanie warstwy attention do LSTM")
print("3. Więcej danych - użycie dłuższej historii lub danych z wielu spółek")
print("4. External features - dodanie wskaźników makroekonomicznych")
print("5. Hyperparameter tuning - użycie Optuna lub GridSearch")