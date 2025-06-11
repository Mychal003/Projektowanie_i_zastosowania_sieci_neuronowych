"""
Analiza i wizualizacja wyników modelu LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import torch
from typing import Dict, Tuple, Any

# Ustawienia wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LSTMAnalysis:
    """Klasa do analizy wyników modelu LSTM."""
    
    def __init__(self, model, trainer, data: Dict[str, Any], df: pd.DataFrame):
        """
        Inicjalizacja analizy.
        
        Parameters:
        -----------
        model : nn.Module
            Wytrenowany model LSTM
        trainer : LSTMTrainer
            Obiekt trenera z historią treningu
        data : dict
            Słownik z danymi (z prepare_data_for_lstm)
        df : pd.DataFrame
            Oryginalny DataFrame z danymi
        """
        self.model = model
        self.trainer = trainer
        self.data = data
        self.df = df
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Automatycznie wykonaj predykcje
        self._make_predictions()
    
    def _make_predictions(self):
        """Wykonuje predykcje na zbiorze testowym."""
        # Predykcja zwrotów
        self.y_pred_returns = self.trainer.predict(self.data['X_test'])
        
        # Denormalizacja
        self.y_test_returns = self.data['target_scaler'].inverse_transform(self.data['y_test'])
        self.y_pred_returns = self.data['target_scaler'].inverse_transform(self.y_pred_returns)
        
        # Oblicz metryki zwrotów
        self.metrics_returns = {
            'mse': mean_squared_error(self.y_test_returns, self.y_pred_returns),
            'rmse': np.sqrt(mean_squared_error(self.y_test_returns, self.y_pred_returns)),
            'mae': np.mean(np.abs(self.y_test_returns - self.y_pred_returns)),
            'r2': r2_score(self.y_test_returns, self.y_pred_returns)
        }
        
        # Dokładność kierunku
        self.actual_direction = (self.y_test_returns > 0).astype(int)
        self.pred_direction = (self.y_pred_returns > 0).astype(int)
        self.direction_accuracy = (self.actual_direction == self.pred_direction).mean()
        
        # Konwersja na ceny
        self._convert_to_prices()
    
    def _convert_to_prices(self):
        """Konwertuje zwroty na ceny."""
        # Znajdź parametry podziału danych
        seq_length = 30  # Domyślna wartość jeśli nie ma w data
        if hasattr(self.data, 'get'):
            seq_length = self.data.get('seq_length', 30)
        
        # Oblicz rzeczywiste rozmiary na podstawie długości danych
        total_samples = len(self.data['X_train']) + len(self.data['X_val']) + len(self.data['X_test'])
        
        # Znajdź indeks startowy dla danych testowych
        # Musimy uwzględnić że dane zostały podzielone po utworzeniu sekwencji
        train_val_size = len(self.data['X_train']) + len(self.data['X_val'])
        
        # Indeks w oryginalnym DataFrame gdzie zaczynają się dane testowe
        # Dodajemy seq_length bo to jest offset od początku
        test_start_idx = seq_length + train_val_size
        
        # Sprawdź czy nie wychodzimy poza zakres
        max_idx = len(self.df) - 1
        test_end_idx = min(test_start_idx + len(self.y_test_returns), max_idx)
        
        # Pobierz rzeczywiste ceny
        if test_end_idx <= len(self.df):
            test_indices = range(test_start_idx, test_end_idx)
            self.actual_prices = self.df['Close/Last'].iloc[test_indices].values
        else:
            # Jeśli indeksy są za duże, użyj ostatnich dostępnych danych
            available_test_size = len(self.df) - test_start_idx
            self.actual_prices = self.df['Close/Last'].iloc[-available_test_size:].values
            print(f"⚠️ Adjusted test data size to available data: {available_test_size}")
        
        # Oblicz przewidywane ceny
        if len(self.actual_prices) > 0:
            initial_price = self.actual_prices[0]
            self.predicted_prices = [initial_price]
            
            # Użyj tylko tyle predykcji ile mamy rzeczywistych cen
            n_predictions = min(len(self.y_pred_returns), len(self.actual_prices) - 1)
            
            for i in range(n_predictions):
                next_price = self.predicted_prices[-1] * (1 + self.y_pred_returns[i][0])
                self.predicted_prices.append(next_price)
            
            self.predicted_prices = np.array(self.predicted_prices[1:])
            
            # Dopasuj długości
            min_len = min(len(self.actual_prices), len(self.predicted_prices))
            self.actual_prices = self.actual_prices[:min_len]
            self.predicted_prices = self.predicted_prices[:min_len]
            
            # Metryki cen
            if len(self.actual_prices) > 0 and len(self.predicted_prices) > 0:
                self.metrics_prices = {
                    'mse': mean_squared_error(self.actual_prices, self.predicted_prices),
                    'rmse': np.sqrt(mean_squared_error(self.actual_prices, self.predicted_prices)),
                    'mae': np.mean(np.abs(self.actual_prices - self.predicted_prices)),
                    'r2': r2_score(self.actual_prices, self.predicted_prices)
                }
            else:
                print("⚠️ Brak wystarczających danych do obliczenia metryk cen")
                self.metrics_prices = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
        else:
            print("⚠️ Nie można pobrać cen testowych")
            self.actual_prices = np.array([])
            self.predicted_prices = np.array([])
            self.metrics_prices = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
    
    def plot_training_history(self, figsize=(12, 5)):
        """Wykres historii treningu."""
        history = {
            'train_loss': self.trainer.train_losses,
            'val_loss': self.trainer.val_losses
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pełna historia
        ax1.plot(history['train_loss'], label='Strata treningowa', linewidth=2)
        ax1.plot(history['val_loss'], label='Strata walidacyjna', linewidth=2)
        ax1.set_title('Historia treningu', fontsize=14)
        ax1.set_xlabel('Epoki')
        ax1.set_ylabel('Strata')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ostatnie 50 epok
        if len(history['train_loss']) > 50:
            ax2.plot(history['train_loss'][-50:], label='Strata treningowa', linewidth=2)
            ax2.plot(history['val_loss'][-50:], label='Strata walidacyjna', linewidth=2)
            ax2.set_title('Ostatnie 50 epok', fontsize=14)
            ax2.set_xlabel('Epoki')
            ax2.set_ylabel('Strata')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Informacje o treningu
        print(f"📊 Końcowa strata treningowa: {history['train_loss'][-1]:.6f}")
        print(f"📊 Końcowa strata walidacyjna: {history['val_loss'][-1]:.6f}")
        print(f"📊 Liczba epok: {len(history['train_loss'])}")
    
    def plot_predictions(self, figsize=(15, 10)):
        """Kompleksowa wizualizacja predykcji."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Porównanie cen
        axes[0, 0].plot(self.actual_prices, label='Rzeczywiste', color='blue', linewidth=2)
        axes[0, 0].plot(self.predicted_prices, label='Przewidywane', color='red', alpha=0.8, linewidth=2)
        axes[0, 0].set_title(f'Porównanie cen akcji (R² = {self.metrics_prices["r2"]:.3f})', fontsize=14)
        axes[0, 0].set_xlabel('Dni')
        axes[0, 0].set_ylabel('Cena ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ostatnie 50 dni
        n_last = min(50, len(self.actual_prices))
        axes[0, 1].plot(range(n_last), self.actual_prices[-n_last:], 'bo-', 
                        label='Rzeczywiste', markersize=4)
        axes[0, 1].plot(range(n_last), self.predicted_prices[-n_last:], 'rx-', 
                        label='Przewidywane', markersize=4)
        axes[0, 1].set_title(f'Ostatnie {n_last} dni', fontsize=14)
        axes[0, 1].set_xlabel('Dni')
        axes[0, 1].set_ylabel('Cena ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot zwrotów
        axes[1, 0].scatter(self.y_test_returns, self.y_pred_returns, alpha=0.5, s=20)
        axes[1, 0].plot([self.y_test_returns.min(), self.y_test_returns.max()],
                        [self.y_test_returns.min(), self.y_test_returns.max()], 
                        'r--', lw=2)
        axes[1, 0].set_title(f'Zwroty: Rzeczywiste vs Przewidywane (R² = {self.metrics_returns["r2"]:.3f})', 
                            fontsize=14)
        axes[1, 0].set_xlabel('Rzeczywiste zwroty')
        axes[1, 0].set_ylabel('Przewidywane zwroty')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Macierz pomyłek dla kierunku
        cm = confusion_matrix(self.actual_direction, self.pred_direction)
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].figure.colorbar(im, ax=axes[1, 1])
        axes[1, 1].set(xticks=np.arange(2),
                       yticks=np.arange(2),
                       xticklabels=['Spadek', 'Wzrost'],
                       yticklabels=['Spadek', 'Wzrost'],
                       title=f'Macierz pomyłek kierunku (Dokładność: {self.direction_accuracy:.1%})',
                       ylabel='Rzeczywisty kierunek',
                       xlabel='Przewidywany kierunek')
        
        # Wartości w macierzy
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_analysis(self, figsize=(14, 8)):
        """Analiza jakości predykcji."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Histogram błędów
        errors = self.predicted_prices - self.actual_prices
        axes[0, 0].hist(errors, bins=50, alpha=0.7, color='darkblue', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Rozkład błędów predykcji', fontsize=14)
        axes[0, 0].set_xlabel('Błąd ($)')
        axes[0, 0].set_ylabel('Częstość')
        
        # 2. Błędy w czasie
        axes[0, 1].plot(errors, linewidth=1, color='darkred', alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].fill_between(range(len(errors)), 0, errors, alpha=0.3, color='red')
        axes[0, 1].set_title('Błędy predykcji w czasie', fontsize=14)
        axes[0, 1].set_xlabel('Dni')
        axes[0, 1].set_ylabel('Błąd ($)')
        
        # 3. Q-Q plot błędów
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot błędów', fontsize=14)
        
        # 4. Względne błędy
        relative_errors = (self.predicted_prices - self.actual_prices) / self.actual_prices * 100
        axes[1, 1].hist(relative_errors, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Rozkład błędów względnych', fontsize=14)
        axes[1, 1].set_xlabel('Błąd względny (%)')
        axes[1, 1].set_ylabel('Częstość')
        
        plt.tight_layout()
        plt.show()
        
        # Statystyki błędów
        print("\n📊 ANALIZA BŁĘDÓW:")
        print(f"Średni błąd: ${np.mean(errors):.2f}")
        print(f"Mediana błędu: ${np.median(errors):.2f}")
        print(f"Odchylenie standardowe błędu: ${np.std(errors):.2f}")
        print(f"Średni błąd względny: {np.mean(relative_errors):.2f}%")
        print(f"MAPE: {np.mean(np.abs(relative_errors)):.2f}%")
    
    def validate_predictions(self):
        """Walidacja jakości predykcji."""
        print("\n" + "="*60)
        print("🔍 WALIDACJA PREDYKCJI")
        print("="*60)
        
        # Analiza zwrotów
        print("\n📈 ANALIZA ZWROTÓW:")
        print(f"Średnia przewidywanych zwrotów: {np.mean(self.y_pred_returns):.6f}")
        print(f"Std przewidywanych zwrotów: {np.std(self.y_pred_returns):.6f}")
        print(f"Min/Max przewidywanych: {np.min(self.y_pred_returns):.6f} / {np.max(self.y_pred_returns):.6f}")
        print(f"\nŚrednia rzeczywistych zwrotów: {np.mean(self.y_test_returns):.6f}")
        print(f"Std rzeczywistych zwrotów: {np.std(self.y_test_returns):.6f}")
        
        
        # Metryki
        print("\n📊 METRYKI NA CENACH:")
        print(f"R²: {self.metrics_prices['r2']:.3f}")
        print(f"RMSE: ${self.metrics_prices['rmse']:.2f}")
        print(f"MAE: ${self.metrics_prices['mae']:.2f}")
        
        print("\n📊 METRYKI NA ZWROTACH:")
        print(f"R²: {self.metrics_returns['r2']:.3f}")
        print(f"MSE: {self.metrics_returns['mse']:.6f}")
        
        print(f"\n🎯 DOKŁADNOŚĆ KIERUNKU: {self.direction_accuracy:.1%}")
        
        # Ocena wyników
        print("\n📋 OCENA MODELU:")
        if self.direction_accuracy > 0.55:
            print("✅ Model przewiduje kierunek lepiej niż losowo!")
        else:
            print("❌ Model ma problemy z przewidywaniem kierunku")
            
        if self.metrics_prices['r2'] > 0:
            print("✅ Pozytywne R² na cenach")
        else:
            print("❌ Negatywne R² na cenach")
            
        if self.metrics_returns['r2'] > 0.01:
            print("✅ Model wychwytuje pewne wzorce w zwrotach")
        else:
            print("⚠️ Model ma trudności z przewidywaniem zwrotów")
        
        print("\n" + "="*60)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generuje podsumowanie analizy."""
        summary = {
            'model_params': {
                'architecture': str(self.model),
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training': {
                'final_train_loss': self.trainer.train_losses[-1],
                'final_val_loss': self.trainer.val_losses[-1],
                'epochs_trained': len(self.trainer.train_losses)
            },
            'metrics_prices': self.metrics_prices,
            'metrics_returns': self.metrics_returns,
            'direction_accuracy': float(self.direction_accuracy),
            'data_info': {
                'train_samples': len(self.data['X_train']),
                'val_samples': len(self.data['X_val']),
                'test_samples': len(self.data['X_test']),
                'features': self.data['input_size']
            }
        }
        
        return summary
    
    def plot_all_analysis(self):
        """Wykonuje pełną analizę z wszystkimi wykresami."""
        print("\n" + "="*60)
        print("🚀 PEŁNA ANALIZA MODELU LSTM")
        print("="*60)
        
        # 1. Historia treningu
        print("\n📈 HISTORIA TRENINGU:")
        self.plot_training_history()
        
        # 2. Predykcje
        print("\n📊 WIZUALIZACJA PREDYKCJI:")
        self.plot_predictions()
        
        # 3. Analiza błędów
        print("\n🔍 ANALIZA BŁĘDÓW:")
        self.plot_prediction_analysis()
        
        # 4. Walidacja
        self.validate_predictions()
        
        # 5. Podsumowanie
        summary = self.generate_summary()
        print("\n📋 PODSUMOWANIE:")
        print(f"Model: {summary['model_params']['total_params']:,} parametrów")
        print(f"Dokładność kierunku: {summary['direction_accuracy']:.1%}")
        print(f"R² (ceny): {summary['metrics_prices']['r2']:.3f}")
        print(f"RMSE: ${summary['metrics_prices']['rmse']:.2f}")
        
        return summary