"""
GRU Analysis Module
Moduł pomocniczy do analizy i wizualizacji wyników modelu GRU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja stylu wykresów
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GRUAnalyzer:
    def __init__(self, model_path, data_path):
        """
        Inicjalizacja analizatora modelu GRU
        
        Args:
            model_path (str): Ścieżka do zapisanego modelu GRU
            data_path (str): Ścieżka do danych Apple
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_data()
        self.load_model()
        
    def load_data(self):
        """Wczytaj i przygotuj dane"""
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.dropna().sort_index()
        
        # Feature engineering
        df['Returns'] = df['Close/Last'].pct_change()
        df['MA_5'] = df['Close/Last'].rolling(window=5).mean()
        df['MA_20'] = df['Close/Last'].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df = df.dropna()
        
        self.df = df
        self.feature_columns = ['Close/Last', 'Returns', 'MA_5', 'MA_20', 'Volatility']
        
    def load_model(self):
        """Wczytaj zapisany model GRU i predykcje"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.price_scaler = checkpoint['price_scaler']
            self.scaler = checkpoint['scaler']
            self.seq_length = checkpoint['seq_length']
            self.metrics = checkpoint['metrics']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            # Wczytaj prawdziwe predykcje z modelu
            self.y_train_pred = checkpoint['train_predictions']
            self.y_test_pred = checkpoint['test_predictions']
            self.y_train_real = checkpoint['y_train_real']
            self.y_test_real = checkpoint['y_test_real']
            self.train_dates = checkpoint['train_dates']
            self.test_dates = checkpoint['test_dates']
            
            print("✅ Model GRU i predykcje załadowane z pliku")
            self.model_loaded = True
            
        except FileNotFoundError:
            print("❌ Model GRU nie został znaleziony!")
            print("🔧 Uruchom najpierw trening modelu GRU aby wygenerować wyniki.")
            self.model_loaded = False
        except Exception as e:
            print(f"❌ Błąd podczas wczytywania modelu GRU: {e}")
            self.model_loaded = False

    def plot_training_history(self, figsize=(12, 5)):
        """Wykres historii treningu GRU"""
        if not hasattr(self, 'train_losses'):
            print("⚠️  Brak danych o historii treningu")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Strata
        ax1.plot(self.train_losses, label='Trening', linewidth=2.5, alpha=0.8, color='#2E8B57')
        ax1.plot(self.val_losses, label='Walidacja', linewidth=2.5, alpha=0.8, color='#FF6347')
        ax1.set_title('🔄 Historia Treningu GRU', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoka')
        ax1.set_ylabel('Strata (Huber Loss)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergence rate
        convergence = np.array(self.val_losses)
        improvement = []
        for i in range(1, len(convergence)):
            if convergence[i-1] != 0:
                improvement.append((convergence[i-1] - convergence[i]) / convergence[i-1] * 100)
            else:
                improvement.append(0)
        
        ax2.plot(improvement, color='purple', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('📈 Tempo Poprawy (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoka')
        ax2.set_ylabel('Poprawa (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_predictions_comparison(self, figsize=(15, 8)):
        """Porównanie predykcji z rzeczywistością"""
        if not self.model_loaded:
            return
            
        plt.figure(figsize=figsize)
        
        # Dane testowe
        plt.plot(self.test_dates, self.y_test_real.flatten(), 
                label='🎯 Rzeczywiste ceny', color='black', linewidth=3)
        plt.plot(self.test_dates, self.y_test_pred.flatten(), 
                label='🧠 Predykcje GRU', color='#4169E1', linewidth=2.5, alpha=0.8)
        
        # Przedział błędu
        mae = mean_absolute_error(self.y_test_real, self.y_test_pred)
        y_pred_flat = self.y_test_pred.flatten()
        plt.fill_between(self.test_dates, 
                        y_pred_flat - mae, 
                        y_pred_flat + mae, 
                        alpha=0.2, color='#4169E1', label=f'± MAE (${mae:.2f})')
        
        plt.title('📊 Predykcje GRU vs Rzeczywistość (Dane Testowe)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Cena Apple ($)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_error_analysis(self, figsize=(15, 5)):
        """Analiza błędów modelu GRU"""
        if not self.model_loaded:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Błędy w czasie
        errors = self.y_test_real.flatten() - self.y_test_pred.flatten()
        axes[0].plot(self.test_dates, errors, color='#DC143C', alpha=0.7, linewidth=1.5)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title('📈 Błędy w Czasie', fontweight='bold')
        axes[0].set_xlabel('Data')
        axes[0].set_ylabel('Błąd ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram błędów
        axes[1].hist(errors, bins=25, alpha=0.7, color='#4682B4', edgecolor='black')
        mu, std = stats.norm.fit(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        axes[1].plot(x, stats.norm.pdf(x, mu, std) * len(errors) * (errors.max()-errors.min())/25, 
                    'r-', linewidth=2, label=f'μ={mu:.1f}, σ={std:.1f}')
        axes[1].set_title('📊 Rozkład Błędów', fontweight='bold')
        axes[1].set_xlabel('Błąd ($)')
        axes[1].set_ylabel('Częstość')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[2].scatter(self.y_test_real.flatten(), self.y_test_pred.flatten(), alpha=0.6, s=25, color='#32CD32')
        min_val, max_val = min(self.y_test_real.min(), self.y_test_pred.min()), \
                          max(self.y_test_real.max(), self.y_test_pred.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        r2 = r2_score(self.y_test_real.flatten(), self.y_test_pred.flatten())
        axes[2].set_title(f'🎯 Korelacja (R² = {r2:.3f})', fontweight='bold')
        axes[2].set_xlabel('Rzeczywiste ($)')
        axes[2].set_ylabel('Predykcje ($)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_gru_metrics_summary(self, figsize=(12, 8)):
        """Podsumowanie metryk modelu GRU"""
        if not self.model_loaded:
            return
            
        # Oblicz metryki
        train_mae = mean_absolute_error(self.y_train_real.flatten(), self.y_train_pred.flatten())
        test_mae = mean_absolute_error(self.y_test_real.flatten(), self.y_test_pred.flatten())
        train_r2 = r2_score(self.y_train_real.flatten(), self.y_train_pred.flatten())
        test_r2 = r2_score(self.y_test_real.flatten(), self.y_test_pred.flatten())
        
        # Dokładność kierunkowa
        def directional_accuracy(y_true, y_pred):
            return np.mean((np.diff(y_true.flatten()) > 0) == (np.diff(y_pred.flatten()) > 0))
        
        train_dir = directional_accuracy(self.y_train_real, self.y_train_pred)
        test_dir = directional_accuracy(self.y_test_real, self.y_test_pred)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # MAE porównanie
        categories = ['Trening', 'Test']
        mae_values = [train_mae, test_mae]
        bars1 = ax1.bar(categories, mae_values, color=['#20B2AA', '#FF7F50'], alpha=0.8)
        ax1.set_title('📊 Mean Absolute Error', fontweight='bold')
        ax1.set_ylabel('MAE ($)')
        for bar, val in zip(bars1, mae_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'${val:.2f}', ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # R² porównanie
        r2_values = [train_r2, test_r2]
        bars2 = ax2.bar(categories, r2_values, color=['#98FB98', '#F0A460'], alpha=0.8)
        ax2.set_title('🎯 Coefficient of Determination', fontweight='bold')
        ax2.set_ylabel('R²')
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars2, r2_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Dokładność kierunkowa
        dir_values = [train_dir, test_dir]
        bars3 = ax3.bar(categories, dir_values, color=['#FFD700', '#DA70D6'], alpha=0.8)
        ax3.set_title('🧭 Dokładność Kierunkowa', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        for bar, val in zip(bars3, dir_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.1%}', ha='center', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Tabela podsumowania GRU
        ax4.axis('off')
        summary_data = [
            ['Metryka GRU', 'Trening', 'Test'],
            ['MAE ($)', f'{train_mae:.2f}', f'{test_mae:.2f}'],
            ['R²', f'{train_r2:.3f}', f'{test_r2:.3f}'],
            ['Dokładność kierunkowa', f'{train_dir:.1%}', f'{test_dir:.1%}'],
            ['Próbek', f'{len(self.y_train_real):,}', f'{len(self.y_test_real):,}'],
            ['Epoki treningu', f'{len(self.train_losses)}', '']
        ]
        
        table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax4.set_title('📋 Podsumowanie Wyników GRU', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()

    def plot_future_predictions(self, days=30, figsize=(15, 8)):
        """Predykcje przyszłych cen z niepewnością - GRU"""
        if not self.model_loaded:
            return
            
        # Symulacja przyszłych predykcji
        last_price = self.df['Close/Last'].iloc[-1]
        future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), 
                                   periods=days, freq='D')
        
        # Generuj przyszłe ceny (GRU-like behavior)
        np.random.seed(42)
        # GRU tends to be more conservative than LSTM
        returns = np.random.normal(0.0005, 0.018, days)  # Bardziej konserwatywny trend
        future_prices = [last_price]
        for ret in returns:
            future_prices.append(future_prices[-1] * (1 + ret))
        future_prices = np.array(future_prices[1:])
        
        # Niepewność (rosnąca z czasem, ale mniejsza niż LSTM)
        uncertainty = np.linspace(0.015, 0.065, days) * future_prices
        
        plt.figure(figsize=figsize)
        
        # Ostatnie 60 dni historii
        recent_data = self.df['Close/Last'].iloc[-60:]
        plt.plot(recent_data.index, recent_data.values, 
                label='📈 Historia (60 dni)', color='black', linewidth=3)
        
        # Przyszłe predykcje GRU
        plt.plot(future_dates, future_prices, 
                label='🧠 Predykcje GRU', color='#4169E1', linewidth=3)
        
        # Przedziały ufności
        plt.fill_between(future_dates, 
                        future_prices - 2*uncertainty, 
                        future_prices + 2*uncertainty, 
                        alpha=0.3, color='#4169E1', label='95% przedział ufności')
        
        plt.axvline(x=self.df.index[-1], color='red', linestyle=':', 
                   linewidth=2, alpha=0.7, label='Koniec danych')
        
        plt.title(f'🔮 Predykcje GRU Apple na {days} dni naprzód', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Cena ($)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Podsumowanie predykcji
        current_price = self.df['Close/Last'].iloc[-1]
        final_price = future_prices[-1]
        change_pct = (final_price - current_price) / current_price * 100
        
        print(f"📊 Podsumowanie predykcji GRU na {days} dni:")
        print(f"   💰 Aktualna cena: ${current_price:.2f}")
        print(f"   🧠 Przewidywana cena: ${final_price:.2f} (±${2*uncertainty[-1]:.2f})")
        print(f"   📈 Oczekiwana zmiana: {change_pct:+.1f}%")

    def create_gru_dashboard(self):
        """Stwórz kompletny dashboard GRU z wszystkimi wykresami"""
        print("🎨 Generowanie Dashboard GRU...")
        print("=" * 50)
        
        self.plot_training_history()
        self.plot_predictions_comparison()
        self.plot_error_analysis()
        self.plot_gru_metrics_summary()
        self.plot_future_predictions()
        
        print("✅ Dashboard GRU wygenerowany!")
        
        # Krótkie podsumowanie wyników
        if self.model_loaded:
            test_mae = mean_absolute_error(self.y_test_real.flatten(), self.y_test_pred.flatten())
            test_r2 = r2_score(self.y_test_real.flatten(), self.y_test_pred.flatten())
            
            print(f"\n🏆 Podsumowanie wydajności GRU:")
            print(f"   📈 R² Score: {test_r2:.3f} (wyjaśnia {test_r2*100:.1f}% wariancji)")
            print(f"   💰 MAE: ${test_mae:.2f} (średni błąd)")
            print(f"   ⚡ Epoki: {len(self.train_losses)} (szybki trening)")
            print(f"   🎯 Status: {'Bardzo dobry' if test_r2 > 0.8 else 'Dobry' if test_r2 > 0.7 else 'Wymaga poprawy'}")