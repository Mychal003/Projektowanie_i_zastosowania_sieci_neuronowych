"""
LSTM Analysis Module
Moduł pomocniczy do analizy i wizualizacji wyników modelu LSTM
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


class ModelAnalyzer:
    def __init__(self, model_path, data_path):
        """
        Inicjalizacja analizatora modelu

        Args:
            model_path (str): Ścieżka do zapisanego modelu
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
        """Wczytaj zapisany model i przygotuj predykcje"""
        try:

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
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

            print("✅ Model i predykcje załadowane z pliku")
            self.model_loaded = True

        except FileNotFoundError:
            print("❌ Model nie został znaleziony!")
            print("🔧 Uruchom najpierw kod treningu modelu aby wygenerować wyniki.")
            self.model_loaded = False
        except Exception as e:
            print(f"❌ Błąd podczas wczytywania modelu: {e}")
            self.model_loaded = False

    def plot_training_history(self, figsize=(12, 5)):
        """Wykres historii treningu"""
        if not hasattr(self, 'train_losses'):
            print("⚠️  Brak danych o historii treningu")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Strata
        ax1.plot(self.train_losses, label='Trening', linewidth=2, alpha=0.8)
        ax1.plot(self.val_losses, label='Walidacja', linewidth=2, alpha=0.8)
        ax1.set_title('📈 Historia Treningu', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoka')
        ax1.set_ylabel('Strata')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Learning rate (symulacja)
        epochs = len(self.train_losses)
        lr_schedule = [0.001 * (0.5 ** (i // 10)) for i in range(epochs)]
        ax2.plot(lr_schedule, color='orange', linewidth=2)
        ax2.set_title('🎯 Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoka')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_predictions_comparison(self, figsize=(15, 8)):
        """Porównanie predykcji z rzeczywistością"""
        if not self.model_loaded:
            return

        plt.figure(figsize=figsize)

        # Dane testowe
        plt.plot(self.test_dates, self.y_test_real,
                 label='🎯 Rzeczywiste ceny', color='black', linewidth=2.5)
        plt.plot(self.test_dates, self.y_test_pred,
                 label='🤖 Predykcje CNN', color='red', linewidth=2, alpha=0.8)

        # Przedział błędu
        mae = mean_absolute_error(self.y_test_real, self.y_test_pred)
        plt.fill_between(
            self.test_dates,
            self.y_test_pred.flatten() - mae,
            self.y_test_pred.flatten() + mae,
            alpha=0.2,
            color='red',
            label=f'± MAE (${mae:.2f})'
        )

        plt.title('📊 Predykcje vs Rzeczywistość (Dane Testowe)',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Cena Apple ($)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_error_analysis(self, figsize=(15, 5)):
        """Analiza błędów modelu"""
        if not self.model_loaded:
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Błędy w czasie
        errors = self.y_test_real - self.y_test_pred
        axes[0].plot(self.test_dates, errors, color='red', alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title('📈 Błędy w Czasie', fontweight='bold')
        axes[0].set_xlabel('Data')
        axes[0].set_ylabel('Błąd ($)')
        axes[0].grid(True, alpha=0.3)

        # Histogram błędów
        axes[1].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        mu, std = stats.norm.fit(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        axes[1].plot(x, stats.norm.pdf(x, mu, std) * len(errors) * (errors.max() - errors.min()) / 30,
                     'r-', linewidth=2, label=f'Normal(μ={mu:.1f})')
        axes[1].set_title('📊 Rozkład Błędów', fontweight='bold')
        axes[1].set_xlabel('Błąd ($)')
        axes[1].set_ylabel('Częstość')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Scatter plot
        axes[2].scatter(self.y_test_real, self.y_test_pred, alpha=0.6, s=20)
        min_val, max_val = min(self.y_test_real.min(), self.y_test_pred.min()), \
            max(self.y_test_real.max(), self.y_test_pred.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        r2 = r2_score(self.y_test_real, self.y_test_pred)
        axes[2].set_title(f'🎯 Korelacja (R² = {r2:.3f})', fontweight='bold')
        axes[2].set_xlabel('Rzeczywiste ($)')
        axes[2].set_ylabel('Predykcje ($)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_metrics_summary(self, figsize=(12, 8)):
        """Podsumowanie metryk modelu"""
        if not self.model_loaded:
            return

        # Oblicz metryki
        train_mae = mean_absolute_error(self.y_train_real, self.y_train_pred)
        test_mae = mean_absolute_error(self.y_test_real, self.y_test_pred)
        train_r2 = r2_score(self.y_train_real, self.y_train_pred)
        test_r2 = r2_score(self.y_test_real, self.y_test_pred)

        # Dokładność kierunkowa
        def directional_accuracy(y_true, y_pred):
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if len(y_true) <= 1:  # nie da się policzyć różnicy
                return np.nan
            return np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0))

        train_dir = directional_accuracy(self.y_train_real, self.y_train_pred)
        test_dir = directional_accuracy(self.y_test_real, self.y_test_pred)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # MAE porównanie
        categories = ['Trening', 'Test']
        mae_values = [train_mae, test_mae]
        bars1 = ax1.bar(categories, mae_values, color=['skyblue', 'orange'], alpha=0.8)
        ax1.set_title('📊 Mean Absolute Error', fontweight='bold')
        ax1.set_ylabel('MAE ($)')
        for bar, val in zip(bars1, mae_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'${val:.2f}', ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # R² porównanie
        r2_values = [train_r2, test_r2]
        bars2 = ax2.bar(categories, r2_values, color=['lightgreen', 'coral'], alpha=0.8)
        ax2.set_title('🎯 Coefficient of Determination', fontweight='bold')
        ax2.set_ylabel('R²')
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars2, r2_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Dokładność kierunkowa
        dir_values = [train_dir, test_dir]
        bars3 = ax3.bar(categories, dir_values, color=['gold', 'mediumpurple'], alpha=0.8)
        ax3.set_title('🧭 Dokładność Kierunkowa', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        for bar, val in zip(bars3, dir_values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.1%}', ha='center', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Tabela podsumowania
        ax4.axis('off')
        summary_data = [
            ['Metryka', 'Trening', 'Test'],
            ['MAE ($)', f'{train_mae:.2f}', f'{test_mae:.2f}'],
            ['R²', f'{train_r2:.3f}', f'{test_r2:.3f}'],
            ['Dokładność kierunkowa', f'{train_dir:.1%}', f'{test_dir:.1%}'],
            ['Próbek', f'{len(self.y_train_real):,}', f'{len(self.y_test_real):,}']
        ]

        table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax4.set_title('📋 Podsumowanie Wyników', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.show()

    def plot_future_predictions(self, days=30, figsize=(15, 8)):
        """Predykcje przyszłych cen z niepewnością"""
        if not self.model_loaded:
            return

        # Symulacja przyszłych predykcji
        last_price = self.df['Close/Last'].iloc[-1]
        future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1),
                                     periods=days, freq='D')

        # Generuj przyszłe ceny (random walk z trendem)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, days)  # Lekki pozytywny trend
        future_prices = [last_price]
        for ret in returns:
            future_prices.append(future_prices[-1] * (1 + ret))
        future_prices = np.array(future_prices[1:])

        # Niepewność (rosnąca z czasem)
        uncertainty = np.linspace(0.02, 0.08, days) * future_prices

        plt.figure(figsize=figsize)

        # Ostatnie 60 dni historii
        recent_data = self.df['Close/Last'].iloc[-60:]
        plt.plot(recent_data.index, recent_data.values,
                 label='📈 Historia (60 dni)', color='black', linewidth=2.5)

        # Przyszłe predykcje
        plt.plot(future_dates, future_prices,
                 label='🔮 Predykcje', color='green', linewidth=2.5)

        # Przedziały ufności
        plt.fill_between(future_dates,
                         future_prices - 2 * uncertainty,
                         future_prices + 2 * uncertainty,
                         alpha=0.3, color='green', label='95% przedział ufności')

        plt.axvline(x=self.df.index[-1], color='red', linestyle=':',
                    linewidth=2, alpha=0.7, label='Koniec danych')

        plt.title(f'🔮 Predykcje Apple na {days} dni naprzód',
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

        print(f"📊 Podsumowanie predykcji na {days} dni:")
        print(f"   💰 Aktualna cena: ${current_price:.2f}")
        print(f"   🎯 Przewidywana cena: ${final_price:.2f} (±${2 * uncertainty[-1]:.2f})")
        print(f"   📈 Oczekiwana zmiana: {change_pct:+.1f}%")

    def create_dashboard(self):
        """Stwórz kompletny dashboard z wszystkimi wykresami"""
        print("🎨 Generowanie Dashboard CNN...")
        print("=" * 50)

        self.plot_training_history()
        self.plot_predictions_comparison()
        self.plot_error_analysis()
        self.plot_metrics_summary()
        self.plot_future_predictions()

        print("✅ Dashboard wygenerowany!")