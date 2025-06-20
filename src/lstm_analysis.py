"""
LSTM Analysis Module - Zaktualizowany z prawdziwymi predykcjami
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

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# DEFINICJA MODELU (skopiowana z trainera)
class ImprovedStockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=1, dropout=0.3):
        super(ImprovedStockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        
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
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.relu(self.bn1(self.fc1(context)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.relu(self.bn3(self.fc3(out)))
        out = self.fc4(out)
        
        return out

def calculate_rsi(prices, period=14):
    """Oblicz RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
        """Wczytaj i przygotuj dane (identycznie jak w trainerze)"""
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.dropna().sort_index()
        
        # Feature engineering (identyczne jak w trainerze)
        df['Returns'] = df['Close/Last'].pct_change()
        df['MA_5'] = df['Close/Last'].rolling(window=5).mean()
        df['MA_20'] = df['Close/Last'].rolling(window=20).mean()
        df['MA_50'] = df['Close/Last'].rolling(window=50).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['RSI'] = calculate_rsi(df['Close/Last'], 14)
        df = df.dropna()
        
        self.df = df
        
    def load_model(self):
        """Wczytaj zapisany model i przygotuj do predykcji"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Wczytaj parametry
            self.price_scaler = checkpoint['price_scaler']
            self.scaler = checkpoint['scaler']
            self.seq_length = checkpoint['seq_length']
            self.feature_columns = checkpoint['feature_columns']
            self.metrics = checkpoint['metrics']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            # Odtwórz model
            config = checkpoint['model_config']
            self.model = ImprovedStockLSTM(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=config['output_size'],
                dropout=config['dropout']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Wczytaj historyczne predykcje
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
        """Wykres historii treningu (prawdziwe dane)"""
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
        
        # Mejlestones treningu
        epochs = len(self.train_losses)
        final_train_loss = self.train_losses[-1]
        final_val_loss = self.val_losses[-1]
        min_val_loss = min(self.val_losses)
        
        ax2.bar(['Końcowa\nStrata Treningu', 'Końcowa\nStrata Walidacji', 'Najlepsza\nStrata Walidacji'], 
                [final_train_loss, final_val_loss, min_val_loss], 
                color=['skyblue', 'orange', 'green'], alpha=0.8)
        ax2.set_title('🎯 Podsumowanie Treningu', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Strata')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_predictions_comparison(self, figsize=(15, 8)):
        """Porównanie predykcji z rzeczywistością"""
        if not self.model_loaded:
            return
            
        plt.figure(figsize=figsize)
        
        # Spłaszczenie tablic - ważne!
        y_test_real_flat = self.y_test_real.flatten()
        y_test_pred_flat = self.y_test_pred.flatten()
        
        # Dane testowe
        plt.plot(self.test_dates, y_test_real_flat, 
                label='🎯 Rzeczywiste ceny', color='black', linewidth=2.5)
        plt.plot(self.test_dates, y_test_pred_flat, 
                label='🤖 Predykcje LSTM', color='red', linewidth=2, alpha=0.8)
        
        # Przedział błędu
        mae = mean_absolute_error(y_test_real_flat, y_test_pred_flat)
        plt.fill_between(self.test_dates, 
                        y_test_pred_flat - mae, 
                        y_test_pred_flat + mae, 
                        alpha=0.2, color='red', label=f'± MAE (${mae:.2f})')
        
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
        
        # Spłaszczenie tablic
        y_test_real_flat = self.y_test_real.flatten()
        y_test_pred_flat = self.y_test_pred.flatten()
        
        # Błędy w czasie
        errors = y_test_real_flat - y_test_pred_flat
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
        axes[1].plot(x, stats.norm.pdf(x, mu, std) * len(errors) * (errors.max()-errors.min())/30, 
                    'r-', linewidth=2, label=f'Normal(μ={mu:.1f})')
        axes[1].set_title('📊 Rozkład Błędów', fontweight='bold')
        axes[1].set_xlabel('Błąd ($)')
        axes[1].set_ylabel('Częstość')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[2].scatter(y_test_real_flat, y_test_pred_flat, alpha=0.6, s=20)
        min_val, max_val = min(y_test_real_flat.min(), y_test_pred_flat.min()), \
                          max(y_test_real_flat.max(), y_test_pred_flat.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        r2 = r2_score(y_test_real_flat, y_test_pred_flat)
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
            
        # Spłaszczenie tablic
        y_train_real_flat = self.y_train_real.flatten()
        y_train_pred_flat = self.y_train_pred.flatten()
        y_test_real_flat = self.y_test_real.flatten()
        y_test_pred_flat = self.y_test_pred.flatten()
        
        # Oblicz metryki
        train_mae = mean_absolute_error(y_train_real_flat, y_train_pred_flat)
        test_mae = mean_absolute_error(y_test_real_flat, y_test_pred_flat)
        train_r2 = r2_score(y_train_real_flat, y_train_pred_flat)
        test_r2 = r2_score(y_test_real_flat, y_test_pred_flat)
        
        # Dokładność kierunkowa
        def directional_accuracy(y_true, y_pred):
            return np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0))
        
        train_dir = directional_accuracy(y_train_real_flat, y_train_pred_flat)
        test_dir = directional_accuracy(y_test_real_flat, y_test_pred_flat)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # MAE porównanie
        categories = ['Trening', 'Test']
        mae_values = [train_mae, test_mae]
        bars1 = ax1.bar(categories, mae_values, color=['skyblue', 'orange'], alpha=0.8)
        ax1.set_title('📊 Mean Absolute Error', fontweight='bold')
        ax1.set_ylabel('MAE ($)')
        for bar, val in zip(bars1, mae_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'${val:.2f}', ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # R² porównanie
        r2_values = [train_r2, test_r2]
        bars2 = ax2.bar(categories, r2_values, color=['lightgreen', 'coral'], alpha=0.8)
        ax2.set_title('🎯 Coefficient of Determination', fontweight='bold')
        ax2.set_ylabel('R²')
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars2, r2_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Dokładność kierunkowa
        dir_values = [train_dir, test_dir]
        bars3 = ax3.bar(categories, dir_values, color=['gold', 'mediumpurple'], alpha=0.8)
        ax3.set_title('🧭 Dokładność Kierunkowa', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        for bar, val in zip(bars3, dir_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.1%}', ha='center', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Tabela podsumowania
        ax4.axis('off')
        summary_data = [
            ['Metryka', 'Trening', 'Test'],
            ['MAE ($)', f'{train_mae:.2f}', f'{test_mae:.2f}'],
            ['R²', f'{train_r2:.3f}', f'{test_r2:.3f}'],
            ['Dokładność kierunkowa', f'{train_dir:.1%}', f'{test_dir:.1%}'],
            ['Próbek', f'{len(y_train_real_flat):,}', f'{len(y_test_real_flat):,}']
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
        """PRAWDZIWE predykcje przyszłych cen z modelu LSTM"""
        if not self.model_loaded:
            return
        
        print(f"🔮 Generowanie prawdziwych predykcji na {days} dni...")
        
        # Przygotuj dane wejściowe (ostatnie seq_length próbek)
        data_for_features = self.df[self.feature_columns].values
        scaled_data = self.scaler.transform(data_for_features)
        last_sequence = scaled_data[-self.seq_length:]
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        self.model.eval()
        with torch.no_grad():
            for day in range(days):
                # Konwersja do tensora
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Predykcja z modelu
                prediction_scaled = self.model(input_tensor).cpu().numpy()
                pred_price = self.price_scaler.inverse_transform(prediction_scaled)[0][0]
                future_predictions.append(pred_price)
                
                # Aktualizuj sekwencję dla następnej predykcji
                # Oblicz nowe cechy na podstawie predykcji
                last_price = pred_price
                if day == 0:
                    prev_price = self.df['Close/Last'].iloc[-1]
                else:
                    prev_price = future_predictions[-2]
                
                # Nowe cechy (uproszczone)
                new_return = (last_price - prev_price) / prev_price if prev_price > 0 else 0
                new_ma5 = last_price  # Uproszczenie
                new_ma20 = last_price  # Uproszczenie
                new_volatility = 0.02  # Założona volatility
                
                new_features = np.array([[last_price, new_return, new_ma5, new_ma20, new_volatility]])
                new_features_scaled = self.scaler.transform(new_features)
                
                # Przesuń okno czasowe
                current_sequence = np.vstack([current_sequence[1:], new_features_scaled])
        
        # Oblicz niepewność na podstawie błędów testowych
        test_errors = np.abs(self.y_test_real.flatten() - self.y_test_pred.flatten())
        base_uncertainty = np.std(test_errors)
        
        # Niepewność rosnąca z czasem
        uncertainty = np.array([base_uncertainty * np.sqrt(1 + i*0.1) for i in range(days)])
        
        # Generuj daty
        future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), 
                                   periods=days, freq='D')
        
        # Rysowanie wykresu
        plt.figure(figsize=figsize)
        
        # Ostatnie 60 dni historii
        recent_data = self.df['Close/Last'].iloc[-60:]
        plt.plot(recent_data.index, recent_data.values, 
                label='📈 Historia (60 dni)', color='black', linewidth=2.5)
        
        # Prawdziwe predykcje z modelu LSTM
        plt.plot(future_dates, future_predictions, 
                label='🤖 Predykcje LSTM', color='green', linewidth=2.5)
        
        # Przedziały ufności
        plt.fill_between(future_dates, 
                        np.array(future_predictions) - 2*uncertainty, 
                        np.array(future_predictions) + 2*uncertainty, 
                        alpha=0.3, color='green', label='95% przedział ufności')
        
        plt.axvline(x=self.df.index[-1], color='red', linestyle=':', 
                   linewidth=2, alpha=0.7, label='Koniec danych')
        
        plt.title(f'🔮 Prawdziwe Predykcje LSTM Apple na {days} dni', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Cena ($)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Podsumowanie predykcji
        current_price = self.df['Close/Last'].iloc[-1]
        final_price = future_predictions[-1]
        change_pct = (final_price - current_price) / current_price * 100
        
        print(f"📊 Podsumowanie prawdziwych predykcji na {days} dni:")
        print(f"   💰 Aktualna cena: ${current_price:.2f}")
        print(f"   🎯 Przewidywana cena: ${final_price:.2f} (±${2*uncertainty[-1]:.2f})")
        print(f"   📈 Oczekiwana zmiana: {change_pct:+.1f}%")
        print(f"   🤖 Predykcje wygenerowane przez wytrenowany model LSTM")

    def create_dashboard(self):
        """Stwórz kompletny dashboard z prawdziwymi wynikami"""
        print("🎨 Generowanie Dashboard LSTM z prawdziwymi danymi...")
        print("=" * 60)
        
        self.plot_training_history()
        self.plot_predictions_comparison()
        self.plot_error_analysis()
        self.plot_metrics_summary()
        self.plot_future_predictions()
        
        print("✅ Dashboard z prawdziwymi wynikami wygenerowany!")

# Przykład użycia:
if __name__ == "__main__":
    model_path = r'C:\Users\pawli\OneDrive\Pulpit\sieci\Projektowanie_i_zastosowania_sieci_neuronowych\improved_spy_lstm_model.pth'
    data_path = r'C:\Users\pawli\OneDrive\Pulpit\sieci\Projektowanie_i_zastosowania_sieci_neuronowych\apple_stock_data.csv'
    
    analyzer = ModelAnalyzer(model_path, data_path)
    analyzer.create_dashboard()