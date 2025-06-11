"""
Exploratory Data Analysis (EDA) dla danych giełdowych
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Ustawienia wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StockEDA:
    """Klasa do przeprowadzania EDA na danych giełdowych."""
    
    def __init__(self, df, date_col='Date', price_col='Close/Last'):
        """Inicjalizacja z danymi."""
        self.df = df.copy()
        self.date_col = date_col
        self.price_col = price_col
        
        # Konwersja daty
        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df = self.df.sort_values(self.date_col).reset_index(drop=True)
        
        # Oblicz zwroty raz
        self.returns = self.df[self.price_col].pct_change().dropna()
    
    def plot_all_analysis(self):
        """Generuje wszystkie wykresy EDA w jednym wywołaniu."""
        # 1. Historia cen
        self._plot_price_history()
        
        # 2. Analiza zwrotów
        self._plot_returns_analysis()
        
        # 3. Analiza zmienności
        self._plot_volatility_analysis()
        
        # 4. Wskaźniki techniczne
        self._plot_technical_indicators()
        
        # 5. Korelacje
        self._plot_correlations()
        
        # 6. Podsumowanie statystyk
        self._print_summary_stats()
    
    def _plot_price_history(self):
        """Historia cen z wolumenem."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Wykres ceny
        ax1.plot(self.df[self.date_col], self.df[self.price_col], 
                 linewidth=2, color='darkblue')
        ax1.fill_between(self.df[self.date_col], self.df[self.price_col], 
                         alpha=0.3, color='lightblue')
        ax1.set_title('Historia cen akcji', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cena ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Dodaj średnie kroczące
        if len(self.df) > 50:
            ma50 = self.df[self.price_col].rolling(window=50).mean()
            ax1.plot(self.df[self.date_col], ma50, 
                     label='MA50', color='orange', alpha=0.8)
        if len(self.df) > 200:
            ma200 = self.df[self.price_col].rolling(window=200).mean()
            ax1.plot(self.df[self.date_col], ma200, 
                     label='MA200', color='red', alpha=0.8)
        ax1.legend()
        
        # Wolumen
        if 'Volume' in self.df.columns:
            ax2.bar(self.df[self.date_col], self.df['Volume'], 
                    alpha=0.7, color='darkgreen')
            ax2.set_ylabel('Wolumen', fontsize=12)
            ax2.set_xlabel('Data', fontsize=12)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
        
        plt.tight_layout()
        plt.show()
        
        # Podstawowe statystyki
        print("\n📊 STATYSTYKI CENOWE:")
        print(f"Cena min: ${self.df[self.price_col].min():.2f}")
        print(f"Cena max: ${self.df[self.price_col].max():.2f}")
        print(f"Cena średnia: ${self.df[self.price_col].mean():.2f}")
        print(f"Zakres dat: {self.df[self.date_col].min().date()} - {self.df[self.date_col].max().date()}")
    
    def _plot_returns_analysis(self):
        """Analiza zwrotów."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram zwrotów
        axes[0, 0].hist(self.returns, bins=50, alpha=0.7, color='darkblue', 
                        density=True, edgecolor='black')
        
        # Rozkład normalny
        mu, std = self.returns.mean(), self.returns.std()
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-', 
                        linewidth=2, label='Normal')
        axes[0, 0].set_title('Rozkład zwrotów', fontsize=14)
        axes[0, 0].set_xlabel('Zwrot dzienny')
        axes[0, 0].set_ylabel('Gęstość')
        axes[0, 0].legend()
        
        # 2. Q-Q plot
        stats.probplot(self.returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot', fontsize=14)
        
        # 3. Zwroty w czasie
        axes[1, 0].plot(self.df[self.date_col][1:], self.returns, 
                        linewidth=1, color='darkgreen', alpha=0.7)
        axes[1, 0].set_title('Zwroty dzienne', fontsize=14)
        axes[1, 0].set_xlabel('Data')
        axes[1, 0].set_ylabel('Zwrot')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Autokorelacja zwrotów
        lags = min(40, len(self.returns)//5)
        autocorr = [self.returns.autocorr(lag=i) for i in range(lags)]
        axes[1, 1].bar(range(lags), autocorr, color='purple', alpha=0.7)
        axes[1, 1].set_title('Autokorelacja zwrotów', fontsize=14)
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autokorelacja')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Linie krytyczne
        n = len(self.returns)
        axes[1, 1].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Statystyki zwrotów
        print("\n📈 STATYSTYKI ZWROTÓW:")
        print(f"Średni zwrot dzienny: {self.returns.mean():.4%}")
        print(f"Średni zwrot roczny: {self.returns.mean() * 252:.2%}")
        print(f"Zmienność dzienna: {self.returns.std():.4%}")
        print(f"Zmienność roczna: {self.returns.std() * np.sqrt(252):.2%}")
        print(f"Sharpe Ratio: {self.returns.mean() / self.returns.std() * np.sqrt(252):.2f}")
        print(f"Skośność: {self.returns.skew():.2f}")
        print(f"Kurtoza: {self.returns.kurtosis():.2f}")
    
    def _plot_volatility_analysis(self):
        """Analiza zmienności."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Rolling volatility
        vol_21 = self.returns.rolling(window=21).std() * np.sqrt(252)
        axes[0, 0].plot(self.df[self.date_col][1:], vol_21, 
                        linewidth=2, color='darkred')
        axes[0, 0].fill_between(self.df[self.date_col][1:], vol_21, 
                                alpha=0.3, color='lightcoral')
        axes[0, 0].set_title('Zmienność 21-dniowa (annualizowana)', fontsize=14)
        axes[0, 0].set_ylabel('Zmienność')
        
        # 2. Różne okna zmienności
        for window, color in zip([5, 10, 21, 63], ['blue', 'green', 'orange', 'red']):
            vol = self.returns.rolling(window=window).std() * np.sqrt(252)
            axes[0, 1].plot(self.df[self.date_col][1:], vol, 
                           label=f'{window}d', color=color, alpha=0.7)
        axes[0, 1].set_title('Zmienność dla różnych okien', fontsize=14)
        axes[0, 1].set_ylabel('Zmienność annualizowana')
        axes[0, 1].legend()
        
        # 3. High-Low spread
        if 'High' in self.df.columns and 'Low' in self.df.columns:
            hl_spread = (self.df['High'] - self.df['Low']) / self.df[self.price_col] * 100
            axes[1, 0].plot(self.df[self.date_col], hl_spread, 
                            linewidth=1, color='darkgreen', alpha=0.7)
            axes[1, 0].set_title('Spread High-Low (%)', fontsize=14)
            axes[1, 0].set_ylabel('(High-Low)/Close (%)')
        
        # 4. Rozkład zmienności
        axes[1, 1].hist(vol_21.dropna(), bins=30, alpha=0.7, 
                        color='darkblue', edgecolor='black')
        axes[1, 1].set_title('Rozkład zmienności 21-dniowej', fontsize=14)
        axes[1, 1].set_xlabel('Zmienność annualizowana')
        axes[1, 1].set_ylabel('Częstość')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_technical_indicators(self):
        """Wskaźniki techniczne."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Bollinger Bands
        ma20 = self.df[self.price_col].rolling(window=20).mean()
        std20 = self.df[self.price_col].rolling(window=20).std()
        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20
        
        axes[0].plot(self.df[self.date_col], self.df[self.price_col], 
                     label='Cena', linewidth=2, color='black')
        axes[0].plot(self.df[self.date_col], ma20, 
                     label='MA20', color='blue', alpha=0.8)
        axes[0].fill_between(self.df[self.date_col], upper_band, lower_band, 
                             alpha=0.2, color='gray', label='Bollinger Bands')
        axes[0].set_title('Cena z Bollinger Bands', fontsize=14)
        axes[0].set_ylabel('Cena ($)')
        axes[0].legend()
        
        # 2. RSI
        delta = self.df[self.price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        axes[1].plot(self.df[self.date_col], rsi, linewidth=2, color='purple')
        axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axes[1].fill_between(self.df[self.date_col], 30, 70, alpha=0.1, color='gray')
        axes[1].set_title('RSI (14)', fontsize=14)
        axes[1].set_ylabel('RSI')
        axes[1].set_ylim(0, 100)
        
        # 3. MACD
        exp12 = self.df[self.price_col].ewm(span=12, adjust=False).mean()
        exp26 = self.df[self.price_col].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        axes[2].plot(self.df[self.date_col], macd, label='MACD', color='blue')
        axes[2].plot(self.df[self.date_col], signal, label='Signal', color='red')
        axes[2].bar(self.df[self.date_col], histogram, label='Histogram', 
                    alpha=0.3, color='gray')
        axes[2].set_title('MACD', fontsize=14)
        axes[2].set_xlabel('Data')
        axes[2].legend()
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_correlations(self):
        """Analiza korelacji."""
        # Wybierz kolumny numeryczne
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. Macierz korelacji
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, linewidths=1, cbar_kws={"shrink": .8},
                        ax=axes[0])
            axes[0].set_title('Macierz korelacji', fontsize=14)
            
            # 2. Scatter: Cena vs Wolumen
            if 'Volume' in self.df.columns:
                axes[1].scatter(self.df['Volume'], self.df[self.price_col], 
                               alpha=0.5, s=20, color='darkblue')
                axes[1].set_xlabel('Wolumen')
                axes[1].set_ylabel('Cena')
                axes[1].set_title('Cena vs Wolumen', fontsize=14)
                
                # Linia regresji
                z = np.polyfit(self.df['Volume'].fillna(0), self.df[self.price_col], 1)
                p = np.poly1d(z)
                axes[1].plot(self.df['Volume'], p(self.df['Volume']), 
                            "r--", alpha=0.8, linewidth=2)
            
            plt.tight_layout()
            plt.show()
    
    def _print_summary_stats(self):
        """Podsumowanie statystyk."""
        print("\n" + "="*60)
        print("📊 PODSUMOWANIE ANALIZY EDA")
        print("="*60)
        
        print(f"\n🗓️ DANE:")
        print(f"Liczba obserwacji: {len(self.df)}")
        print(f"Zakres dat: {self.df[self.date_col].min().date()} - {self.df[self.date_col].max().date()}")
        print(f"Liczba dni: {(self.df[self.date_col].max() - self.df[self.date_col].min()).days}")
        
        print(f"\n💰 CENY:")
        print(f"Min: ${self.df[self.price_col].min():.2f}")
        print(f"Max: ${self.df[self.price_col].max():.2f}")
        print(f"Średnia: ${self.df[self.price_col].mean():.2f}")
        print(f"Obecna: ${self.df[self.price_col].iloc[-1]:.2f}")
        
        print(f"\n📈 ZWROTY:")
        print(f"Średni dzienny: {self.returns.mean():.4%}")
        print(f"Średni roczny: {self.returns.mean() * 252:.2%}")
        print(f"Zmienność roczna: {self.returns.std() * np.sqrt(252):.2%}")
        print(f"Sharpe Ratio: {self.returns.mean() / self.returns.std() * np.sqrt(252):.2f}")
        
        print(f"\n🎯 EKSTREMALNE WARTOŚCI:")
        print(f"Największy wzrost: {self.returns.max():.2%} ({self.df[self.date_col][self.returns.argmax()+1].date()})")
        print(f"Największy spadek: {self.returns.min():.2%} ({self.df[self.date_col][self.returns.argmin()+1].date()})")
        
        print("\n" + "="*60)