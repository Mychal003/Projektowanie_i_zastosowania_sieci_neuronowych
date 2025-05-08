import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from typing import Optional

def plot_summary(df: pd.DataFrame) -> None:
    """
    Wyświetla wykres podsumowujący ceny akcji w czasie.
    
    Parameters:
    df (pd.DataFrame): Dane giełdowe zawierające kolumny 'Date' i 'Close/Last'.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Date', y='Close/Last')

    # Ustawienie etykiet na osi X
    step = max(1, len(df) // 7)
    x_ticks = df['Date'].iloc[::step]
    x_labels = df['Date'].dt.strftime('%B').iloc[::step]

    # Ustawienie etykiet na osi Y
    y_min, y_max = df['Close/Last'].min(), df['Close/Last'].max()
    y_ticks = [round(y_min + i * (y_max - y_min) / 5, 2) for i in range(6)]

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=30)
    plt.yticks(y_ticks)

    plt.xlabel('Miesiąc roku 2024')
    plt.ylabel('Cena zamknięcia [USD]')
    plt.title('Ceny akcji spółki')
    plt.tight_layout()
    plt.show()


def plot_price_interpolation(df: pd.DataFrame, order: int = 5) -> None:
    """
    Wykres zmienności cen akcji jako interpolacja wielomianowa.

    Parameters:
    df (pd.DataFrame): Dane giełdowe zawierające kolumny 'Date' i 'Close/Last'.
    order (int): Stopień wielomianu do dopasowania (domyślnie 5).
    """
    warnings.filterwarnings("ignore", category=np.RankWarning)  # ignoruj ostrzeżenia Polyfit

    # Przekształcenie daty na wartość numeryczną
    df = df.copy()  # Tworzymy kopię DataFrame, by nie modyfikować oryginału
    df['DateNumeric'] = (df['Date'] - df['Date'].min()).dt.days

    # Wykres regresji
    axes = sns.lmplot(data=df, x='DateNumeric', y='Close/Last', order=order, ci=None, markers=".")

    # Ustawienia osi Y
    y_min = df['Close/Last'].min()
    y_max = df['Close/Last'].max()
    y_ticks = [round(y_min + i * (y_max - y_min) / 5, 2) for i in range(6)]

    # Jawnie ustawiamy ticki i etykiety
    for ax in axes.axes.flatten():
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks])
        ax.set_xlabel('Dzień roku 2024')
        ax.set_ylabel('Cena zamknięcia [USD]')

    plt.suptitle('Wykres zmienności cen akcji – interpolacja', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_price_distribution(df: pd.DataFrame) -> None:
    """
    Wyświetla histogram z gęstością (KDE) dla rozkładu cen zamknięcia akcji.

    Parameters:
    df (pd.DataFrame): Dane zawierające kolumnę 'Close/Last'.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df['Close/Last'], kde=True)
    plt.xlabel('Cena zamknięcia [USD]')
    plt.ylabel('Ilość')
    plt.title('Rozkład cen akcji spółki')
    plt.tight_layout()
    plt.show()


def show_basic_statistics(df: pd.DataFrame) -> None:
    """
    Wyświetla podstawowe statystyki opisowe danych.
    
    Parameters:
    df (pd.DataFrame): Dane giełdowe.
    """
    print("Statystyki opisowe:\n", df.describe())
    print("\nBraki danych:\n", df.isnull().sum())


def plot_close_distribution(df: pd.DataFrame) -> None:
    """
    Wyświetla histogram rozkładu cen zamknięcia.
    
    Parameters:
    df (pd.DataFrame): Dane zawierające kolumnę 'Close/Last'.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Close/Last'], kde=True)
    plt.xlabel('Cena zamknięcia [USD]')
    plt.ylabel('Ilość')
    plt.title('Rozkład cen zamknięcia')
    plt.tight_layout()
    plt.show()


def plot_trend_line(df: pd.DataFrame) -> None:
    """
    Wyświetla linię trendu cen akcji w czasie.
    
    Parameters:
    df (pd.DataFrame): Dane zawierające kolumny 'Date' i 'Close/Last'.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Close/Last')
    plt.title('Trend cen akcji w czasie')
    plt.xlabel('Data')
    plt.ylabel('Cena zamknięcia [USD]')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_daily_returns(df: pd.DataFrame) -> None:
    """
    Wyświetla histogram dziennych stóp zwrotu.
    
    Parameters:
    df (pd.DataFrame): Dane zawierające kolumnę 'Close/Last'.
    """
    plt.figure(figsize=(10, 6))
    df_copy = df.copy()  # Tworzymy kopię DataFrame, by nie modyfikować oryginału
    df_copy['Return'] = df_copy['Close/Last'].pct_change()
    sns.histplot(df_copy['Return'].dropna(), bins=50, kde=True)
    plt.title('Rozkład dziennej stopy zwrotu')
    plt.xlabel('Stopa zwrotu')
    plt.ylabel('Ilość')
    plt.tight_layout()
    plt.show()


def plot_monthly_boxplot(df: pd.DataFrame) -> None:
    """
    Wyświetla wykres pudełkowy cen według miesięcy.
    
    Parameters:
    df (pd.DataFrame): Dane zawierające kolumny 'Date' i 'Close/Last'.
    """
    plt.figure(figsize=(12, 6))
    df_copy = df.copy()  # Tworzymy kopię DataFrame, by nie modyfikować oryginału
    df_copy['Month'] = df_copy['Date'].dt.month_name()
    
    # Sortowanie miesięcy w porządku chronologicznym
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Filtrowanie tylko tych miesięcy, które występują w danych
    available_months = [m for m in month_order if m in df_copy['Month'].unique()]
    
    sns.boxplot(data=df_copy, x='Month', y='Close/Last', order=available_months)
    plt.xticks(rotation=45)
    plt.title('Rozkład cen według miesiąca')
    plt.xlabel('Miesiąc')
    plt.ylabel('Cena zamknięcia [USD]')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Wyświetla macierz korelacji dla zmiennych numerycznych.
    
    Parameters:
    df (pd.DataFrame): Dane giełdowe.
    """
    plt.figure(figsize=(10, 8))
    numerical = df.select_dtypes(include='number')
    sns.heatmap(numerical.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Macierz korelacji')
    plt.tight_layout()
    plt.show()


def plot_volume_analysis(df: pd.DataFrame) -> None:
    """
    Wyświetla analizę wolumenu transakcji.
    
    Parameters:
    df (pd.DataFrame): Dane zawierające kolumny 'Date' i 'Volume'.
    """
    if 'Volume' not in df.columns:
        print("Kolumna 'Volume' nie istnieje w danych.")
        return
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Volume')
    plt.title('Wolumen transakcji w czasie')
    plt.xlabel('Data')
    plt.ylabel('Wolumen')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_price_vs_volume(df: pd.DataFrame) -> None:
    """
    Wyświetla wykres rozrzutu ceny vs wolumen.
    
    Parameters:
    df (pd.DataFrame): Dane zawierające kolumny 'Close/Last' i 'Volume'.
    """
    if 'Volume' not in df.columns:
        print("Kolumna 'Volume' nie istnieje w danych.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Close/Last', y='Volume')
    plt.title('Cena vs Wolumen')
    plt.xlabel('Cena zamknięcia [USD]')
    plt.ylabel('Wolumen')
    plt.tight_layout()
    plt.show()


def plot_moving_averages(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> None:
    """
    Wyświetla średnie ruchome cen akcji.
    
    Parameters:
    df (pd.DataFrame): Dane zawierające kolumny 'Date' i 'Close/Last'.
    short_window (int): Okno dla krótkiej średniej ruchomej (domyślnie 20 dni).
    long_window (int): Okno dla długiej średniej ruchomej (domyślnie 50 dni).
    """
    df_copy = df.copy()  # Tworzymy kopię DataFrame, by nie modyfikować oryginału
    
    # Obliczenie średnich ruchomych
    df_copy[f'SMA_{short_window}'] = df_copy['Close/Last'].rolling(window=short_window).mean()
    df_copy[f'SMA_{long_window}'] = df_copy['Close/Last'].rolling(window=long_window).mean()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_copy, x='Date', y='Close/Last', label='Cena zamknięcia')
    sns.lineplot(data=df_copy, x='Date', y=f'SMA_{short_window}', label=f'SMA {short_window} dni')
    sns.lineplot(data=df_copy, x='Date', y=f'SMA_{long_window}', label=f'SMA {long_window} dni')
    
    plt.title('Ceny akcji i średnie ruchome')
    plt.xlabel('Data')
    plt.ylabel('Cena [USD]')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def analyze_data(df: pd.DataFrame, show_all: bool = False) -> None:
    """
    Przeprowadza pełną analizę eksploracyjną danych.
    
    Parameters:
    df (pd.DataFrame): Dane giełdowe.
    show_all (bool): Czy wyświetlić wszystkie możliwe wykresy (domyślnie False).
    """
    print("=== Analiza danych giełdowych ===")
    
    # Podstawowe statystyki
    show_basic_statistics(df)
    
    # Podstawowe wykresy
    plot_trend_line(df)
    plot_price_distribution(df)
    plot_daily_returns(df)
    
    # Dodatkowe wykresy jeśli show_all=True
    if show_all:
        plot_monthly_boxplot(df)
        plot_correlation_matrix(df)
        plot_price_interpolation(df)
        
        # Jeśli istnieje kolumna Volume
        if 'Volume' in df.columns:
            plot_volume_analysis(df)
            plot_price_vs_volume(df)
        
        # Średnie ruchome
        plot_moving_averages(df)
    
    print("Analiza zakończona.")