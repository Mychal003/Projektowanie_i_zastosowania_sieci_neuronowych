# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import preprocess
import numpy as np
import warnings


def plot_summary(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Date', y='Close/Last')

    step = max(1, len(df) // 7)
    x_ticks = df['Date'].iloc[::step]
    x_labels = df['Date'].dt.strftime('%B').iloc[::step]

    y_min, y_max = df['Close/Last'].min(), df['Close/Last'].max()
    y_ticks = [round(y_min + i * (y_max - y_min) / 5, 2) for i in range(6)]

    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=30)
    plt.yticks(y_ticks)

    plt.xlabel('Miesiąc roku 2024')
    plt.ylabel('Cena zamknięcia [USD]')
    plt.title('Ceny akcji spółki')
    plt.tight_layout()
    plt.show()




def plot_price_interpolation(df, order=5):
    """
    Wykres zmienności cen akcji jako interpolacja wielomianowa.

    Parameters:
    df (pd.DataFrame): Dane giełdowe zawierające kolumny 'Date' i 'Close/Last'.
    order (int): Stopień wielomianu do dopasowania (domyślnie 5).
    """
    warnings.filterwarnings("ignore", category=np.RankWarning)  # ignoruj ostrzeżenia Polyfit

    # Przekształcenie daty na wartość numeryczną
    df['DateNumeric'] = (df['Date'] - df['Date'].min()).dt.days

    # Wykres regresji
    axes = sns.lmplot(data=df, x='DateNumeric', y='Close/Last', order=order, ci=None, markers=".")

    # Ustawienia osi Y
    y_min = df['Close/Last'].min()
    y_max = df['Close/Last'].max()
    y_ticks = [round(y_min + i * (y_max - y_min) / 5, 2) for i in range(6)]

    # jawnie ustawiamy ticki i etykiety
    for ax in axes.axes.flatten():
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks])
        ax.set_xlabel('Dzień roku 2024')
        ax.set_ylabel('Cena zamknięcia [USD]')

    plt.suptitle('Wykres zmienności cen akcji – interpolacja', y=1.02)
    plt.tight_layout()
    plt.show()





def plot_price_distribution(df):
    """
    Wyświetla histogram z gęstością (KDE) dla rozkładu cen zamknięcia akcji.

    Parameters:
    df (pd.DataFrame): Dane zawierające kolumnę 'Close/Last'.
    """
    sns.histplot(data=df['Close/Last'], kde=True).set_xlabel('Cena zamknięcia [USD]')
    plt.ylabel('Ilość')
    plt.title('Rozkład cen akcji spółki')
    plt.tight_layout()
    plt.show()




def show_basic_statistics(df):
    print("Statystyki opisowe:\n", df.describe())
    print("\nBraki danych:\n", df.isnull().sum())

def plot_close_distribution(df):
    sns.histplot(df['Close/Last'], kde=True)
    plt.xlabel('Cena zamknięcia [USD]')
    plt.ylabel('Ilość')
    plt.title('Rozkład cen zamknięcia')
    plt.show()

def plot_trend_line(df):
    sns.lineplot(data=df, x='Date', y='Close/Last')
    plt.title('Trend cen akcji w czasie')
    plt.xlabel('Data')
    plt.ylabel('Cena zamknięcia [USD]')
    plt.xticks(rotation=45)
    plt.show()

def plot_daily_returns(df):
    df['Return'] = df['Close/Last'].pct_change()
    sns.histplot(df['Return'].dropna(), bins=50, kde=True)
    plt.title('Rozkład dziennej stopy zwrotu')
    plt.xlabel('Stopa zwrotu')
    plt.ylabel('Ilość')
    plt.show()

def plot_monthly_boxplot(df):
    df['Month'] = df['Date'].dt.month_name()
    sns.boxplot(data=df, x='Month', y='Close/Last', order=df['Month'].unique())
    plt.xticks(rotation=45)
    plt.title('Rozkład cen według miesiąca')
    plt.xlabel('Miesiąc')
    plt.ylabel('Cena zamknięcia [USD]')
    plt.show()

def plot_correlation_matrix(df):
    numerical = df.select_dtypes(include='number')
    sns.heatmap(numerical.corr(), annot=True, cmap='coolwarm')
    plt.title('Macierz korelacji')
    plt.show()


