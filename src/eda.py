# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
