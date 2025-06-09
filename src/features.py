# features.py
import pandas as pd
import numpy as np

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje zaawansowane cechy techniczne do ramki danych.
    """
    df = df.copy()

    df['Return'] = df['Close/Last'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=20).std()

    df['SMA_10'] = df['Close/Last'].rolling(window=10).mean()
    df['SMA_50'] = df['Close/Last'].rolling(window=50).mean()

    delta = df['Close/Last'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    ma20 = df['Close/Last'].rolling(window=20).mean()
    std20 = df['Close/Last'].rolling(window=20).std()
    df['BB_position'] = (df['Close/Last'] - ma20) / (2 * std20)

    df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    df = df.dropna().reset_index(drop=True)
    return df
