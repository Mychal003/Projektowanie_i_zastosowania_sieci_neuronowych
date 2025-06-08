# Poprawiony kod do przygotowania danych dla LSTM

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 1. Poprawiona funkcja analizy
def analyze_data_quality(df, ticker):
    """Sprawdź jakość danych giełdowych"""
    print(f"\n=== Analiza {ticker} ===")
    
    # Podstawowe info
    print(f"Okres: {df.index[0].strftime('%Y-%m-%d')} - {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Liczba dni: {len(df)}")
    
    # Braki
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"⚠️ Braki danych: {missing}")
    else:
        print("✓ Brak braków danych")
    
    # Sprawdź dni handlowe
    days_diff = (df.index[-1] - df.index[0]).days
    expected_trading_days = days_diff * 252 / 365
    actual_percentage = len(df) / expected_trading_days * 100
    print(f"Pokrycie dni handlowych: {actual_percentage:.1f}%")
    
    # Sprawdź anomalie
    returns = df['Close'].pct_change()
    extreme_moves = (abs(returns) > 0.2).sum()
    print(f"Ekstremalne ruchy (>20%): {extreme_moves}")
    
    # Sprawdź objętość - POPRAWKA
    zero_volume = (df['Volume'] == 0).sum()
    if zero_volume > 0:
        print(f"⚠️ Dni z zerowym wolumenem: {zero_volume}")
    else:
        print("✓ Wszystkie dni mają wolumen > 0")
    
    # Dodatkowe statystyki
    print(f"\nStatystyki cen zamknięcia:")
    print(f"Min: ${df['Close'].min():.2f}")
    print(f"Max: ${df['Close'].max():.2f}")
    print(f"Średnia: ${df['Close'].mean():.2f}")
    print(f"Odchylenie std: ${df['Close'].std():.2f}")
    
    return df

# 2. Główna funkcja przygotowania danych
def prepare_best_data_for_model(ticker='SPY', start_date='2015-01-01'):
    """Przygotuj dane do modelu LSTM"""
    
    print(f"Pobieranie danych {ticker}...")
    
    # Pobierz dane
    df = yf.download(ticker, start=start_date, progress=False)
    
    if len(df) == 0:
        print(f"❌ Nie udało się pobrać danych dla {ticker}")
        return None
    
    # Analiza jakości
    analyze_data_quality(df, ticker)
    
    # Przygotuj format
    df = df.copy()
    df = df.rename(columns={
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close/Last',  # Dopasowanie do Twojego kodu
        'Volume': 'Volume'
    })
    
    # Usuń Adj Close
    if 'Adj Close' in df.columns:
        df = df.drop('Adj Close', axis=1)
    
    # Dodaj Date
    df['Date'] = df.index
    df = df.reset_index(drop=True)
    
    # Sortuj
    df = df.sort_values('Date')
    
    # Usuń NaN
    df = df.dropna()
    
    # Utwórz katalogi
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Zapisz
    output_path = 'data/processed/cleaned_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Zapisano {len(df)} dni danych do: {output_path}")
    
    # Info o danych
    print("\nStruktura danych:")
    print(df.head())
    print(f"\nKolumny: {df.columns.tolist()}")
    print(f"Typy danych:\n{df.dtypes}")
    
    return df

# 3. URUCHOM TO - przygotuj dane SPY
print("="*60)
print("PRZYGOTOWANIE DANYCH DLA MODELU LSTM")
print("="*60)

# Opcja 1: SPY (REKOMENDOWANE)
df_spy = prepare_best_data_for_model('SPY', start_date='2015-01-01')

if df_spy is not None:
    print("\n✅ SUKCES! Dane gotowe do użycia w modelu LSTM.")
    print(f"Ścieżka: data/processed/cleaned_data.csv")
    print(f"\nTeraz możesz uruchomić swój notebook LSTM!")

# 4. Opcjonalnie - pobierz też Apple dla porównania
print("\n" + "="*60)
print("OPCJONALNIE: Dane Apple")
print("="*60)

user_input = input("\nCzy pobrać też dane Apple dla porównania? (t/n): ")
if user_input.lower() == 't':
    df_aapl = prepare_best_data_for_model('AAPL', start_date='2015-01-01')
    if df_aapl is not None:
        # Zapisz jako osobny plik
        df_aapl.to_csv('data/processed/AAPL_cleaned.csv', index=False)
        print(f"✅ Dane Apple zapisane do: data/processed/AAPL_cleaned.csv")

# 5. Test - sprawdź czy plik istnieje i ma odpowiedni format
print("\n" + "="*60)
print("WERYFIKACJA PLIKU")
print("="*60)

if os.path.exists('data/processed/cleaned_data.csv'):
    test_df = pd.read_csv('data/processed/cleaned_data.csv')
    print(f"✅ Plik istnieje")
    print(f"✅ Liczba wierszy: {len(test_df)}")
    print(f"✅ Kolumny: {test_df.columns.tolist()}")
    
    # Sprawdź wymagane kolumny
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close/Last', 'Volume']
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    
    if missing_cols:
        print(f"❌ Brakujące kolumny: {missing_cols}")
    else:
        print("✅ Wszystkie wymagane kolumny obecne")
        print("\n🎉 DANE GOTOWE DO UŻYCIA W MODELU LSTM!")
else:
    print("❌ Plik nie istnieje!")

print("\n" + "="*60)
print("CO DALEJ?")
print("="*60)
print("1. Upewnij się, że plik został utworzony: data/processed/cleaned_data.csv")
print("2. Uruchom swój notebook LSTM")
print("3. Model powinien teraz działać znacznie lepiej z danymi SPY!")
print("\nDlaczego SPY jest lepszy niż pojedyncze akcje?")
print("- Reprezentuje cały rynek (500 spółek)")
print("- Mniej szumu i losowości")
print("- Bardziej przewidywalne trendy")
print("- Brak gwałtownych ruchów związanych z pojedynczą firmą")