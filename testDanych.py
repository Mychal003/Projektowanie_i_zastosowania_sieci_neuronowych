#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple Data Downloader - z automatyczną instalacją bibliotek
"""

import sys
import subprocess
import pkg_resources

def install_package(package):
    """Instaluje pakiet jeśli nie jest zainstalowany"""
    try:
        pkg_resources.get_distribution(package)
        print(f"✅ {package} już zainstalowany")
    except pkg_resources.DistributionNotFound:
        print(f"🔄 Instaluję {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} zainstalowany")

def check_and_install_dependencies():
    """Sprawdza i instaluje wszystkie potrzebne biblioteki"""
    required_packages = [
        'yfinance',
        'pandas', 
        'numpy',
        'scikit-learn'
    ]
    
    print("🔍 Sprawdzam zależności...")
    
    for package in required_packages:
        install_package(package)
    
    print("✅ Wszystkie zależności gotowe!\n")

# Sprawdź i zainstaluj zależności
check_and_install_dependencies()

# Teraz importuj biblioteki
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from sklearn.preprocessing import MinMaxScaler
    print("✅ Wszystkie biblioteki zaimportowane pomyślnie!\n")
except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    print("Spróbuj ponownie uruchomić skrypt lub zainstaluj ręcznie:")
    print("pip install yfinance pandas numpy scikit-learn")
    sys.exit(1)

def pobierz_dane_apple():
    """
    Pobiera dane giełdowe Apple (AAPL) z Yahoo Finance
    """
    print("🍎 Pobieranie danych Apple (AAPL)...")
    
    try:
        # Ustaw zakres dat (ostatnie 3 lata)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1095)  # 3 lata
        
        print(f"📅 Okres: {start_date.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}")
        
        # Pobierz dane Apple
        apple = yf.Ticker("AAPL")
        data = apple.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError("Nie udało się pobrać danych")
        
        # Resetuj indeks żeby Date stał się kolumną
        data = data.reset_index()
        
        # Formatuj datę zgodnie z wymaganiami
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        
        # Zmień nazwy kolumn na wymagane
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close/Last', 'Volume', 'Dividends', 'Stock Splits']
        
        # Zostaw tylko potrzebne kolumny
        data_final = data[['Date', 'Open', 'High', 'Low', 'Close/Last', 'Volume']].copy()
        
        return data_final
        
    except Exception as e:
        print(f"❌ Błąd podczas pobierania danych: {e}")
        return None

def zapisz_i_analizuj_dane():
    """
    Pobiera, zapisuje i analizuje dane Apple
    """
    # Pobierz dane
    dane_apple = pobierz_dane_apple()
    
    if dane_apple is None:
        print("❌ Nie udało się pobrać danych")
        return None
    
    # Zapisz do CSV
    filename = 'apple_stock_data.csv'
    dane_apple.to_csv(filename, index=False)
    
    print(f"✅ Pobrano {len(dane_apple)} rekordów dla Apple (AAPL)")
    print(f"💾 Dane zapisane do pliku: {filename}")
    
    # Pokaż pierwsze 10 wierszy
    print("\n📊 Pierwsze 10 wierszy danych:")
    print(dane_apple.head(10).to_string(index=False))
    
    # Podstawowe statystyki
    print("\n📈 Podstawowe statystyki:")
    stats = dane_apple[['Open', 'High', 'Low', 'Close/Last', 'Volume']].describe()
    print(stats.round(2))
    
    # Informacje o cenach
    current_price = dane_apple['Close/Last'].iloc[-1]
    first_price = dane_apple['Close/Last'].iloc[0]
    min_price = dane_apple['Close/Last'].min()
    max_price = dane_apple['Close/Last'].max()
    
    print(f"\n💰 Analiza cen Apple:")
    print(f"Pierwsza cena w okresie: ${first_price:.2f}")
    print(f"Ostatnia cena w okresie: ${current_price:.2f}")
    print(f"Najniższa cena: ${min_price:.2f}")
    print(f"Najwyższa cena: ${max_price:.2f}")
    print(f"Zmiana w całym okresie: {((current_price - first_price) / first_price * 100):.2f}%")
    
    # Analiza wolumenu
    avg_volume = dane_apple['Volume'].mean()
    max_volume = dane_apple['Volume'].max()
    min_volume = dane_apple['Volume'].min()
    
    print(f"\n📊 Analiza wolumenu:")
    print(f"Średni dzienny wolumen: {avg_volume:,.0f}")
    print(f"Maksymalny wolumen: {max_volume:,.0f}")
    print(f"Minimalny wolumen: {min_volume:,.0f}")
    
    # Znajdź dni z największymi zmianami
    dane_apple['Daily_Change'] = ((dane_apple['Close/Last'] - dane_apple['Open']) / dane_apple['Open'] * 100)
    biggest_gain = dane_apple.loc[dane_apple['Daily_Change'].idxmax()]
    biggest_loss = dane_apple.loc[dane_apple['Daily_Change'].idxmin()]
    
    print(f"\n📈 Największy dzienny wzrost:")
    print(f"Data: {biggest_gain['Date'][:10]}, Zmiana: +{biggest_gain['Daily_Change']:.2f}%")
    print(f"📉 Największy dzienny spadek:")
    print(f"Data: {biggest_loss['Date'][:10]}, Zmiana: {biggest_loss['Daily_Change']:.2f}%")
    
    return dane_apple

def przygotuj_lstm_data(df, sequence_length=60):
    """
    Przygotowuje dane dla modelu LSTM
    """
    print(f"\n🤖 Przygotowywanie danych dla LSTM (sekwencje po {sequence_length} dni)...")
    
    # Wybierz kolumny numeryczne
    feature_columns = ['Open', 'High', 'Low', 'Close/Last', 'Volume']
    data_for_scaling = df[feature_columns].values.astype(float)
    
    # Normalizacja danych (0-1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_for_scaling)
    
    # Tworzenie sekwencji dla LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])  # Ostatnie N dni jako input
        y.append(scaled_data[i, 3])  # Cena zamknięcia jako target
    
    X, y = np.array(X), np.array(y)
    
    print(f"✅ Kształt danych wejściowych (X): {X.shape}")
    print(f"   - Liczba próbek: {X.shape[0]}")
    print(f"   - Długość sekwencji: {X.shape[1]} dni")
    print(f"   - Liczba cech: {X.shape[2]} (Open, High, Low, Close, Volume)")
    print(f"✅ Kształt danych docelowych (y): {y.shape}")
    
    # Podziel na train/validation/test
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\n📊 Podział danych:")
    print(f"   Trening: {len(X_train)} próbek ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Walidacja: {len(X_val)} próbek ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} próbek ({len(X_test)/len(X)*100:.1f}%)")
    
    # Zapisz przetworzone dane
    np.save('apple_X_train.npy', X_train)
    np.save('apple_y_train.npy', y_train)
    np.save('apple_X_val.npy', X_val)
    np.save('apple_y_val.npy', y_val)
    np.save('apple_X_test.npy', X_test)
    np.save('apple_y_test.npy', y_test)
    np.save('apple_X_all.npy', X)
    np.save('apple_y_all.npy', y)
    
    print("\n💾 Dane LSTM zapisane do plików:")
    print("   - apple_X_train.npy / apple_y_train.npy (zbiór treningowy)")
    print("   - apple_X_val.npy / apple_y_val.npy (zbiór walidacyjny)")
    print("   - apple_X_test.npy / apple_y_test.npy (zbiór testowy)")
    print("   - apple_X_all.npy / apple_y_all.npy (wszystkie dane)")
    
    return X, y, scaler

def stwórz_przykład_lstm():
    """
    Tworzy przykładowy kod modelu LSTM
    """
    lstm_code = '''
# Przykład użycia danych Apple w modelu LSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Załaduj dane
X_train = np.load('apple_X_train.npy')
y_train = np.load('apple_y_train.npy')
X_val = np.load('apple_X_val.npy')
y_val = np.load('apple_y_val.npy')
X_test = np.load('apple_X_test.npy')
y_test = np.load('apple_y_test.npy')

print(f"Dane treningowe: {X_train.shape}, {y_train.shape}")
print(f"Dane walidacyjne: {X_val.shape}, {y_val.shape}")
print(f"Dane testowe: {X_test.shape}, {y_test.shape}")

# Stwórz model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Kompiluj model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Trenuj model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Ewaluacja
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Predykcje
predictions = model.predict(X_test)

# Zapisz model
model.save('apple_lstm_model.h5')
print("Model zapisany jako apple_lstm_model.h5")
'''
    
    with open('apple_lstm_example.py', 'w', encoding='utf-8') as f:
        f.write(lstm_code)
    
    print("📝 Utworzono przykładowy kod LSTM: apple_lstm_example.py")

def main():
    """
    Główna funkcja programu
    """
    print("🍎" * 20)
    print("   APPLE STOCK DATA DOWNLOADER")
    print("🍎" * 20)
    print()
    
    # Pobierz informacje o firmie
    try:
        apple = yf.Ticker("AAPL")
        info = apple.info
        print(f"📋 Informacje o Apple Inc.:")
        print(f"   Pełna nazwa: {info.get('longName', 'Apple Inc.')}")
        print(f"   Sektor: {info.get('sector', 'Technology')}")
        print(f"   Branża: {info.get('industry', 'Consumer Electronics')}")
        print(f"   Kraj: {info.get('country', 'United States')}")
        print(f"   Liczba pracowników: {info.get('fullTimeEmployees', 'N/A'):,}")
        print()
    except:
        print("📋 Apple Inc. - wiodący producent elektroniki użytkowej\n")
    
    # Pobierz i zapisz dane
    dane_apple = zapisz_i_analizuj_dane()
    
    if dane_apple is not None:
        # Przygotuj dane dla LSTM
        X, y, scaler = przygotuj_lstm_data(dane_apple)
        
        # Stwórz przykładowy kod LSTM
        stwórz_przykład_lstm()
        
        print("\n" + "🎉" * 30)
        print("SUKCES! Dane Apple pobrane i przygotowane")
        print("🎉" * 30)
        print("\n📁 Utworzone pliki:")
        print("   1. apple_stock_data.csv - surowe dane giełdowe")
        print("   2. apple_X_train.npy, apple_y_train.npy - dane treningowe")
        print("   3. apple_X_val.npy, apple_y_val.npy - dane walidacyjne")
        print("   4. apple_X_test.npy, apple_y_test.npy - dane testowe")
        print("   5. apple_lstm_example.py - przykładowy kod modelu")
        
        print("\n💡 Następne kroki:")
        print("   1. Uruchom: python apple_lstm_example.py")
        print("   2. Lub załaduj dane ręcznie:")
        print("      X_train = np.load('apple_X_train.npy')")
        print("      y_train = np.load('apple_y_train.npy')")
        
        print(f"\n📊 Gotowe do trenowania na {len(X)} próbkach!")
        print(f"🎯 Predykcja cen Apple na podstawie {X.shape[1]} dni historii")
    
    else:
        print("❌ Program zakończony z błędem")

if __name__ == "__main__":
    main()