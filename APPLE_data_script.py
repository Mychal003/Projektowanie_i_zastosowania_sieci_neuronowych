#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prosty downloader danych Apple - tylko CSV
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
    """Sprawdza i instaluje biblioteki"""
    required_packages = ['yfinance', 'pandas']
    
    print("🔍 Sprawdzam zależności...")
    for package in required_packages:
        install_package(package)
    print("✅ Wszystkie zależności gotowe!\n")

# Sprawdź i zainstaluj zależności
check_and_install_dependencies()

# Importuj biblioteki
try:
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    print("✅ Biblioteki zaimportowane pomyślnie!\n")
except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    sys.exit(1)

def pobierz_dane_apple():
    """
    Pobiera dane Apple i zapisuje do CSV
    """
    print("🍎 Pobieranie danych Apple (AAPL)...")
    
    try:
        # Ostatnie 3 lata
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1095)
        
        print(f"📅 Okres: {start_date.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}")
        
        # Pobierz dane
        apple = yf.Ticker("AAPL")
        data = apple.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError("Nie udało się pobrać danych")
        
        # Formatuj dane
        data = data.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        
        # Zmień nazwy kolumn
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close/Last', 'Volume', 'Dividends', 'Stock Splits']
        
        # Zostaw tylko potrzebne kolumny
        data_final = data[['Date', 'Open', 'High', 'Low', 'Close/Last', 'Volume']].copy()
        
        # Zapisz do CSV
        filename = 'apple_stock_data.csv'
        data_final.to_csv(filename, index=False)
        
        print(f"✅ Pobrano {len(data_final)} rekordów")
        print(f"💾 Zapisano do pliku: {filename}")
        
        # Pokaż próbkę danych
        print("\n📊 Pierwsze 5 wierszy:")
        print(data_final.head().to_string(index=False))
        
        print("\n📊 Ostatnie 5 wierszy:")
        print(data_final.tail().to_string(index=False))
        
        # Podstawowe info
        first_price = data_final['Close/Last'].iloc[0]
        last_price = data_final['Close/Last'].iloc[-1]
        change = ((last_price - first_price) / first_price * 100)
        
        print(f"\n💰 Podsumowanie:")
        print(f"Pierwsza cena: ${first_price:.2f}")
        print(f"Ostatnia cena: ${last_price:.2f}")
        print(f"Zmiana w okresie: {change:.2f}%")
        
        return data_final
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return None

def main():
    print("🍎" * 30)
    print("   APPLE DATA DOWNLOADER")
    print("🍎" * 30)
    print()
    
    dane = pobierz_dane_apple()
    
    if dane is not None:
        print("\n🎉 GOTOWE! Plik apple_stock_data.csv utworzony")
        print("📋 Format: Date,Open,High,Low,Close/Last,Volume")
        print("✅ Dane gotowe do użycia w modelu LSTM")
    else:
        print("❌ Błąd podczas pobierania danych")

if __name__ == "__main__":
    main()