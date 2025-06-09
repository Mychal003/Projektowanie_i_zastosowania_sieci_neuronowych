# quick_fix.py - Napraw format MultiIndex z Yahoo Finance

import pandas as pd
import os

def fix_multiindex_csv():
    """Napraw plik CSV z MultiIndex"""
    
    # Ścieżka do pliku
    base_path = r"notebooks"
    input_path = os.path.join(base_path, "data", "processed", "cleaned_data.csv")
    
    print("Naprawiam format danych...")
    
    # Wczytaj plik
    # Pierwszy wiersz to Price, Date, Close, High, Low, Open, Volume
    # Drugi wiersz to Ticker, '', SPY, SPY, SPY, SPY, SPY
    # Pomijamy te dwa wiersze nagłówka
    df = pd.read_csv(input_path, skiprows=2, header=None)
    
    # Nadaj właściwe nazwy kolumn
    df.columns = ['Index', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Usuń kolumnę Index (niepotrzebna)
    df = df.drop('Index', axis=1)
    
    # Konwertuj Date do datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Zmień nazwę Close na Close/Last
    df = df.rename(columns={'Close': 'Close/Last'})
    
    # Upewnij się, że wszystkie kolumny numeryczne są float
    numeric_cols = ['Close/Last', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Usuń ewentualne NaN
    df = df.dropna()
    
    # Sortuj po dacie
    df = df.sort_values('Date')
    
    print(f"✅ Naprawiono {len(df)} wierszy danych")
    print(f"Kolumny: {df.columns.tolist()}")
    print(f"\nPierwsze wiersze:")
    print(df.head())
    
    # Statystyki
    print(f"\nStatystyki cen zamknięcia:")
    print(f"Min: ${df['Close/Last'].min():.2f}")
    print(f"Max: ${df['Close/Last'].max():.2f}")
    print(f"Średnia: ${df['Close/Last'].mean():.2f}")
    
    # Zapisz poprawiony plik
    output_path = os.path.join(base_path, "data", "processed", "cleaned_data_fixed.csv")
    df.to_csv(output_path, index=False)
    
    # Nadpisz oryginalny plik
    df.to_csv(input_path, index=False)
    
    print(f"\n✅ Zapisano poprawione dane do:")
    print(f"   - {input_path}")
    print(f"   - {output_path} (kopia)")
    
    return df

def alternative_download():
    """Alternatywna metoda pobierania - prosta i skuteczna"""
    import yfinance as yf
    
    print("\n" + "="*60)
    print("ALTERNATYWNA METODA - NAJPROSTSZA")
    print("="*60)
    
    # Pobierz dane
    ticker = 'SPY'
    print(f"Pobieranie {ticker}...")
    
    # Użyj starego API które zwraca prosty DataFrame
    spy = yf.Ticker(ticker)
    df = spy.history(start="2015-01-01", end="2025-06-08")
    
    # Reset indeksu
    df.reset_index(inplace=True)
    
    # Zmień nazwy kolumn
    df = df.rename(columns={
        'Close': 'Close/Last',
        'Date': 'Date'
    })
    
    # Wybierz tylko potrzebne kolumny
    df = df[['Date', 'Open', 'High', 'Low', 'Close/Last', 'Volume']]
    
    # Zapisz
    base_path = r"notebooks"
    output_path = os.path.join(base_path, "data", "processed", "cleaned_data.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Pobrano i zapisano {len(df)} dni danych")
    print(f"Ścieżka: {output_path}")
    print(f"\nPrzykład danych:")
    print(df.head())
    
    return df

# Główna funkcja
if __name__ == "__main__":
    print("Co zrobić?")
    print("1. Napraw istniejący plik CSV")
    print("2. Pobierz dane od nowa (prostą metodą)")
    
    choice = input("\nWybór (1 lub 2): ")
    
    if choice == '1':
        try:
            df = fix_multiindex_csv()
            print("\n🎉 SUKCES! Dane naprawione i gotowe do użycia!")
        except Exception as e:
            print(f"❌ Błąd: {e}")
            print("Spróbuj opcji 2")
    
    elif choice == '2':
        try:
            df = alternative_download()
            print("\n🎉 SUKCES! Dane pobrane i gotowe do użycia!")
        except Exception as e:
            print(f"❌ Błąd: {e}")
    
    else:
        print("Nieprawidłowy wybór!")
    
    print("\n" + "="*60)
    print("CO DALEJ?")
    print("="*60)
    print("1. Uruchom swój notebook LSTM")
    print("2. Dane są w: data/processed/cleaned_data.csv")
    print("3. Model powinien teraz działać znacznie lepiej z danymi SPY!")
    print("\nPowodzenia! 🚀")