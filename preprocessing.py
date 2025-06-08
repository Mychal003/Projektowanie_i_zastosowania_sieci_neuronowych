# preprocessing.py - Preprocessing danych z Yahoo Finance

import pandas as pd
import os
import numpy as np
from datetime import datetime

def preprocess(input_filename='SPY_yahoo.csv', ticker_name='SPY'):
    """
    Preprocessing danych pobranych z Yahoo Finance
    
    Args:
        input_filename: nazwa pliku w folderze data/raw/
        ticker_name: nazwa tickera (dla informacji)
    """
    
    # Ścieżki - dostosuj do swojej struktury
    base_path = "C:\\Users\\pawli\\Desktop\\sieci\\Projektowanie_i_zastosowania_sieci_neuronowych"
    input_path = os.path.join(base_path, "data", "raw", input_filename)
    output_dir = os.path.join(base_path, "data", "processed")
    output_path = os.path.join(output_dir, "cleaned_data.csv")
    
    print(f"Preprocessing danych {ticker_name}...")
    print(f"Ścieżka wejściowa: {input_path}")
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(input_path):
        print(f"❌ Błąd: Plik {input_path} nie istnieje!")
        print("Dostępne pliki w data/raw/:")
        raw_dir = os.path.join(base_path, "data", "raw")
        if os.path.exists(raw_dir):
            files = os.listdir(raw_dir)
            for f in files:
                print(f"  - {f}")
        return None
    
    # Wczytanie danych
    df = pd.read_csv(input_path)
    print(f"✓ Wczytano {len(df)} wierszy")
    
    # Sprawdź kolumny
    print(f"Kolumny w pliku: {df.columns.tolist()}")
    
    # Yahoo Finance ma różne formaty - sprawdź który mamy
    if 'Date' in df.columns:
        # Format z kolumną Date
        df['Date'] = pd.to_datetime(df['Date'])
    elif df.index.name == 'Date':
        # Format gdzie Date jest indeksem
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print("❌ Błąd: Brak kolumny Date!")
        return None
    
    # Sortowanie po dacie
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Sprawdź czy dane mają symbol dolara (starszy format)
    if 'Close' in df.columns and isinstance(df['Close'].iloc[0], str):
        print("Usuwanie symboli dolara...")
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = df[col].str.replace('$', '', regex=False)
                df[col] = pd.to_numeric(df[col])
    
    # Standaryzacja nazw kolumn dla kompatybilności z Twoim kodem
    column_mapping = {
        'Close': 'Close/Last',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Volume': 'Volume',
        'Date': 'Date'
    }
    
    # Zmień nazwy kolumn jeśli potrzeba
    df_renamed = df.rename(columns=column_mapping)
    
    # Wybierz tylko potrzebne kolumny
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close/Last', 'Volume']
    
    # Sprawdź które kolumny są dostępne
    available_cols = [col for col in required_cols if col in df_renamed.columns]
    missing_cols = [col for col in required_cols if col not in df_renamed.columns]
    
    if missing_cols:
        print(f"⚠️ Brakujące kolumny: {missing_cols}")
        # Spróbuj naprawić brakujące kolumny
        if 'Close/Last' not in df_renamed.columns and 'Close' in df.columns:
            df_renamed['Close/Last'] = df['Close']
            available_cols.append('Close/Last')
    
    # Wybierz dostępne kolumny
    df_final = df_renamed[available_cols].copy()
    
    # Usuń wiersze z brakującymi danymi
    before_drop = len(df_final)
    df_final = df_final.dropna()
    after_drop = len(df_final)
    if before_drop > after_drop:
        print(f"Usunięto {before_drop - after_drop} wierszy z brakującymi danymi")
    
    # Dodatkowe czyszczenie
    # Usuń duplikaty dat jeśli istnieją
    df_final = df_final.drop_duplicates(subset=['Date'])
    
    # Sprawdź czy Volume jest numeryczny i > 0
    if 'Volume' in df_final.columns:
        df_final = df_final[df_final['Volume'] > 0]
    
    # Statystyki
    print(f"\n=== Statystyki po przetworzeniu ===")
    print(f"Zakres dat: {df_final['Date'].min()} - {df_final['Date'].max()}")
    print(f"Liczba dni: {len(df_final)}")
    print(f"Kolumny: {df_final.columns.tolist()}")
    
    if 'Close/Last' in df_final.columns:
        print(f"\nCeny zamknięcia:")
        print(f"Min: ${df_final['Close/Last'].min():.2f}")
        print(f"Max: ${df_final['Close/Last'].max():.2f}")
        print(f"Średnia: ${df_final['Close/Last'].mean():.2f}")
    
    # Zapis
    os.makedirs(output_dir, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Zapisano przetworzone dane do: {output_path}")
    
    return df_final

def preprocess_multiple_files():
    """Przetwórz wszystkie pliki CSV w folderze raw"""
    base_path = "C:\\Users\\pawli\\Desktop\\sieci\\Projektowanie_i_zastosowania_sieci_neuronowych"
    raw_dir = os.path.join(base_path, "data", "raw")
    
    if not os.path.exists(raw_dir):
        print(f"❌ Folder {raw_dir} nie istnieje!")
        return
    
    # Lista plików CSV
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ Brak plików CSV w folderze raw!")
        return
    
    print(f"Znaleziono {len(csv_files)} plików CSV:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")
    
    # Wybór pliku
    choice = input("\nKtóry plik przetworzyć? (numer lub nazwa): ")
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(csv_files):
            selected_file = csv_files[idx]
        else:
            print("❌ Nieprawidłowy numer!")
            return
    else:
        selected_file = choice if choice.endswith('.csv') else choice + '.csv'
    
    # Przetwórz wybrany plik
    ticker = selected_file.replace('_yahoo.csv', '').replace('.csv', '')
    df = preprocess(selected_file, ticker)
    
    return df

# Główna funkcja
if __name__ == "__main__":
    print("="*60)
    print("PREPROCESSING DANYCH GIEŁDOWYCH")
    print("="*60)
    
    # Sprawdź czy istnieją pliki do przetworzenia
    df = preprocess_multiple_files()
    
    if df is not None:
        print("\n✅ Preprocessing zakończony sukcesem!")
        print("Możesz teraz uruchomić notebook LSTM.")
    else:
        print("\n❌ Preprocessing nie powiódł się.")
        print("\nUpewnij się, że:")
        print("1. Pobrałeś dane używając wcześniejszego skryptu")
        print("2. Pliki CSV znajdują się w folderze data/raw/")
        print("3. Ścieżki w kodzie są poprawne")