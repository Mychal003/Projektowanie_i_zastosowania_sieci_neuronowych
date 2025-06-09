"""Preprocessing danych"""
import pandas as pd
import os

def preprocess():
    """Podstawowy preprocessing danych."""
    # Użyj ścieżek względnych
    input_path = "../data/raw/SPY_yahoo.csv"  # Zmień nazwę pliku na ten który masz
    output_dir = "../data/processed"
    output_path = os.path.join(output_dir, "cleaned_data.csv")
    
    # Sprawdź czy plik istnieje w różnych lokalizacjach
    possible_paths = [
        input_path,
        "../data/raw/AAPL_yahoo.csv",
        "../data/raw/SPY.csv",
        "../data/raw/AAPL.csv",
        "data/raw/SPY_yahoo.csv",
        "data/raw/AAPL_yahoo.csv",
    ]
    
    # Znajdź istniejący plik
    existing_file = None
    for path in possible_paths:
        if os.path.exists(path):
            existing_file = path
            print(f"Znaleziono plik: {path}")
            break
    
    if existing_file is None:
        print("Nie znaleziono pliku z danymi!")
        print("Sprawdź czy masz plik CSV w folderze data/raw/")
        print("Szukano w lokalizacjach:")
        for path in possible_paths:
            print(f"  - {path}")
        raise FileNotFoundError("Brak pliku z danymi giełdowymi")
    
    # Wczytanie danych
    df = pd.read_csv(existing_file)
    print(f"Wczytano {len(df)} wierszy z pliku {existing_file}")
    
    # Konwersja Date do datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    
    # Sprawdź nazwy kolumn
    print(f"Kolumny w pliku: {df.columns.tolist()}")
    
    # Obsługa różnych formatów plików Yahoo Finance
    if 'Close' in df.columns:
        # Nowszy format bez znaku dolara
        price_columns = ['Close', 'Open', 'High', 'Low']
        for col in price_columns:
            if col in df.columns:
                df[f'{col}/Last'] = df[col]
    else:
        # Starszy format ze znakiem dolara
        price_columns = ['Close/Last', 'Open', 'High', 'Low']
        for col in price_columns:
            if col in df.columns and df[col].dtype == 'object':
                # Usuń znak dolara jeśli jest
                df[col] = df[col].str.replace('$', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Upewnij się że mamy kolumnę Close/Last
    if 'Close/Last' not in df.columns and 'Close' in df.columns:
        df['Close/Last'] = df['Close']
    
    # Usuń wiersze z brakującymi danymi
    df = df.dropna()
    
    # Zapis przetworzonych danych
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Zapisano przetworzone dane do: {output_path}")
    
    return df


def check_data_structure():
    """Funkcja pomocnicza do sprawdzenia struktury projektu."""
    print("Sprawdzanie struktury katalogów...")
    
    # Sprawdź gdzie jesteśmy
    print(f"Bieżący katalog: {os.getcwd()}")
    
    # Sprawdź strukturę
    dirs_to_check = [
        "data",
        "data/raw",
        "data/processed",
        "../data",
        "../data/raw",
        "../data/processed"
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✓ Katalog istnieje: {dir_path}")
            # Wylistuj pliki CSV
            try:
                files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                if files:
                    print(f"  Pliki CSV: {files}")
            except:
                pass
        else:
            print(f"✗ Brak katalogu: {dir_path}")


if __name__ == "__main__":
    # Jeśli uruchamiasz ten plik bezpośrednio
    check_data_structure()
    try:
        df = preprocess()
        print(f"\nPrzetworzono pomyślnie {len(df)} wierszy danych")
        print(f"Kolumny: {df.columns.tolist()}")
        print(f"Zakres dat: {df['Date'].min()} - {df['Date'].max()}")
    except Exception as e:
        print(f"\nBłąd: {e}")
        print("\nUpewnij się że masz plik CSV z danymi giełdowymi w folderze data/raw/")