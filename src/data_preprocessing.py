import pandas as pd
import os

def preprocess():
    input_path = "C:\\Users\\pawli\\Desktop\\sieci\\Projektowanie_i_zastosowania_sieci_neuronowych\\data\\raw\\AAPL_yahoo.csv" ## TRZEBA TO POPRAWIC
    output_dir = "C:\\Users\\pawli\\OneDrive\\Pulpit\\Projekt_sieci\\data\\processed"
    output_path = os.path.join(output_dir, "cleaned_data.csv")

    # Wczytanie danych
    df = pd.read_csv(input_path)

    # Konwersja "Date" do datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)

    # Usuwanie znaku dolara i konwersja do float
    for i in ["Close/Last", "Open", "High", "Low"]:
        df[i] = df[i].str.replace("$", "", regex=False)
        df[i] = pd.to_numeric(df[i])

    # Zapis przetworzonych danych
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
