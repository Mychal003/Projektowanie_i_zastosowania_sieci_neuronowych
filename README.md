# Projekt porównania sieci neuronowych w prognozowaniu akcji

Ten projekt ma na celu porównanie wydajności różnych architektur sieci neuronowych (LSTM oraz CNN) w zadaniu przewidywania cen akcji.

## Struktura projektu

```
projekt-porownanie-sieci/
├── data/
│   ├── raw/                # Surowe dane pobrane z giełdy
│   └── processed/          # Dane po wstępnym przetworzeniu
├── notebooks/
│   ├── 01_EDA.ipynb        # Notebook do analizy eksploracyjnej danych
│   ├── 02_LSTM_Model.ipynb # Notebook do trenowania i ewaluacji modelu LSTM
│   └── 03_CNN_Model.ipynb  # Notebook do trenowania i ewaluacji modelu konwolucyjnego
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Skrypty do wstępnego przetwarzania danych
│   ├── eda.py                 # Funkcje pomocnicze do analizy eksploracyjnej
│   ├── lstm_model.py          # Definicja architektury sieci LSTM
│   ├── cnn_model.py           # Definicja architektury sieci konwolucyjnej
│   └── utils.py               # Narzędzia pomocnicze (np. wizualizacje, metryki)
├── models/                    # Zapisywane modele
│   ├── lstm/                  # Modele LSTM (np. pliki .h5 dla Keras)
│   └── cnn/                   # Modele konwolucyjne
├── requirements.txt           # Lista bibliotek niezbędnych do uruchomienia projektu
└── README.md                  # Dokumentacja projektu, instrukcje uruchomienia, opis struktury
```

## Opis projektu

Projekt koncentruje się na porównaniu dwóch różnych architektur sieci neuronowych:
- **LSTM (Long Short-Term Memory)** - sieć rekurencyjna dobrze radząca sobie z analizą szeregów czasowych
- **CNN (Convolutional Neural Network)** - sieć konwolucyjna, która może wykrywać wzorce w danych

Porównanie ma na celu określenie, która architektura lepiej sprawdza się w zadaniu prognozowania cen akcji.

## Instalacja

1. Sklonuj repozytorium:
```
git clone https://github.com/Mychal003/Projektowanie_i_zastosowania_sieci_neuronowych.git
cd Projektowanie_i_zastosowania_sieci_neuronowych
```

2. Zainstaluj wymagane biblioteki:
```
pip install -r requirements.txt
```

## Użycie

1. Umieść surowe dane w katalogu `data/raw/`
2. Uruchom notebook do przetwarzania danych i analizy eksploracyjnej:
```
jupyter notebook notebooks/01_EDA.ipynb
```
3. Trenuj i ewaluuj modele używając odpowiednich notebooków:
```
jupyter notebook notebooks/02_LSTM_Model.ipynb
jupyter notebook notebooks/03_CNN_Model.ipynb
```

## Struktura kodu

- **data_preprocessing.py** - zawiera funkcje do przygotowania danych, normalizacji i podziału na zbiory treningowe/testowe
- **eda.py** - funkcje do analizy eksploracyjnej i wizualizacji danych
- **lstm_model.py** - implementacja sieci LSTM
- **cnn_model.py** - implementacja sieci konwolucyjnej
- **utils.py** - funkcje pomocnicze, wizualizacje wyników i obliczanie metryk

## Wymagania

- Python 3.8+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

## Licencja

MIT
