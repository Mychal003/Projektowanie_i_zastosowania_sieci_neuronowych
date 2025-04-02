projekt-akcje/
├── data/
│   ├── raw/                # Surowe dane pobrane z giełdy
│   └── processed/          # Dane po wstępnym przetworzeniu
├── notebooks/
│   ├── 01_EDA.ipynb        # Notebook do analizy eksploracyjnej
│   └── 02_Model_Training.ipynb  # Notebook do trenowania i ewaluacji modelu
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Skrypty do wstępnego przetwarzania danych
│   ├── eda.py                 # Funkcje pomocnicze do EDA (można też część kodu mieć bezpośrednio w notebooku)
│   ├── model.py               # Definicja architektury sieci neuronowej
│   └── utils.py               # Narzędzia pomocnicze (np. wizualizacje, metryki)
├── models/                    # Zapisywane modele (np. pliki .h5 dla Keras/PyTorch)
├── requirements.txt           # Lista bibliotek potrzebnych do uruchomienia projektu
└── README.md                  # Dokumentacja projektu, instrukcje uruchomienia, opis struktury
