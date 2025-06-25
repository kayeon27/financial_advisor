import os
import pandas as pd

class FinancialDataLoader:
    #constructeur
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier introuvable {csv_path}")
        self.csv_path = csv_path

    #methode de chargement des donnees    
    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df.to_dict(orient='records') #convertion des données sous forme de dictionnaire(chaque dictionnaire = ligne du csv)       