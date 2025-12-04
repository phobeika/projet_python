import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
from lets_plot import *
from palmerpenguins import load_penguins
from urllib import request
from io import BytesIO
from sklearn.cluster import KMeans

def read_csv_from_zip(url, backup_url=None, filename_keyword=None, **kwargs):
    """
    Télécharge un ZIP depuis une URL (ou sa version de secours) et lit un CSV à l'intérieur.
    
    Paramètres :
    - url : URL principale du ZIP
    - backup_url : URL de secours à utiliser en cas d'échec
    - filename_keyword : mot-clé pour filtrer le CSV à ouvrir (facultatif)
    - kwargs : paramètres additionnels passés à pd.read_csv
    
    Retour :
    - DataFrame pandas
    """
    
    def try_read(url):
        """Sous-fonction : essaye de lire un ZIP depuis une URL donnée"""
        response = requests.get(url)
        response.raise_for_status()  # lèvera une erreur si le téléchargement échoue
        zip_bytes = BytesIO(response.content)

        with ZipFile(zip_bytes) as myzip:
            csv_files = [f for f in myzip.namelist() if f.endswith('.csv')]
            if len(csv_files) == 0:
                raise ValueError(f"Aucun fichier CSV trouvé dans le ZIP à {url}")

            # Si mot-clé fourni
            if filename_keyword:
                csv_files = [f for f in csv_files if filename_keyword in f]
                if len(csv_files) == 0:
                    raise ValueError(f"Aucun CSV contenant '{filename_keyword}' trouvé à {url}")
                elif len(csv_files) > 1:
                    raise ValueError(f"Plusieurs CSV contiennent '{filename_keyword}' à {url}: {csv_files}")

            # Si plusieurs CSV et pas de mot-clé, prendre le plus gros
            if len(csv_files) > 1:
                file_sizes = {f: myzip.getinfo(f).file_size for f in csv_files}
                largest_file = max(file_sizes, key=file_sizes.get)
                print(f"Aucun mot-clé fourni : sélection du CSV le plus gros : {largest_file}")
                csv_files = [largest_file]

            with myzip.open(csv_files[0]) as file:
                return pd.read_csv(file, **kwargs)

    # --- Étape 1 : tentative principale ---
    try:
        print(f"Téléchargement depuis l'URL principale : {url}")
        return try_read(url)
    
    # --- Étape 2 : tentative backup ---
    except Exception as e:
        if backup_url:
            print(f"⚠️ Erreur avec l'URL principale ({e}). Tentative avec le backup : {backup_url}")
            try:
                return try_read(backup_url)
            except Exception as e2:
                raise RuntimeError(f"Échec avec les deux URLs.\nErreur principale : {e}\nErreur backup : {e2}")
        else:
            raise RuntimeError(f"Échec avec l'URL principale et aucun backup fourni.\nErreur : {e}")
