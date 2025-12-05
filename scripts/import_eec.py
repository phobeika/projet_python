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

def read_from_zip(url, backup_url=None, filename_keyword=None, **kwargs):
    """
    Télécharge un ZIP depuis une URL (ou sa version de secours), lit un fichier CSV ou DBF à l'intérieur et l'importe sous la forme d'un dataframe Panda
    
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
            files = [f for f in myzip.namelist() if f.endswith(('.csv', '.bdf'))]
            if len(files) == 0:
                raise ValueError(f"Aucun fichier CSV ou BDF trouvé dans le ZIP à {url}")

            # Si mot-clé fourni
            if filename_keyword:
                files = [f for f in csv_files if filename_keyword in f]
                if len(files) == 0:
                    raise ValueError(f"Aucun CSV contenant '{filename_keyword}' trouvé à {url}")
                elif len(files) > 1:
                    raise ValueError(f"Plusieurs CSV contiennent '{filename_keyword}' à {url}: {csv_files}")

            # Choix du fichier le plus gros si pas de mot clé
            if len(files) > 1:
                sizes = {f: myzip.getinfo(f).file_size for f in files}
                selected = max(sizes, key=sizes.get)
                print(f"Aucun mot-clé : sélection du fichier le plus gros : {selected}")
            else:
                selected = files[0]

# ------ LECTURE DES FICHIERS -------
            if selected.endswith(".csv"):
                with myzip.open(selected) as file:
                    return pd.read_csv(file, **kwargs)

            elif selected.endswith(".bdf"):
                # Extraction du BDF dans un buffer
                bdf_bytes = BytesIO(myzip.read(selected))
                raw = mne.io.read_raw_bdf(bdf_bytes, preload=True, **kwargs)
                
                # Conversion direct → DataFrame
                df = raw.to_data_frame()
                return df

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
