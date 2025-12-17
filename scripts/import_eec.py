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





def plot_score_exposition(
    df,
    score_col="score_exposition",
    by=None,
    labels=None,
    figsize=(8,5),
    title=None
):
    """
    Trace la distribution du score d'exposition.
    
    - Sans `by` : barres simples + pourcentages
    - Avec `by` : barres empilées selon la variable
    """

    # ---------------------------
    # Labels par défaut intégrés
    # ---------------------------
    default_labels = {
        "SEXE": {1: "Homme", 2: "Femme"},
        "CSER": {
            0: "Non renseigné",
            1: "Agriculteurs exploitants",
            2: "Artisans, commerçants et chefs d'entreprise",
            3: "Cadres et prof. intellectuelles supérieures",
            4: "Professions intermédiaires",
            5: "Employés",
            6: "Ouvriers"
        }
    }

    # ===========================
    # CAS 1 — graphique simple
    # ===========================
    if by is None:
        effectifs = df[score_col].value_counts().sort_index()
        frequences = effectifs / effectifs.sum() * 100

        plt.figure(figsize=figsize)
        bars = plt.bar(effectifs.index, effectifs.values, edgecolor="black")

        for bar, pct in zip(bars, frequences):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom"
            )

        plt.xlabel("Score d'exposition")
        plt.ylabel("Nombre d'individus")
        plt.title(title or "Distribution du score d'exposition")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # ===========================
    # CAS 2 — graphique empilé
    # ===========================
    df_plot = df.copy()

    # Choix des labels
    if labels is not None:
        label_map = labels
    elif by in default_labels:
        label_map = default_labels[by]
    else:
        label_map = None  # aucun renommage

    if label_map is not None:
        df_plot[by] = df_plot[by].map(label_map)

    table = pd.crosstab(df_plot[score_col], df_plot[by])

    plt.figure(figsize=figsize)
    bottom = np.zeros(len(table))

    for col in table.columns:
        plt.bar(
            table.index,
            table[col],
            bottom=bottom,
            label=col,
            edgecolor="black"
        )
        bottom += table[col].values

    # Pourcentages dans les barres
    for i, row in table.iterrows():
        total = row.sum()
        cumul = 0
        for val in row:
            if val > 0:
                plt.text(
                    i,
                    cumul + val / 2,
                    f"{val/total*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                )
            cumul += val

    plt.xlabel("Score d'exposition")
    plt.ylabel("Nombre d'individus")
    plt.title(title or f"Distribution du score d'exposition par {by}")
    plt.legend(title=by, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()
