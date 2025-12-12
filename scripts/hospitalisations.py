import requests
import pandas as pd
from lets_plot import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def import_hosp(
    url_main="https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/covid-19-synthese-des-indicateurs-de-suivi-de-la-pandemie-dep/exports/json",
    url_backup="https://minio.lab.sspcloud.fr/phobeika/open_eec/covid-19-synthese-des-indicateurs-de-suivi-de-la-pandemie-dep.json"
):
    
    for url in [url_main, url_backup]:
        try:
            print(f"Tentative de téléchargement depuis: {url}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()      # Erreur si status != 200
            
            data = response.json()           # Erreur si JSON invalide
            df = pd.DataFrame(data)
            
            print("Importation réussie.")
            return df
        
        except Exception as e:
            print(f"Le téléchargement a échoué depuis {url}: {e}")
    
    # Si tout a échoué
    raise RuntimeError("❌ L'importation a échoué depuis les deux sources.")




from lets_plot import *
import pandas as pd


from lets_plot import *
import pandas as pd

def plot_hosp_year(df, year=2020):
    """
    Filtre par année (par défaut 2020), vérifie la cohérence des données (même nombre de départements chaque jour),
    agrège les nombres d'hospitalisations par jour pour l'ensemble des départements puis renvoie en graphique l'évolution
    du nombre d'hospitalisations par jour pendant l'année choisie.
    """

    df = df.copy()

    # --- 1. On met la variable `date` au format date ---
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # --- 2. Filtre pour l'année choisie ---
    df_year = df[df["date"].dt.year == year]

    if df_year.empty:
        raise ValueError(f"Aucune donnée trouvée pour l'année {year}")

    # --- 3. Vérifie que les données sont complètes ---
    dept_count = df_year.groupby("date")["dep"].nunique()

    if dept_count.nunique() != 1:
        raise ValueError(
            "Incohérence détectée : le nombre de départements varie selon les dates."
        )

    # Génère un avertissement s'il y a des valeurs manquantes
    if df_year["hosp"].isna().any():
        raise ValueError("Présence de valeurs manquantes dans 'hosp'.")

    # --- 4. Somme le nombre d'hospitalisations à chaque date pour l'ensemble des départements ---
    df_daily = df_year.groupby("date", as_index=False)["hosp"].sum()

    # --- 5. Plot ---
    LetsPlot.setup_html()

    p = (
        ggplot(df_daily, aes(x="date", y="hosp"))
        + geom_line(color="steelblue", size=1.2)
        + ggtitle(f"Hospitalisations Covid - France - {year}")
        + xlab("Date")
        + ylab("Nb hospitalisations")
        + theme_minimal()
    )

    return p



def plot_abs_hosp_by_quarter_base100(df_abs, df_daily, year=2020):
    """
    Trace absences maladie (RABS=2) et hospitalisations Covid par trimestre
    en base 100 relativement au T1.
    """

    # ----------------------------------------------------------------------
    # 1) ABSENCES PAR TRIMESTRE
    # ----------------------------------------------------------------------
    df_rabs = df_abs[df_abs['RABS'] == 2.0].copy()
    df_trim_abs = (
        df_rabs.groupby('TRIM', as_index=False)
               .size()
               .rename(columns={'size': 'Effectifs'})
    )

    df_trim_abs['TRIM'] = pd.Categorical(df_trim_abs['TRIM'],
                                         categories=['T1','T2','T3','T4'],
                                         ordered=True)
    df_trim_abs = df_trim_abs.sort_values('TRIM')
    df_trim_abs['TRIM'] = df_trim_abs['TRIM'].astype('Int64')
    df_trim_abs['TRIM'] = "T" + df_trim_abs['TRIM'].astype(str)



    # --- Base 100 pour les absences ---
    base_abs = df_trim_abs.loc[df_trim_abs['TRIM']=='T1','Effectifs'].values[0]
    df_trim_abs['Index100'] = df_trim_abs['Effectifs'] / base_abs * 100

    # ----------------------------------------------------------------------
    # 2) HOSPITALISATIONS PAR TRIMESTRE
    # ----------------------------------------------------------------------
    data_year = df_daily[df_daily['date'].dt.year == year].copy()
    data_year['month'] = data_year['date'].dt.month

    T1 = data_year[data_year['month'] == 3]['hosp'].sum() * 3
    T2 = data_year[data_year['month'].isin([4,5,6])]['hosp'].sum()
    T3 = data_year[data_year['month'].isin([7,8,9])]['hosp'].sum()
    T4 = data_year[data_year['month'].isin([10,11,12])]['hosp'].sum()

    df_trim_hosp = pd.DataFrame({
        "TRIM": ["T1","T2","T3","T4"],
        "hosp": [T1, T2, T3, T4]
    })

    # --- Base 100 pour les hospitalisations ---
    base_hosp = df_trim_hosp.loc[df_trim_hosp['TRIM']=='T1','hosp'].values[0]
    df_trim_hosp['Index100'] = df_trim_hosp['hosp'] / base_hosp * 100

    # ----------------------------------------------------------------------
    # 3) PLOT EN BASE 100
    # ----------------------------------------------------------------------
    plt.figure(figsize=(10,6))

    plt.plot(df_trim_abs['TRIM'], df_trim_abs['Index100'],
             marker='o', linestyle='-', color='blue', label="Absences (base 100)")

    plt.plot(df_trim_hosp['TRIM'], df_trim_hosp['Index100'],
             marker='s', linestyle='-', color='red', label="Hospitalisations (base 100)")

    plt.title(f"Évolution trimestrielle (base 100 T1) : Absences vs Hospitalisations ({year})")
    plt.xlabel("Trimestre")
    plt.ylabel("Index (Base 100 en T1)")
    plt.xticks(['T1','T2','T3','T4'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df_trim_abs, df_trim_hosp