import requests
import pandas as pd
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


def plot_hosp(df, year=2020):
    """
    Filtre les données pour une année donnée, vérifie la complétude
    et trace la somme quotidienne des hospitalisations.
    """

    # --- 1. Filter by year ---
    df_year = df[df["date"].str.contains(str(year), na=False)].copy()
    
    if df_year.empty:
        raise ValueError(f"Aucune donnée trouvée pour l'année {year}")

    # --- 2. Vérification qu'il existe bien une donné par date et par département ---
    dept_count = df_year.groupby("date")["dep"].nunique()

    if dept_count.nunique() != 1:
        raise ValueError("Incohérence : le nombre de départements varie selon les dates.")

    if df_year["hosp"].isna().any():
        raise ValueError("Présence de valeurs manquantes dans 'hosp'.")


    # --- 3. Aggregate hospitalisations ---
    df_daily = df_year.groupby("date", as_index=False)["hosp"].sum()
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    # --- 4. Plot ---
    plt.figure(figsize=(14, 6))
    plt.plot(df_daily["date"], df_daily["hosp"])

    # Month labels
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # X-axis limits for the selected year
    plt.xlim(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))

    plt.title(f"Hospitalisations Covid - France - {year}")
    plt.xlabel("Date")
    plt.ylabel("Nb hospitalisations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()