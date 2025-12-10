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