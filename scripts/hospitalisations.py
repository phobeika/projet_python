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
        + geom_line(color="#248BFF", size=1.2)
        + ggtitle(f"Hospitalisations Covid - France - {year}")
        + xlab("Date")
        + ylab("Nb hospitalisations")
        + theme_minimal()
        + ggsize(800, 400)
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
    df_rabs = df_abs[df_abs['RABS'] == '2'].copy()
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


def plot_hosp_arrets_trim(df_eec, df_daily, year = 2020):
    """
    Traite les données d'enquête emploi et d'hospitalisation pour une année (par défaut 2020),
    calcule les agrégats par trimestre et affiche un graphique à double axe.
    
    arguments:
        df_eec (pd.DataFrame): DataFrame contenant l'enquête emploi (colonne 'RABS', 'TRIM')
        df_daily (pd.DataFrame): DataFrame contenant les données hospitalières (colonne 'date', 'hosp')
        year : l'année pour laquelle on fait l'analyse
    outputs:
        pd.DataFrame: Le dataframe fusionné utilisé pour le graphique.
    """
    
    # --- 0. On s'assure que df_eec contient bien l'année 'year'
    df_eec = df_eec[df_eec['ANNEE'] == year].copy()
    if df_eec.empty:
        raise ValueError(f"Aucune donnée trouvée pour l'année {year}")

    # --- 1. Traitement des Arrêts Maladie (Source: EEC) ---
    # On ne compte que les absences pour congé maladie (RABS = '2')
    df_rabs2 = df_eec[df_eec['RABS'] == '2'].copy()
    
    # Groupement par trimestre et comptage
    df_trim_count = df_rabs2.groupby('TRIM', as_index=False).size().rename(columns={'size': 'Effectifs'})
    
    # Formatage de la colonne TRIM (ajout du 'T' devant le numéro)
    df_trim_count['TRIM'] = 'T' + df_trim_count['TRIM'].astype(str)


    # --- 2. Traitement des Hospitalisations (Source: df_daily) ---
    # Création d'une copie pour ne pas modifier l'original
    df_daily_clean = df_daily.copy()
    
    # Extraction année et mois (si ce n'est pas déjà fait)
    if not pd.api.types.is_datetime64_any_dtype(df_daily_clean['date']):
        df_daily_clean['date'] = pd.to_datetime(df_daily_clean['date'])
        
    df_daily_clean['year'] = df_daily_clean['date'].dt.year
    df_daily_clean['month'] = df_daily_clean['date'].dt.month
    
    # Filtrage sur year
    df_daily_year = df_daily_clean[df_daily_clean['year'] == year]

    # Calcul spécifique des trimestres (selon votre logique : T1 = Mars uniquement)
    t1 = df_daily_year[df_daily_year['month'] == 3]['hosp'].sum()
    t2 = df_daily_year[df_daily_year['month'].isin([4, 5, 6])]['hosp'].sum()
    t3 = df_daily_year[df_daily_year['month'].isin([7, 8, 9])]['hosp'].sum()
    t4 = df_daily_year[df_daily_year['month'].isin([10, 11, 12])]['hosp'].sum()

    df_hosp_trim = pd.DataFrame({
        "TRIM": ["T1", "T2", "T3", "T4"],
        "hosp": [t1, t2, t3, t4]
    })


    # --- 3. Fusion des données ---
    df_fusionne = pd.merge(
        df_hosp_trim,
        df_trim_count,
        on='TRIM',
        how='left'
    )


    # --- 4. Création du Graphique ---
    # Définition des couleurs
    color_hosp = '#248BFF'  # Bleu Insee
    color_eff = '#7A57FF'   # Violet Insee

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Premier axe (Hospitalisations)
    ax1.set_xlabel('Trimestre')
    ax1.set_ylabel('Hospitalisations (en millions)', color=color_hosp, fontsize=12)
    ax1.plot(df_fusionne['TRIM'], df_fusionne['hosp'], color=color_hosp, marker='o', linewidth=2, label='Hospitalisations')
    ax1.tick_params(axis='y', labelcolor=color_hosp)
    
    # Ajout d'une grille verticale légère pour la lisibilité
    ax1.grid(axis='x', alpha=0.3)

    # Second axe (Arrêts maladie)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Arrêts maladie', color=color_eff, fontsize=12)
    ax2.plot(df_fusionne['TRIM'], df_fusionne['Effectifs'], color=color_eff, marker='^', linewidth=2, label='Arrêts maladie')
    ax2.tick_params(axis='y', labelcolor=color_eff)

    plt.title(f'Hospitalisations liées au Covid-19 et arrêts maladie en {year}')
    plt.show()
    
    correlation = df_fusionne[['hosp', 'Effectifs']].corr().iloc[0, 1]
    print('Coefficient de corrélation entre les deux variables :', round(correlation,2))

    # return df_fusionne