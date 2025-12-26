import requests
import pandas as pd
import lets_plot as lp
import matplotlib.pyplot as plt


# Import des données d'hospitalisation depuis Santé Publique France
def import_hosp(url_main, url_backup):
    """
    Cette fonction  importe les données d'hospitalisations depuis le site de Santé publique France

    Arguments :
        - url_main : url principal pour accéder aux données ;
        - url_backup : url de sécurité.
    """

    # Tentative de téléchargement
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


# Représentation des arrêts de travail en 2020
def plot_hosp_year(df, year=2020):
    """
    Cette fonction :
        - filtre les données d'hospitalisation par année ;
        - vérifie la cohérence des données (même nombre de départements chaque jour ;
        - agrège les hospitalisations par jour pour l'ensemble des départements ;
        - renvoie en graphique l'évolution du nombre d'hospitalisations par jour pendant l'année choisie.

    Arguments :
        - df : dataframe en entrée ;
        - year : année désirée.
    """

    # Copie des données
    df = df.copy()


    # Conversion des données au dormat date
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # Sélection de l'année choisie
    df_year = df[df["date"].dt.year == year]
    if df_year.empty:
        raise ValueError(f"❌ Aucune donnée trouvée pour l'année {year}")

    # Vérification de la complétude des données
    dept_count = df_year.groupby("date")["dep"].nunique()
    if dept_count.nunique() != 1:
        raise ValueError(
            "⚠️ Incohérence détectée : le nombre de départements varie selon les dates."
        )
    if df_year["hosp"].isna().any():
        raise ValueError("⚠️ Présence de valeurs manquantes dans 'hosp'.")


    # Agrégation des hospitalisations par date
    df_daily = df_year.groupby("date", as_index=False)["hosp"].sum()


    # Graphique
    tous_les_noms = [
        "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
        "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
    ]
    toutes_les_dates = pd.date_range(start=f'{year}-01-01', periods=12, freq='MS')
    ticks_dates = toutes_les_dates[2:]
    ticks_noms = tous_les_noms[2:]
    date_debut = ticks_dates[0]

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_daily["date"], 
        df_daily["hosp"], 
        color='#248BFF', 
        linewidth=2,
        label='Nombre d\'hospitalisations'
    )

    plt.fill_between(
        df_daily["date"], 
        df_daily["hosp"], 
        color='#248BFF', 
        alpha=0.4
    )

    plt.grid(True, alpha=0.3)
    plt.title(f"Hospitalisations liées au Covid-19 en France en {year}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Nombre d'hospitalisations")

    plt.xticks(
        ticks=ticks_dates, 
        labels=ticks_noms, 
        rotation=45,
        ha='right'
    )
    plt.xlim(left=date_debut, right=toutes_les_dates[-1] + pd.Timedelta(days=30))
    plt.tight_layout()

    # Résultat
    plt.show()


# Comparaison des hospitalisations et des arrêts de travail
def plot_hosp_arrets_trim(df_eec, df_daily, year=2020):
    """
    Cette fonction :
        - traite les données d'enquête emploi et d'hospitalisation pour une année (par défaut 2020) ;
        - calcule les agrégats par trimestre et affiche un graphique à deux axes des ordonnées.

    Arguments:
        - df_eec : dataset en entrée contenant l'enquête emploi (colonne 'RABS', 'TRIM')
        - df_daily : dataframe contenant les données hospitalières (colonne 'date', 'hosp')
        - year : l'année pour laquelle on fait l'analyse
    
    Outputs:
        pd : le dataframe fusionné utilisé pour le graphique.
    """

    # Import des données
    df_eec = df_eec[df_eec['ANNEE'] == year].copy()
    if df_eec.empty:
        raise ValueError(f"❌ Aucune donnée trouvée pour l'année {year}")


    # Traitement des Arrêts Maladie (Source: EEC)
    
    # On ne compte que les absences pour congé maladie (RABS = '2')
    df_rabs2 = df_eec[df_eec['RABS'] == '2'].copy()

    # Agrégation par trimestre
    df_trim_count = df_rabs2.groupby('TRIM', as_index=False)\
        .size().rename(columns={'size': 'Effectifs'})

    # Formatage de la colonne TRIM (ajout du 'T' devant le numéro)
    df_trim_count['TRIM'] = 'T' + df_trim_count['TRIM'].astype(str)

    
    # Traitement des Hospitalisations (Source: df_daily)
    df_daily_clean = df_daily.copy()

    # Extraction année et mois (si ce n'est pas déjà fait)
    if not pd.api.types.is_datetime64_any_dtype(df_daily_clean['date']):
        df_daily_clean['date'] = pd.to_datetime(df_daily_clean['date'])

    df_daily_clean['year'] = df_daily_clean['date'].dt.year
    df_daily_clean['month'] = df_daily_clean['date'].dt.month

    # Sélection de l'année
    df_daily_year = df_daily_clean[df_daily_clean['year'] == year]

    # Calcul spécifique des trimestres
    t1 = df_daily_year[df_daily_year['month'] == 3]['hosp'].sum()
    t2 = df_daily_year[df_daily_year['month'].isin([4, 5, 6])]['hosp'].sum()
    t3 = df_daily_year[df_daily_year['month'].isin([7, 8, 9])]['hosp'].sum()
    t4 = df_daily_year[df_daily_year['month'].isin([10, 11, 12])]['hosp'].sum()

    # Construction des dataframes
    df_hosp_trim = pd.DataFrame({
        "TRIM": ["T1", "T2", "T3", "T4"],
        "hosp": [t1, t2, t3, t4]
    })


    # Fusion des données
    df_fusionne = pd.merge(
        df_hosp_trim,
        df_trim_count,
        on='TRIM',
        how='left'
    )


    # Graphique
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Hospitalisations (premier axe)
    ax1.set_xlabel('Trimestre')
    ax1.set_ylabel('Nombre d\'hospitalisations (en millions)', color='#248BFF', fontsize=12)
    ax1.plot(
        df_fusionne['TRIM'],
        df_fusionne['hosp'],
        color='#248BFF',
        marker='o',
        linewidth=2,
        label='Hospitalisations'
        )
    ax1.tick_params(axis='y', labelcolor='#248BFF')
    ax1.grid(True, alpha=0.3)

    # Arrêts maladie (second axe)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Arrêts maladie', color='#7A57FF', fontsize=12)
    ax2.plot(
        df_fusionne['TRIM'],
        df_fusionne['Effectifs'],
        color='#7A57FF',
        marker='^',
        linewidth=2,
        label='Arrêts maladie'
        )
    ax2.tick_params(axis='y', labelcolor='#7A57FF')

    plt.title(f'Hospitalisations liées au Covid-19 et arrêts maladie en France en {year}')
    
    # Résultat
    plt.show()

    # Corrélation obtenue
    correlation = df_fusionne[['hosp', 'Effectifs']].corr().iloc[0, 1]
    print('Coefficient de corrélation entre les deux variables :', round(correlation, 2))
