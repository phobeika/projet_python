# import_eec_all.py
import pandas as pd
from scripts import import_eec

def load_eec_all(years=None, core_vars=None, use_backup=False):
    """
    Télécharge et concatène les fichiers EEC 2010-2024 depuis l'INSEE (avec backup optionnel)
    et retourne un DataFrame pandas contenant uniquement les colonnes demandées (core_vars), ou
    par défaut l'ensemble des colonnes communes à toutes les années.

    Paramètres :
    - years : liste ou range des années à importer (entre 2010 et 2024). Si None, toutes les années 2010-2024 sont importées.
    - core_vars : liste de colonnes à conserver. Si None, on utilisera toutes les colonnes communes.
    - use_backup : bool, si True, utilise les URLs de secours comme principales (utile si les principales sont indisponibles).
    
    Retour :
    - pandas DataFrame
    """

    # Si aucun argument `years` n'a été indiqué
    if years is None:
        years = range(2010, 2025)

    # Filtre pour s'assurer que les années indiquées sont bien entre 2010 et 2024
    years = [y for y in years if 2010 <= y <= 2024]

    # URLs principales
    urls = {
        2010: "https://www.insee.fr/fr/statistiques/fichier/2415256/eec10_dbase.zip",
        2011: "https://www.insee.fr/fr/statistiques/fichier/2415227/eec11_dbase.zip",
        2012: "https://www.insee.fr/fr/statistiques/fichier/2415221/eec12_dbase.zip",
        2013: "https://www.insee.fr/fr/statistiques/fichier/2414892/eec13_eec13_dbase.zip",
        2014: "https://www.insee.fr/fr/statistiques/fichier/1406342/eec14_eec14_dbase.zip",
        2015: "https://www.insee.fr/fr/statistiques/fichier/2388681/eec15_eec15_dbase.zip",
        2016: "https://www.insee.fr/fr/statistiques/fichier/2892163/fd_eec16_dbase.zip",
        2017: "https://www.insee.fr/fr/statistiques/fichier/3555153/fd_eec17_dbase.zip",
        2018: "https://www.insee.fr/fr/statistiques/fichier/4191029/fd_eec18_csv.zip",
        2019: "https://www.insee.fr/fr/statistiques/fichier/4809583/fd_eec19_csv.zip",
        2020: "https://www.insee.fr/fr/statistiques/fichier/5393560/fd_eec20_csv.zip",
        2021: "https://www.insee.fr/fr/statistiques/fichier/6654604/FD_csv_EEC21.zip",
        2022: "https://www.insee.fr/fr/statistiques/fichier/7657353/FD_csv_EEC22.zip",
        2023: "https://www.insee.fr/fr/statistiques/fichier/8241122/FD_csv_EEC23.zip",
        2024: "https://www.insee.fr/fr/statistiques/fichier/8632441/FD_csv_EEC_2024.zip"
    }

    # URLs backup
    backups = {
        2010: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2010.dbf",
        2011: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2011.dbf",
        2012: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2012.dbf",
        2013: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2013.dbf",
        2014: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2014.dbf",
        2015: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2015.dbf",
        2016: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2016.dbf",
        2017: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2017.dbf",
        2018: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_csv_EEC18.csv",
        2019: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2019.csv",
        2020: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_csv_EEC20.csv",
        2021: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2021.csv",
        2022: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2022.csv",
        2023: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_csv_EEC23.csv",
        2024: "https://minio.lab.sspcloud.fr/phobeika/open_eec/FD_EEC_2024.csv"
    }

    if use_backup:
        urls, backups = backups, urls

    dfs = []
    for year in years:
        sep = ";" if year >= 2018 else None  # CSV plus récent utilise ;
        df = import_eec.read_from_zip(urls[year], backup_url=backups[year], sep=sep)
        

        # Renommage de variables pour harmoniser les noms entre années
        if year in [2010, 2011, 2012]:
            df = df.rename(columns={'NAFG38UN': 'NAFG038UN'})
        if year in [2021, 2022]:
            df = df.rename(columns={'PCS2': 'CSE'})
        if year in [2023, 2024]:    
            df = df.rename(columns={'APCS2': 'CSE'})
        
        # Vérification des colonnes CSE et NAF
        # cse_cols = [c for c in df.columns if 'CS' in c.upper()]
        # naf_cols = [c for c in df.columns if 'NAFG038UN' in c.upper()]
        # print(f"Year {year}: CS columns: {cse_cols}")
        # print(f"Year {year}: NAFG038UN columns: {naf_cols}")
        
        print(df.columns.tolist())

        dfs.append(df)

    # Colonnes communes si core_vars non fourni
    if core_vars is None:
        vars_communes = set(dfs[0].columns)
        for df in dfs[1:]:
            vars_communes &= set(df.columns)
        core_vars = sorted(vars_communes)

    # Concaténation
    eec_all = pd.concat([df[core_vars] for df in dfs], ignore_index=True)

    return eec_all
