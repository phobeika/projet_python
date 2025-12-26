import pandas as pd
import requests
from dbfread import DBF
from io import BytesIO
from zipfile import ZipFile
import tempfile
import os


def read_from_zip(url, backup_url=None, filename_keyword=None, **kwargs):
    """
    Télécharge un ZIP ou un fichier direct depuis une URL, lit un fichier CSV
    ou DBF à l'intérieur et l'importe sous la forme d'un dataframe Pandas

    Paramètres :
    - url : URL du fichier (ZIP ou direct CSV/DBF)
    - backup_url : URL de secours à utiliser en cas d'échec
    - filename_keyword : mot-clé pour filtrer le CSV à ouvrir (facultatif, pour ZIP)
    - kwargs : paramètres additionnels passés à pd.read_csv

    Retour :
    - DataFrame pandas
    """

    def try_read(url):
        """Sous-fonction : essaye de lire un fichier depuis une URL donnée"""
        response = requests.get(url)
        response.raise_for_status()  # lèvera une erreur si le téléchargement échoue

        if url.endswith('.zip'):
            # Traitement pour ZIP
            zip_bytes = BytesIO(response.content)
            with ZipFile(zip_bytes) as myzip:
                files = [f for f in myzip.namelist() if f.endswith(('.csv', '.dbf'))]
                if len(files) == 0:
                    raise ValueError(f"Aucun fichier CSV ou DBF trouvé dans le ZIP à {url}")

                # Si mot-clé fourni
                if filename_keyword:
                    files = [f for f in files if filename_keyword in f]
                    if len(files) == 0:
                        raise ValueError(f"Aucun CSV contenant '{filename_keyword}' trouvé à {url}")
                    elif len(files) > 1:
                        raise ValueError(f"Plusieurs CSV contiennent \
                            '{filename_keyword}'à {url}: {files}")

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

                # ---- Lecture DBF ----
                if selected.endswith(".dbf"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dbf") as tmp:
                        tmp.write(myzip.read(selected))
                        tmp_path = tmp.name

                    try:
                        table = DBF(tmp_path, ignore_missing_memofile=True)
                        df = pd.DataFrame(iter(table))
                        return df
                    finally:
                        os.remove(tmp_path)
        else:
            # Traitement pour fichier direct
            content = BytesIO(response.content)
            if url.endswith('.csv'):
                return pd.read_csv(content, **kwargs)
            elif url.endswith('.dbf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dbf") as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

                try:
                    table = DBF(tmp_path, ignore_missing_memofile=True)
                    df = pd.DataFrame(iter(table))
                    return df
                finally:
                    os.remove(tmp_path)
            else:
                raise ValueError(f"Type de fichier non supporté pour {url}.")

    # ---- Tentative principale ----
    try:
        print(f"Téléchargement depuis l'URL principale : {url}")
        return try_read(url)

    # ---- Tentative backup ----
    except Exception as e:
        if backup_url:
            print(f"⚠️ Échec avec URL principale ({e}). Tentative avec {backup_url}")
            try:
                return try_read(backup_url)
            except Exception as e2:
                raise RuntimeError(
                    f"Échec avec les deux URLs.\nErreur principale : {e}\nErreur backup : {e2}"
                )
        else:
            raise RuntimeError(f"Erreur : {e}")


def convert_to_categorical(df, var):
    df[var] = pd.to_numeric(df[var], errors='coerce').astype('Int64')
    df[var] = df[var].astype('category')
    return df


def recodages(df, vars_cat, vars_num):
    # Normaliser les colonnes avec des problèmes de type mixte (ex: "2.0" vs 2.0)
    # On convertit d'abord en numérique pour nettoyer, puis en catégories si nécessaire
    for col in vars_cat:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64').astype(str)

    for col in vars_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)

    return df


sexe_map = {
    "1": "Homme",
    "2": "Femme"
}

cse_map = {
    "0":  "Non renseigné",
    "10": "Agriculteurs",
    "11": "Agriculteurs petite exploitation",
    "12": "Agriculteurs moyenne exploitation",
    "13": "Agriculteurs grande exploitation",
    "21": "Artisans",
    "22": "Commerçants et assimilés",
    "23": "Chefs d'entreprise (10 salariés ou plus)",
    "31": "Professions libérales",
    "33": "Cadres de la fonction publique",
    "34": "Professeurs, professions scientifiques",
    "35": "Information, arts, spectacles",
    "37": "Cadres administratifs et commerciaux",
    "38": "Ingénieurs et cadres techniques",
    "42": "Instituteurs et assimilés",
    "43": "Professions santé & travail social",
    "44": "Clergé, religieux",
    "45": "Intermédiaires admin. fonction publique",
    "46": "Intermédiaires admin. & commerciaux",
    "47": "Techniciens",
    "48": "Contremaîtres, agents de maîtrise",
    "52": "Employés civils & agents de service FP",
    "53": "Policiers & militaires",
    "54": "Employés administratifs d'entreprise",
    "55": "Employés de commerce",
    "56": "Services directs aux particuliers",
    "62": "Ouvriers qualifiés type industriel",
    "63": "Ouvriers qualifiés type artisanal",
    "64": "Chauffeurs",
    "65": "Ouvriers manutention / magasinage / transport",
    "67": "Ouvriers non qualifiés type industriel",
    "68": "Ouvriers non qualifiés type artisanal",
    "69": "Ouvriers agricoles",
    "81": "Chômeurs n’ayant jamais travaillé"
}


naf_map = {
    "": "Sans objet (inactifs occupés)",
    "00": "Non renseigné",
    "AZ": "Agriculture, sylviculture et pêche",
    "BZ": "Industries extractives",
    "CA": "Alimentaire, boissons, tabac",
    "CB": "Textile, habillement, cuir, chaussure",
    "CC": "Bois, papier, imprimerie",
    "CD": "Cokéfaction et raffinage",
    "CE": "Industrie chimique",
    "CF": "Industrie pharmaceutique",
    "CG": "Caoutchouc, plastique, minéraux non métalliques",
    "CH": "Métallurgie et produits métalliques",
    "CI": "Informatique, électronique, optique",
    "CJ": "Équipements électriques",
    "CK": "Machines et équipements",
    "CL": "Matériels de transport",
    "CM": "Autres industries manufacturières; réparation & installation",
    "DZ": "Électricité, gaz, vapeur, air conditionné",
    "EZ": "Eau, déchets, dépollution",
    "FZ": "Construction",
    "GZ": "Commerce, réparation auto/moto",
    "HZ": "Transports et entreposage",
    "IZ": "Hébergement et restauration",
    "JA": "Édition, audiovisuel, diffusion",
    "JB": "Télécommunications",
    "JC": "Informatique & services d'information",
    "KZ": "Activités financières et d’assurance",
    "LZ": "Activités immobilières",
    "MA": "Juridique, comptable, gestion, architecture, ingénierie",
    "MB": "Recherche-développement scientifique",
    "MC": "Autres activités spécialisées, scientifiques & techniques",
    "NZ": "Services administratifs & soutien",
    "OZ": "Administration publique",
    "PZ": "Enseignement",
    "QA": "Activités pour la santé humaine",
    "QB": "Hébergement médico-social & action sociale",
    "RZ": "Arts, spectacles, activités récréatives",
    "SZ": "Autres activités de services",
    "TZ": "Activités des ménages employeurs & production pour usage propre",
    "UZ": "Activités extra-territoriales"
}

pub_map = {
    "1": "État",
    "2": "Collectivités locales",
    "3": "Hôpitaux publics",
    "4": "Secteur privé"
}


def add_labels(df):
    df = df.copy()
    # --------------------
    # SEXE
    # --------------------
    if "SEXE" in df.columns:
        df["SEXE_label"] = df["SEXE"].map(sexe_map)
    # PUB3FP
    # --------------------
    if "PUB3FP" in df.columns:
        df["PUB_label"] = df["PUB3FP"].map(pub_map)
    # --------------------
    # CSE + CSE_label
    #     # --------------------
    if "CSE" in df.columns:
        df["CSE_label"] = df["CSE"].map(cse_map)
        # --------------------
        # AJOUT : CSER
        # --------------------
        # On travaille sur une version nettoyée uniquement pour CSER
        cse_clean = pd.to_numeric(df["CSE"], errors="coerce").astype("Int64")
        cse_first_digit = cse_clean.astype(str).str[0]
        cser_map = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "8": 8
        }
        df["CSER"] = cse_first_digit.map(cser_map)
        cser_labels = {
            1: "Agriculteurs exploitants",
            2: "Artisans, commerçants et chefs d'entreprise",
            3: "Cadres et professions intellectuelles supérieures",
            4: "Professions intermédiaires",
            5: "Employés",
            6: "Ouvriers",
            8: "Chômeurs n'ayant jamais travaillé"
        }
        df["CSER_label"] = df["CSER"].map(cser_labels)
    # --------------------
    # NAF
    # --------------------
    if "NAFG038UN" in df.columns:
        df["NAF_label"] = df["NAFG038UN"].map(naf_map)
    return df
