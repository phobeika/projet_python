import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

# Dictionnaire de recodage CSE → label texte

cse_map = {
    '0':  "Non renseigné",
    '10': "Agriculteurs",
    '11': "Agriculteurs petite exploitation",
    '12': "Agriculteurs moyenne exploitation",
    '13': "Agriculteurs grande exploitation",
    '21': "Artisans",
    '22': "Commerçants et assimilés",
    '23': "Chefs d'entreprise (10 salariés ou plus)",
    '31': "Professions libérales",
    '33': "Cadres de la fonction publique",
    '34': "Professeurs, professions scientifiques",
    '35': "Information, arts, spectacles",
    '37': "Cadres administratifs et commerciaux",
    '38': "Ingénieurs et cadres techniques",
    '42': "Instituteurs et assimilés",
    '43': "Professions santé & travail social",
    '44': "Clergé, religieux",
    '45': "Intermédiaires admin. fonction publique",
    '46': "Intermédiaires admin. & commerciaux",
    '47': "Techniciens",
    '48': "Contremaîtres, agents de maîtrise",
    '52': "Employés civils & agents de service FP",
    '53': "Policiers & militaires",
    '54': "Employés administratifs d'entreprise",
    '55': "Employés de commerce",
    '56': "Services directs aux particuliers",
    '62': "Ouvriers qualifiés type industriel",
    '63': "Ouvriers qualifiés type artisanal",
    '64': "Chauffeurs",
    '65': "Ouvriers manutention / magasinage / transport",
    '67': "Ouvriers non qualifiés type industriel",
    '68': "Ouvriers non qualifiés type artisanal",
    '69': "Ouvriers agricoles",
    '81': "Chômeurs n’ayant jamais travaillé"
}

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

nafr_map = {
    "": "Sans objet (inactifs occupés)",
    "A": "Agriculture, sylviculture et pêche",
    "B": "Industries extractives",
    "C": "Industrie manufacturière",
    "D": "Production et distribution d’électricité, de gaz, de vapeur et d’air conditionné",
    "E": "Production et distribution d’eau ; assainissement, gestion des déchets et dépollution",
    "F": "Construction",
    "G": "Commerce ; réparation d’automobiles et de motocycles",
    "H": "Transports et entreposage",
    "I": "Hébergement et restauration",
    "J": "Information et communication",
    "K": "Activités financières et d’assurance",
    "L": "Activités immobilières",
    "M": "Activités spécialisées, scientifiques et techniques",
    "N": "Activités de services administratifs et de soutien",
    "O": "Administration publique",
    "P": "Enseignement",
    "Q": "Santé humaine et action sociale",
    "R": "Arts, spectacles et activités récréatives",
    "S": "Autres activités de services",
    "T": "Activités des ménages en tant qu’employeurs ; activités indifférenciées des ménages",
    "U": "Activités extra-territoriales"
}


def exposition(df, var="CSE", year=None, year_col="ANNEE"):
    """
    Renvoie la proportion d'individus ayant déclaré un arrêt maladie (RABS == 2)
    selon les modalités de `var`.
    """

    df = df.copy()

    # --- 1. Vérification année ---
    # If year is provided, filter
    if year is not None and year_col in df.columns:
        df = df[df[year_col] == year]

    # --- 2. Variable NAF agrégée
    if var == "NAFR":
        df["NAFR"] = df["NAFG038UN"].str.upper().str[0]

    # --- 3. Comptages ---
    totaux = df.groupby(var).size().rename("effectif_total")

    rabs2 = (
        df[df["RABS"] == '2']
        .groupby(var)
        .size()
        .rename("effectif_rabs2")
    )

    # --- 4. Fusion + proportion ---
    df_prop = pd.concat([totaux, rabs2], axis=1).fillna(0)

    df_prop["proportion_rabs2"] = (
        df_prop["effectif_rabs2"] / df_prop["effectif_total"]*100
    ).round(2)

    # --- 5. Ajouter les labels si var == "CSE" ou "NAF" etc.---

    if var == "CSE":
        df_prop["label"] = df_prop.index.map(cse_map)
        df_prop = df_prop.reset_index().rename(columns={'index': var})[
            ['label', var, 'effectif_total', 'effectif_rabs2', 'proportion_rabs2']]

    if var == "NAFG038UN":
        df_prop["label"] = df_prop.index.map(naf_map)
        df_prop = df_prop.reset_index().rename(columns={'index': var})[
            ['label', var, 'effectif_total', 'effectif_rabs2', 'proportion_rabs2']]

    if var == "NAFR":
        df_prop["label"] = df_prop.index.map(nafr_map)
        df_prop = df_prop.reset_index().rename(columns={'index': var})[
            ['label', var, 'effectif_total', 'effectif_rabs2', 'proportion_rabs2']]

    if var == "SEXE":
        df_prop["label"] = df_prop.index.map(sexe_map)
        df_prop = df_prop.reset_index().rename(columns={'index': var})[
            ['label', var, 'effectif_total', 'effectif_rabs2', 'proportion_rabs2']]

    return df_prop.sort_values("proportion_rabs2", ascending=False)


def exposition_annee(df, var="SEXE", year_col="ANNEE", annee=None):
    """
    Proportion d'arrêts maladie (RABS == 2) par année et selon `var`
    """

    df = df.copy()

    if year_col not in df.columns:
        raise ValueError(f"La colonne '{year_col}' est absente du DataFrame.")

    if var == "NAFR":
        df["NAFR"] = df["NAFG038UN"].str.upper().str[0]

    # Filtrage sur l'année si demandé
    if annee is not None:
        if isinstance(annee, (list, tuple, set)):
            df = df[df[year_col].isin(annee)]
        else:
            df = df[df[year_col] == annee]

    totaux = (
        df.groupby([year_col, var])
        .size()
        .rename("effectif_total")
    )

    rabs2 = (
        df[df["RABS"] == '2']
        .groupby([year_col, var])
        .size()
        .rename("effectif_rabs2")
    )

    df_prop = (
        pd.concat([totaux, rabs2], axis=1)
        .fillna(0)
        .reset_index()
    )

    df_prop["proportion_rabs2"] = (
        df_prop["effectif_rabs2"] / df_prop["effectif_total"]
    )

    df_prop["proportion_pct"] = 100 * df_prop["proportion_rabs2"]

    return df_prop


def exposition_diff(df, var="CSE", year1=None, year2=None, year_col="ANNEE"):
    """
    Compute the difference in proportion of RABS == 2 between year2 and year1
    for each modality of var.

    Returns a DataFrame with columns: var, year1_prop, year2_prop, difference
    Sorted by difference descending.
    """
    if year1 is None or year2 is None:
        raise ValueError("Provide year1 and year2")

    # Get proportions for each year using exposition()
    df1 = exposition(df, var=var, year=year1)
    df2 = exposition(df, var=var, year=year2)

    # Ensure var is a column
    df1 = df1.reset_index().rename(columns={'index': var}) if var not in df1.columns else df1
    df2 = df2.reset_index().rename(columns={'index': var}) if var not in df2.columns else df2

    # Add labels if available
    label_maps = {"CSE": cse_map, "SEXE": sexe_map, "NAFG038UN": naf_map}
    if var in label_maps:
        df1['label'] = df1[var].map(label_maps[var])
        df2['label'] = df2[var].map(label_maps[var])

    # Rename proportion columns
    df1 = df1.rename(columns={'proportion_rabs2': f'prop_{year1}'})
    df2 = df2.rename(columns={'proportion_rabs2': f'prop_{year2}'})

    # Merge on var
    merged = pd.merge(
        df1[[var, f'prop_{year1}']],
        df2[[var, f'prop_{year2}']],
        on=var,
        how='outer'
        ).fillna(0)

    # Compute difference
    merged['difference'] = merged[f'prop_{year2}'] - merged[f'prop_{year1}']

    # Add labels if present
    if 'label' in df1.columns:
        labels = df1.set_index(var)['label']
        merged = merged.set_index(var).join(labels).reset_index()
        merged = merged[['label', var, f'prop_{year1}', f'prop_{year2}', 'difference']]

    else:
        merged = merged[[var, f'prop_{year1}', f'prop_{year2}', 'difference']]

    return merged.sort_values("difference", ascending=False)


def score_exposition(df, var_list, year1=2019, year2=2020, year_col="ANNEE"):
    """
    Calcule un score d'exposition cumulée.

    Pour chaque variable de var_list :
    - calcule la différence d'exposition (RABS==2) entre year2 et year1
    - classe les modalités en terciles d'exposition
    - attribue un score 0 (faible), 1 (moyen), 2 (élevé)

    Le score final est la somme des scores par variable.
    """

    df = df.copy()
    score_cols = []

    for var in var_list:
        # --- 1. Différences d'exposition par modalité ---
        expo = exposition_diff(
            df,
            var=var,
            year1=year1,
            year2=year2,
            year_col=year_col
        )

        # On récupère la variable + la différence
        expo = expo[[var, "difference"]].dropna()

        # --- 2. Classement en terciles ---
        expo = expo.sort_values("difference")
        n = len(expo)

        expo["score"] = 0
        expo.iloc[n//3:2*n//3, expo.columns.get_loc("score")] = 1
        expo.iloc[2*n//3:, expo.columns.get_loc("score")] = 2

        # --- 3. Mapping modalité → score ---
        score_map = expo.set_index(var)["score"]

        # --- 4. Attribution aux individus ---
        col_score = f"score_{var}"
        df[col_score] = df[var].map(score_map)

        score_cols.append(col_score)

    # --- 5. Score total ---
    df["score_exposition"] = df[score_cols].sum(axis=1)

    return df


def plot_evolution_proportion(df, year_col, group_col, value_col, title, ylabel, colors=None):
    """
    Cette fonction trace l'évolution d'une variable quantitative en fonction du temps,
    avec une courbe par modalité d'une variable de groupe.

    Arguments:
        df (pandas.DataFrame) : données sources
        year_col (str) : colonne représentant le temps (ex : année)
        group_col (str) : colonne de regroupement (ex: sexe)
        value_col (str) : variable à représenter (ex: proportion)
        title (str) : titre du graphique
        ylabel (str) : label de l'axe Y
    """

    # Pivot pour éviter les doublons d'années
    df_pivot = (
        df
        .pivot(index=year_col, columns=group_col, values=value_col)
        .sort_index()
    )

    plt.figure(figsize=(10, 6))

    for col in df_pivot.columns:
        plt.plot(
            df_pivot.index,
            df_pivot[col],
            marker="o",
            label=str(col),
            color=colors.get(col) if colors else None
        )

    plt.xlabel(year_col)
    plt.ylabel(ylabel if ylabel else value_col)
    plt.title(title if title else f"Évolution de {value_col} selon {group_col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_score_exposition(df, score_col="score_exposition", by=None, labels=None, title=None):

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
        },
        "NAFR": {
            "": "Sans objet (inactifs occupés)",
            "A": "Agriculture, sylviculture et pêche",
            "B": "Industries extractives",
            "C": "Industrie manufacturière",
            "D": "Production et distribution d’électricité, de gaz, de vapeur et d’air conditionné",
            "E": "Production et distribution d’eau ; assainissement, "
            "gestion des déchets et dépollution",
            "F": "Construction",
            "G": "Commerce ; réparation d’automobiles et de motocycles",
            "H": "Transports et entreposage",
            "I": "Hébergement et restauration",
            "J": "Information et communication",
            "K": "Activités financières et d’assurance",
            "L": "Activités immobilières",
            "M": "Activités spécialisées, scientifiques et techniques",
            "N": "Activités de services administratifs et de soutien",
            "O": "Administration publique",
            "P": "Enseignement",
            "Q": "Santé humaine et action sociale",
            "R": "Arts, spectacles et activités récréatives",
            "S": "Autres activités de services",
            "T": "Activités des ménages en tant qu’employeurs ;"
            " activités indifférenciées des ménages",
            "U": "Activités extra-territoriales"
        }
    }

    # ===========================
    # CAS 1 — graphique simple
    # ===========================
    if by is None:
        effectifs = df[score_col].value_counts().sort_index()
        frequences = effectifs / effectifs.sum() * 100

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            effectifs.index,
            effectifs.values, 
            edgecolor="#7A57FF", 
            color="#7A57FF"
            )

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

    # 🔹 mémoriser l’ordre initial AVANT renommage
    order_initial = df_plot[by].dropna().unique()

    # Choix des labels
    if labels is not None:
        label_map = labels
    elif by in default_labels:
        label_map = default_labels[by]
    else:
        label_map = None

    if label_map is not None:
        df_plot[by] = df_plot[by].map(label_map)

        # 🔹 reconstruire l’ordre APRÈS renommage
        ordered_labels = [label_map[x] for x in order_initial if x in label_map]
        df_plot[by] = pd.Categorical(
            df_plot[by],
            categories=ordered_labels,
            ordered=True
        )

    table = pd.crosstab(df_plot[score_col], df_plot[by])

    plt.figure(figsize=(10, 6))
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


def plot_freq_exposition(df, score_col="score_exposition", by="SEXE", labels=None, title=None ):
    """
    Graphique en barres empilées à 100 % :
    répartition de `by` pour chaque valeur du score d'exposition.
    """

    df = df.copy()

    # -----------------------
    # Configuration interne
    # -----------------------
    modality_config = {
        "CSER_label": {
            "order": [
                "Agriculteurs exploitants",
                "Cadres et professions intellectuelles supérieures",
                "Artisans, commerçants et chefs d'entreprise",
                "Ouvriers",
                "Professions intermédiaires",
                "Employés"
            ],
            "colors": [
                "#EDC948",
                "#E15759",
                "#59A14F",
                "#F28E2B",

                "#B07AA1",
                "#4E79A7"
            ]
        },
        "SEXE_label": {
            "order": ["Femme", "Homme"],
            "colors": ["#CABCFF", "#4E79A7"]  # ou ["#E15759","#4E79A7"]
        },
        "PUB_label": {
            "order": [
                "Hôpitaux publics", "État", "Collectivités locales", "Secteur privé"
            ],
            "colors": [
                "#A7D1FF",
                "#E9C3FF",
                "#CABCFF",

                "#D3D3D3"
            ]
        }
    }

    # -----------------------
    # Recodage éventuel
    # -----------------------
    if labels is not None:
        df[by] = df[by].map(labels)

    # -----------------------
    # Table de contingence
    # -----------------------
    table = pd.crosstab(df[score_col], df[by])

    # Ordre des modalités si défini
    if by in modality_config:
        order = modality_config[by]["order"]
        table = table.reindex(columns=order, fill_value=0)
        colors = modality_config[by]["colors"]
    else:
        colors = None

    # Normalisation en base 100 %
    table_pct = table.div(table.sum(axis=1), axis=0) * 100

    # -----------------------
    # Graphique
    # -----------------------
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(table_pct))

    for i, col in enumerate(table_pct.columns):
        plt.bar(
            table_pct.index,
            table_pct[col],
            bottom=bottom,
            label=col,
            edgecolor="black",
            color=None if colors is None else colors[i]
        )
        bottom += table_pct[col].values

    # Ajouter les pourcentages
    for x, row in enumerate(table_pct.values):
        cumul = 0
        for val in row:
            if val > 5:
                plt.text(
                    table_pct.index[x],
                    cumul + val / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                )
            cumul += val

    plt.xlabel("Score d'exposition")
    plt.ylabel("Répartition (%)")
    plt.title(title or f"Répartition de {by} selon le score d'exposition (base 100 %)")
    plt.legend(title=by, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


def regression_exposition(df, target="score_exposition", categorical_vars=None, reference_dict=None):
    """
    Effectue une régression linéaire OLS avec covariance robuste (HC3) sur un dataframe.
    Retourne un DataFrame avec coefficients, intervalles de confiance et p-values,
    avec des noms de variables simplifiés pour les qualitatives.
    """
    # On conserve seulement les colonnes nécessaires et on supprime les NA
    cols = [target] + (categorical_vars if categorical_vars else [])
    df_clean = df[cols].dropna()

    # Construction de la formule
    formula_terms = []
    for var in categorical_vars:
        if reference_dict and var in reference_dict:
            formula_terms.append(f'C({var}, Treatment(reference="{reference_dict[var]}"))')
        else:
            formula_terms.append(f'C({var})')
    formula = f"{target} ~ " + " + ".join(formula_terms)

    # Ajustement du modèle OLS
    modele = smf.ols(formula=formula, data=df_clean).fit()
    modele_robust = modele.get_robustcov_results(cov_type="HC3")

    # Extraction des coefficients et statistiques
    coef = modele_robust.params
    conf = modele_robust.conf_int()
    pval = modele_robust.pvalues
    names = modele_robust.model.exog_names  # noms des variables

    # Création du DataFrame
    df_coef = pd.DataFrame({
        "variable": names,
        "coef": coef,
        "ci_low": conf[:, 0],
        "ci_high": conf[:, 1],
        "p_value": pval
    })

    # Nettoyage des noms pour les variables qualitatives
    cleaned_names = []
    for name in df_coef['variable']:
        # Si c'est une variable catégorielle codée par patsy, elle contient '[T.level]'
        if "[T." in name:
            level = name.split("[T.")[1].rstrip("]")
            cleaned_names.append(level)
        elif name == "Intercept":
            cleaned_names.append("Intercept")
        else:
            cleaned_names.append(name)
    df_coef['variable'] = cleaned_names

    return df_coef, modele_robust


def plot_regression_exposition(df_coef):
    """
    Trace les coefficients d'une régression avec intervalles de confiance.

    Parameters:
    -----------
    df_coef : pd.DataFrame
        DataFrame contenant 'coef', 'ci_low', 'ci_high' et 'variable'.
    title : str
        Titre du graphique.
    """
    plt.figure(figsize=(10, 3))
    y_pos = np.arange(len(df_coef))

    plt.errorbar(
        df_coef["coef"],
        y_pos,
        xerr=[df_coef["coef"] - df_coef["ci_low"], df_coef["ci_high"] - df_coef["coef"]],
        fmt='o',
        color="#7A57FF",
        ecolor="#7A57FF",
        capsize=4
    )
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.yticks(y_pos, df_coef["variable"])
    plt.xlabel("Effet marginal sur le score d'exposition")
    plt.title("Représentation graphique des coefficients de la régression du score d'exposition")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def regression_exposition2(df, target="score_exposition", categorical_vars=None, reference_dict=None):
    """
    Régression logistique ordinale sur un score discret (0-4) avec covariables qualitatives.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    target : str
        Variable dépendante ordinale.
    categorical_vars : list of str
        Liste de variables qualitatives à inclure.
    reference_dict : dict
        Dictionnaire de références pour chaque variable qualitative.

    Returns
    -------
    df_coef : pd.DataFrame
        Coefficients avec noms simplifiés.
    res : statsmodels OrderedModelResults
        Résultat du modèle pour analyses supplémentaires.
    """
    # Nettoyage des colonnes et NA
    cols = [target] + (categorical_vars if categorical_vars else [])
    df_clean = df[cols].dropna().copy()

    # Convertir le score en int (obligatoire pour OrderedModel)
    df_clean[target] = df_clean[target].astype(int)

    # Normaliser les colonnes (remplacer espaces/accents pour get_dummies)
    for var in categorical_vars:
        df_clean[var] = df_clean[var].astype(str).str.replace(" ", "_")

    # Création des dummies
    df_dummies = pd.get_dummies(df_clean, columns=categorical_vars, drop_first=False)

    # Supprimer les colonnes de référence
    if reference_dict:
        for var, ref in reference_dict.items():
            ref_col = f"{var}_{ref.replace(' ', '_')}"
            if ref_col in df_dummies.columns:
                df_dummies = df_dummies.drop(columns=[ref_col])

    X = df_dummies.drop(columns=[target])
    y = df_dummies[target]

    # Ajustement du modèle ordinal
    mod = OrderedModel(y, X, distr='logit')
    res = mod.fit(method='bfgs', disp=False)

    # Création du DataFrame des coefficients
    df_coef = pd.DataFrame({
        'variable': X.columns.str.replace("_", ": ", regex=False),
        'coef': res.params.values[:X.shape[1]]
    })

    return df_coef, res


def plot_regression_exposition2(df_coef, title="Effets sur le score d'exposition"):
    """
    Trace les coefficients d'un modèle ordinal.
    """
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(df_coef))
    plt.barh(y_pos, df_coef['coef'], color="#7A57FF")
    plt.yticks(y_pos, df_coef['variable'])
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel("Effet marginal log-odds")
    plt.title(title)
    plt.tight_layout()
    plt.show()
