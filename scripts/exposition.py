import pandas as pd
import matplotlib.pyplot as plt

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



def exposition(df, var="CSE", year=None):
    """
    Renvoie la proportion d'individus ayant déclaré un arrêt maladie (RABS == 2)
    selon les modalités de `var`.
    """

    df = df.copy()

    # --- 1. Vérification année ---
    if year is None:
        raise ValueError("Merci d'indiquer une année")
    
    # Filtrage éventuel si la colonne existe
    if "year" in df.columns:
        df = df[df["year"] == year]

    # --- 2. Recodage CSE --- pas besoin, c'est fait avant
    # if var == "CSE":
    #    df["CSE_int"] = df["CSE"].astype("Int64")
    #    df["CSE_label"] = df["CSE_int"].map(cse_map)
    #    var = "CSE_label"

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

    # --- 5. Ajouter les labels si var == "CSE" ou "NAF"---

    if var == "CSE":
        df_prop["label"] = df_prop.index.map(cse_map)
        df_prop = df_prop[['label', 'effectif_total', 'effectif_rabs2', 'proportion_rabs2']]

    if var == "XXX":
        df_prop["label"] = df_prop.index.map(naf_map)
    
    return df_prop.sort_values("proportion_rabs2", ascending=False)


def exposition_annee(df, var="SEXE", year_col="ANNEE", annee=None):
    """
    Proportion d'arrêts maladie (RABS == 2) par année et selon `var`
    """

    df = df.copy()

    if year_col not in df.columns:
        raise ValueError(f"La colonne '{year_col}' est absente du DataFrame.")


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


def plot_evolution_proportion(
    df,
    year_col,
    group_col,
    value_col,
    title=None,
    ylabel=None,
    figsize=(10, 6),
    marker="o",
    colors=None
):

    """
    Trace l'évolution d'une variable quantitative en fonction du temps,
    avec une courbe par modalité d'une variable de groupe.

    Parameters
    ----------
    df : pandas.DataFrame
        Données sources
    year_col : str
        Colonne représentant le temps (ex: année)
    group_col : str
        Colonne de regroupement (ex: sexe)
    value_col : str
        Variable à représenter (ex: proportion)
    title : str, optional
        Titre du graphique
    ylabel : str, optional
        Label de l'axe Y
    figsize : tuple, optional
        Taille de la figure
    marker : str, optional
        Marqueur des points
    """

    # Pivot pour éviter les doublons d'années
    df_pivot = (
        df
        .pivot(index=year_col, columns=group_col, values=value_col)
        .sort_index()
    )

    plt.figure(figsize=figsize)

    for col in df_pivot.columns:
        plt.plot(
            df_pivot.index,
            df_pivot[col],
            marker=marker,
            label=str(col),
            color=colors.get(col) if colors else None
        )

    plt.xlabel(year_col)
    plt.ylabel(ylabel if ylabel else value_col)
    plt.title(title if title else f"Évolution de {value_col} selon {group_col}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_score_exposition(
    df,
    score_col="score_exposition",
    by=None,
    labels=None,
    figsize=(8,5),
    title=None
):

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
