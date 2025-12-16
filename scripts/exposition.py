import pandas as pd
import matplotlib.pyplot as plt

# Dictionnaire de recodage CSE → label texte

cse_map = {
    0:  "Non renseigné",
    10: "Agriculteurs",
    11: "Agriculteurs petite exploitation",
    12: "Agriculteurs moyenne exploitation",
    13: "Agriculteurs grande exploitation",
    21: "Artisans",
    22: "Commerçants et assimilés",
    23: "Chefs d'entreprise (10 salariés ou plus)",
    31: "Professions libérales",
    33: "Cadres de la fonction publique",
    34: "Professeurs, professions scientifiques",
    35: "Information, arts, spectacles",
    37: "Cadres administratifs et commerciaux",
    38: "Ingénieurs et cadres techniques",
    42: "Instituteurs et assimilés",
    43: "Professions santé & travail social",
    44: "Clergé, religieux",
    45: "Intermédiaires admin. fonction publique",
    46: "Intermédiaires admin. & commerciaux",
    47: "Techniciens",
    48: "Contremaîtres, agents de maîtrise",
    52: "Employés civils & agents de service FP",
    53: "Policiers & militaires",
    54: "Employés administratifs d'entreprise",
    55: "Employés de commerce",
    56: "Services directs aux particuliers",
    62: "Ouvriers qualifiés type industriel",
    63: "Ouvriers qualifiés type artisanal",
    64: "Chauffeurs",
    65: "Ouvriers manutention / magasinage / transport",
    67: "Ouvriers non qualifiés type industriel",
    68: "Ouvriers non qualifiés type artisanal",
    69: "Ouvriers agricoles",
    81: "Chômeurs n’ayant jamais travaillé"
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

    # --- 2. Recodage CSE ---
    if var == "CSE":
        df["CSE_int"] = df["CSE"].astype("Int64")
        df["CSE_label"] = df["CSE_int"].map(cse_map)
        var = "CSE_label"

    # --- 3. Comptages ---
    totaux = df.groupby(var).size().rename("effectif_total")

    rabs2 = (
        df[df["RABS"] == 2]
        .groupby(var)
        .size()
        .rename("effectif_rabs2")
    )

    # --- 4. Fusion + proportion ---
    df_prop = pd.concat([totaux, rabs2], axis=1).fillna(0)

    df_prop["proportion_rabs2"] = (
        df_prop["effectif_rabs2"] / df_prop["effectif_total"]
    )

    return df_prop.sort_values("proportion_rabs2", ascending=False)


def exposition_annee(df, var="SEXE", year_col="ANNEE"):
    """
    Proportion d'arrêts maladie (RABS == 2) par année et selon `var`
    """

    df = df.copy()

    if year_col not in df.columns:
        raise ValueError(f"La colonne '{year_col}' est absente du DataFrame.")

    totaux = (
        df.groupby([year_col, var])
        .size()
        .rename("effectif_total")
    )

    rabs2 = (
        df[df["RABS"] == 2]
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
