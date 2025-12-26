import pandas as pd
from linearmodels import PanelOLS
import matplotlib.pyplot as plt


# Préparation des données
def data_prep(df, min_obs):
    """
    Cette fonction filtre les groupes ayant suffisamment d'observations sur la période et agrège les données au niveau Groupe x Année.

    Arguments :
        df : le dataframe source contenant les données individuelles ;
        min_obs : le nombre minimum d'observations requis par année pour conserver un groupe ;

    Outputs
        df_agg : le dataframe agrégé
    """
    
    ## Sélection des CSP x NAF avec au moins vingt actifs pour toutes les années de la période
    df = df.groupby("groupe").filter(
        lambda x: (x.groupby('ANNEE').size() >= min_obs).all()
    )

    ## Agrégation par CSP x NAF
    df = df.groupby(["groupe", 'ANNEE']).agg(
        sh_arret=('is_RABS_2', 'mean'),
        n_individus=('is_RABS_2', 'size'),
        cat_expose=('score_exposition', 'mean'),
    ).reset_index()

    ## Évaluation des groupes obtenus
    n_groupes = df["groupe"].nunique()
    print(f"Nombre de combinaisons {"groupe"} conservées : {n_groupes}")
    
    return df


# Estimation de la double-différences statistique
def estimation_statique(df, traitement):
    """
    Cette fonction estime un modèle de Différence-in-Différences (DiD) canonique avec 
    des effets fixes par groupe et année (Two-Way Fixed Effects).

    Arguments
        - df : le dataframe contenant les données au format long ;
        - traitement : le nom de la variable de traitement (variable indicatrice).

    Output: tableau de résultat
    """

    ## Définition de l'index pour les données de panel dans linearmodels
    df_reg = df.set_index(['groupe', 'ANNEE'])

    ## Définition de la variable expliquée Y et de la variable dépendante X
    Y = df_reg['sh_arret']
    X = df_reg[[traitement]]
    
    # Configuration du modèle
    modele = PanelOLS(
        Y, 
        X, 
        entity_effects=True,  # Effets fixes Groupe
        time_effects=True,    # Effets fixes Temps
        drop_absorbed=True    # Pour éviter les erreurs de colinéarité
    )

    # Estimation avec erreurs-types clusterisées (Robustes à l'autocorrélation intra-groupe)
    resultat = modele.fit(cov_type='clustered', cluster_entity=True)

    ## Résultat
    print(resultat.summary)
        
    return resultat


# Estimation de la double-différences dynamique
def estimation_dynamique(df, interaction_cols):
    """
    Cette fonction estime le modèle d'étude d'événement.
    
    Arguments :
        - df : les données sur lesquelles estimer le modèle ;
        - interaction_cols : les noms de colonnes d'interaction (leads et lags).

    Output : la table de résultats.
    """

    # Préparation de l'index (sur une copie pour sécurité)
    df_reg = df.set_index(["groupe", "ANNEE"])
    
    ## Définition de la variable expliquée Y et des variables d'interaction X
    Y = df_reg["sh_arret"]
    X = df_reg[interaction_cols]

    ## Configuration du modèle
    modele = PanelOLS(
        Y,                      # Variable expliquée: part d'arrêts maladie Y_it
        X,                      # Variable explicatives: 1{t == j} x 1{exposé == 1}
        entity_effects=True,    # Effets-fixes groupes: gamma_i (capte les différences de niveau entre les groupes)
        time_effects=True,      # Effets-fixes années: delta_t (capte l'évolution commune à tous les groupes)
        drop_absorbed=True
    )

    ## Estimation avec erreurs-types clusterisées (Robustes à l'autocorrélation intra-groupe)
    resultat = modele.fit(cov_type='clustered', cluster_entity=True)

    ## Résultat
    print(resultat.summary)

    return resultat


# Représentation graphique de l'étude d'évènement
def plot_event_study(resultat):
    """
    Trace le graphique des coefficients de l'étude d'évènement à partir du résultat de estimation_dynamique
    """

    ## Récupération des coefficients estimés et des intervalles de confiance
    est = resultat.params
    ic = resultat.conf_int()
    
    ## Enregistrement dans un dataframe
    df_plot = pd.DataFrame({
        'coef': est,
        'lower': ic.iloc[:, 0],
        'upper': ic.iloc[:, 1]
    })

    ## Extraction des variables d'années relatives à partir du nom des variables
    def _extract_time_internal(col_name):
        suffix = col_name.split('_')[-1]
        if 'm' in suffix:
            return -int(suffix.replace('m', ''))
        return int(suffix)

    df_plot['temps_relatif'] = [_extract_time_internal(idx) for idx in df_plot.index]

    ## Ajout manuel de l'année de référence (-1) où le coef est 0 par définition
    row_ref = pd.DataFrame({
        'coef': [0], 
        'lower': [0], 
        'upper': [0], 
        'temps_relatif': [-1]
    })
    
    df_plot = pd.concat([df_plot, row_ref]).sort_values('temps_relatif')

    ## Graphique
    plt.figure(figsize=(10, 6))

    # Zone grise pour l'année du traitement
    plt.axvspan(-0.5, 0.5, color='grey', alpha=0.2, label='Année du traitement', zorder=0)

    # Barres d'erreur
    plt.errorbar(
        df_plot['temps_relatif'], 
        df_plot['coef'], 
        yerr=[df_plot['coef'] - df_plot['lower'], df_plot['upper'] - df_plot['coef']],
        fmt='o',
        color='#7A57FF',
        ecolor='#CABCFF',
        elinewidth=4,
        capsize=0,
        label='Coefficients estimés et IC à 95%'
    )

    plt.axhline(0, linestyle='dashed', color='grey')

    plt.title("Étude d\'évènement : Impact sur les arrêts maladie")
    plt.xlabel("Années relatives au traitement (0 = 2020)")
    plt.ylabel("Effet estimé sur l\'activité")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()