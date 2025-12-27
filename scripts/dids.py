import pandas as pd
from linearmodels import PanelOLS
import matplotlib.pyplot as plt


# Préparation des données
def data_prep(df, min_obs):
    """
    Cette fonction filtre les groupes ayant suffisamment d'observations sur la période
    et agrège les données au niveau Groupe x Année.

    Arguments :
        df : le dataframe source contenant les données individuelles ;
        min_obs : le nombre minimum d'observations requis par année pour conserver un groupe ;

    Output : df_agg : le dataframe agrégé
    """
    
    # Sélection des CSP x NAF avec au moins 'min_obs' actifs pour toutes les années de la période
    df = df.groupby("groupe").filter(
        lambda x: (x.groupby('ANNEE').size() >= min_obs).all()
    )

    # Agrégation par CSP x NAF
    df = df.groupby(["groupe", 'ANNEE']).agg(
        sh_arret=('is_RABS_2', 'mean'),
        n_individus=('is_RABS_2', 'size'),
        cat_expose=('score_exposition', 'mean'),
    ).reset_index()

    # Évaluation des groupes obtenus
    n_groupes = df["groupe"].nunique()
    print(f"Nombre de combinaisons conservées : {n_groupes}")
    
    return df


# Estimation de la double-différences statistique
def estimation_statique(df, traitement, tableau=False):
    """
    Cette fonction estime un modèle de Différence-in-Différences (DiD) canonique avec 
    des effets fixes par groupe et année (Two-Way Fixed Effects).

    Arguments :
        - df : le dataframe contenant les données au format long ;
        - traitement : le nom de la variable de traitement (variable indicatrice).
        - tableau : booléen.
        Si True, affiche le tableau complet.
        Si False, affiche une phrase de synthèse.

    Output : tableau de résultat
    """

    # Définition de l'index pour les données de panel dans linearmodels
    df_reg = df.set_index(['groupe', 'ANNEE'])

    # Définition de la variable expliquée Y et de la variable dépendante X
    Y = df_reg['sh_arret']
    X = df_reg[[traitement]]

    # Configuration du modèle
    modele = PanelOLS(
        Y,                      # Variable expliquée: part d'arrêts maladie Y_it
        X,                      # Variable explicatives: 1{t >= 2020} x 1{exposé == 1}
        entity_effects=True,    # Effets-fixes groupes: gamma_i (capte les différences de niveau entre les groupes)
        time_effects=True,      # Effets-fixes années: delta_t (capte l'évolution commune à tous les groupes)
        drop_absorbed=True      # Pour éviter les erreurs de colinéarité
    )

    # Estimation avec erreurs-types clusterisées (Robustes à l'autocorrélation intra-groupe)
    resultat = modele.fit(cov_type='clustered', cluster_entity=True)

    if tableau:
        print(resultat.summary)
    else:
        coef = resultat.params[traitement]
        pval = resultat.pvalues[traitement]
        ic = resultat.conf_int().loc[traitement]
        lower = ic['lower']
        upper = ic['upper']

        print(f"L'effet estimé est de {coef:.4f}, avec un intervalle de confiance à 95 % de [{lower:.4f}, {upper:.4f}]\n et une p-valeur associée de {pval:.4f}.")

    #return resultat


# Estimation de la double-différences dynamique
def estimation_dynamique(df, var_traitement='expose_1'):
    """
    Estime un modèle d'étude d'événement (Event Study).
    
    Arguments :
        - df : DataFrame contenant les données.
        - var_traitement : Nom de la variable indicatrice du groupe traité (ex: 'expose_1').
        - verbose : Si True, affiche le résumé statistique.

    Output : Le résultat de l'estimation PanelOLS.
    """
    # 1. Copie pour ne pas modifier le DataFrame original
    df_reg = df.copy()

    # 2. Gestion de l'index pour éviter la KeyError
    # Si 'groupe' n'est pas dans les colonnes, c'est qu'il est déjà dans l'index
    if 'groupe' not in df_reg.columns:
        df_reg = df_reg.reset_index()

    # 3. Création automatique des termes d'interaction
    # On cherche toutes les colonnes créées par get_dummies (commençant par 'ttt_')
    dummies_annees = [c for c in df_reg.columns if c.startswith('ttt_')]
    vars_interaction = []

    for col in dummies_annees:
        nom_interaction = f'{var_traitement}_{col}'
        # Calcul de l'interaction : 1{t == j} * 1{exposé == 1}
        df_reg[nom_interaction] = df_reg[var_traitement] * df_reg[col]
        vars_interaction.append(nom_interaction)

    # 4. Préparation finale de l'index pour PanelOLS
    df_reg = df_reg.set_index(["groupe", "ANNEE"])
    
    # 5. Définition des variables
    Y = df_reg["sh_arret"]
    X = df_reg[vars_interaction]

    # 6. Estimation du modèle
    modele = PanelOLS(
        Y,                      # Variable expliquée: part d'arrêts maladie Y_it
        X,                      # Variable explicatives: 1{t == j} x 1{exposé == 1}
        entity_effects=True,    # Effets-fixes groupes: gamma_i (capte les différences de niveau entre les groupes)
        time_effects=True,      # Effets-fixes années: delta_t (capte l'évolution commune à tous les groupes)
        drop_absorbed=True      # Pour éviter les erreurs de colinéarité
    )

    # Erreurs-types robustes clusterisées par entité
    resultat = modele.fit(cov_type='clustered', cluster_entity=True)

    return resultat

# Représentation graphique de l'étude d'évènement
def plot_event_study(resultat):
    """
    Cette fonction trace le graphique des coefficients de l'étude d'évènement
    à partir du résultat de {estimation_dynamique}

    Argument : le dataset produit par "estimation_dynamique"

    Output : représentation graphique de l'étude d'évènement
    """

    # Récupération des coefficients estimés et des intervalles de confiance
    est = resultat.params
    ic = resultat.conf_int()
    
    # Enregistrement dans un dataframe
    df_plot = pd.DataFrame({
        'coef': est,
        'lower': ic.iloc[:, 0],
        'upper': ic.iloc[:, 1]
    })

    # Extraction des variables d'années relatives à partir du nom des variables
    def _extract_time_internal(col_name):
        suffix = col_name.split('_')[-1]
        if 'm' in suffix:
            return -int(suffix.replace('m', ''))
        return int(suffix)
    df_plot['temps_relatif'] = [_extract_time_internal(idx) for idx in df_plot.index]

    # Ajout manuel de l'année de référence (-1) où le coef est 0 par définition
    row_ref = pd.DataFrame({
        'coef': [0],            # Pas de point estimate pour cette année relative
        'lower': [0],           # Pas d'intervalle de confiance pour cette année relative
        'upper': [0],           # Pas d'intervalle de confiance pour cette année relative
        'temps_relatif': [-1]
    })
    df_plot = pd.concat([df_plot, row_ref]).sort_values('temps_relatif')


    # Graphique
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
        label='Coefficients estimés et intervalle de confiance à 95 %'
    )

    plt.axhline(0, linestyle='dashed', color='grey')

    plt.title("Étude d\'évènement : Impact sur les arrêts maladie")
    plt.xlabel("Années relatives au traitement (0 = 2020)")
    plt.ylabel("Effet estimé sur la part d'arrêts de travail")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()