# Projet python : l'effet durable du Covid-19 sur l'offre de travail

Projet réalisé en 2025 par Lucile Aubain, Jean Lavallée et Paul Hobeika dans le cadre du cours de *Python pour la data science* enseigné en deuxième année de l'ENSAE.

## Introduction <a name="introduction">

Notre projet vise à étudier les éventuels effets durables de l'exposition au Covid-19 sur les arrêts de travail pour raison de santé, et ainsi contribuer à l'analyse des liens entre santé et offre de travail. 

## Données <a name="donnees">

### Données d'arrêts maladie

L'[enquête sur l'emploi, le chômage et l'inactivité](https://www.insee.fr/fr/metadonnees/source/serie/s1223) (appelée Enquête emploi en continu, ou "EEC") est réalisée par l'Insee. Elle vise à dresser un état des lieux en continu du marché du travail français. Menée une fois par an depuis 1950, elle est réalisée sur toutes les semaines de l'année depuis 2003. Nous obtenons les données pour les années 2010 à 2024 à partir des fichiers détails disponibles sur le site de l'Insee.

### Propagation du Covid-19

Nous estimons l'intensité de la propagation du Covid-19 pour l'année 2020 à partir des données d'hopitalisation disponibles *via* l'API [Odissé](https://odisse.santepubliquefrance.fr/api/explore/v2.1/console) de Santé Publique France.


## Organisation du projet

Le fichier `script.ipnyb` constitue notre rapport final. Le code qu'il contient fait appel aux différents fichiers présents dans le dossier `scripts/` : 

- `requirements.txt` contient les librairies python que l'on utilise, elles sont installées par `pip` si nécessaire
- `import_eec` contient les différentes fonctions utilisées pour l'import et le recodage des données de l'enquête mploi en continu
- `import_eec_all` comporte les liens URL des différentes bases de données présentes sur le site de l'Insee (plus des copies de secours sauvegardées sur le stockage du SSPCloud) et une fonction permettant d'importer et d'assembler l'ensemble des bases de données de l'EEC de 2010 à 2024. Nous avons réalisé une copie de cette dernière base de données sur le SSPCloud de manière à écourter le temps d'importation (le script invite à choisir l'une des deux méthodes).
- `hospitalisation.py` rassemble les fonctions permettant d'importer les données d'hospitalisation pour Covid-19 depuis l'API Odissé, de les visualiser et d'étudier leur corrélation avec les arrêts maladie déclarés dans l'enquête emploi
- `exposition.py` contient le code permettant de construire et d'étudier la variable d'exposition au Covid-19 que l'on utilise par la suite comme variable explicative
- enfin, `dids.py` rassemble les fonctions utilisées dans l'analyse en doubles différences.

## Méthode

Notre analyse repose sur la construction d'une variable d'exposition au Covid-19 durant l'année 2020 à partir des données sur les arrêts maladie de la même année, méthode présentée dans la première partie du rapport. Dans la seconde, nous estimons l'effet causal de cette variable d'exposition sur les arrêts maladie dans les années qui suivent.


## Résultats

Nous estimons que l'exposition au Covid-19 en 2020 augmente de 0,9 point de pourcentage en moyenne la probabilité d’être en arrêt maladie dans les années suivantes pour les groupes exposés par rapport aux autres, soit une augmentation relative d’environ un tiers par rapport à la moyenne de l’échantillon. Nous concluons donc à un effet significatif et durable de l'exposition au Covid-19 en 2020 sur le nombre d'arrêts maladie et donc sur l'offre de travail dans les années suivantes.