# Analyse des Corrélations des Prix de l'Électricité en Europe

##  Objectif

Ce projet a pour but de réaliser une **analyse statistique avancée** des prix de l’électricité dans plusieurs pays européens afin de :
- Identifier les **corrélations** entre les marchés.
- Réduire la **dimensionnalité** du jeu de données par **Analyse en Composantes Principales (ACP)**.
- Visualiser les **relations géométriques** entre les pays sur les premiers axes principaux.
- Évaluer combien de facteurs sont nécessaires pour expliquer au moins **93 % à 95 % de la variance totale**.

##  Technologies utilisées

- **Langage :** Python 3
- **Bibliothèques :**
  - `pandas` pour le traitement de données
  - `numpy` pour le calcul scientifique
  - `numpy.lib.scimath` pour la gestion des racines complexes
  - `math` pour les calculs trigonométriques
  - `plotly` (dans la version Jupyter) pour les visualisations 3D

##  Données

Le fichier source est :
- `EU electricity price_v2.csv` : contient les prix de l’électricité par pays sur une période donnée.

Les colonnes incluent :
- `Date`
- `Country`
- `price`

##  Détails des traitements

- Calcul de la **matrice de corrélation** entre pays.
- Décomposition spectrale pour obtenir les **valeurs propres** et **vecteurs propres**.
- Vérification de la relation :  
  `M = P * D * P.T` (où `M` est la matrice de corrélation).
- Calcul des **composantes principales** (PC1, PC2, PC3).
- Représentation des pays dans un espace réduit à 3 dimensions.
- Calcul des **angles** entre les vecteurs pays et les composantes principales (cosinus).
- Estimation de la **variance expliquée cumulée**.
- Réduction des valeurs propres négatives par `1e-8` pour stabiliser l’ACP
