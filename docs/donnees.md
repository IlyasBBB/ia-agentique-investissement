# Jeu de données

État au 16 novembre 2026 : jeu choisi et préparé. Reste à mettre le contrat à jour et à prévenir l'équipe (étapes 4 et 5 plus bas).

## Choix : jeu hybride

- **Données réelles** : projets de la Banque mondiale (Projects & Operations), licence CC BY 4.0. Nom, pays, région, secteur, coût, dates et textes de description.
- **Cibles simulées** : `rentabilite` et `risque`, calculées par une formule connue à partir des colonnes réelles, plus du bruit.

Le détail (source, filtres, colonnes, formules, anomalies) est dans `data/README.md`.

## Pourquoi

- La Banque mondiale fournit ce qu'il fallait : plus de 500 projets, des colonnes chiffrées et catégorielles, une licence ouverte, et un vrai texte par projet pour le rôle 3.
- Elle ne publie ni rentabilité ni niveau de risque exploitables dans ces données. Les simuler est accepté par la consigne si la génération est décrite, ce que fait `data/README.md`.
- Une formule connue permet de vérifier les modèles : on sait quels facteurs comptent, donc on peut contrôler que SHAP ou l'importance par permutation les retrouvent.
- Les anomalies injectées donnent une vérité terrain pour évaluer la détection du rôle 2.

Pistes écartées : les jeux Kaggle (licence et qualité à vérifier au cas par cas, pas de texte par projet) et un jeu entièrement simulé (pas de vrais textes).

## Résultat

- `data/processed/projets.csv` : 2 000 projets, 1 500 terminés (cibles connues) et 500 en cours (cibles vides, dossiers à évaluer).
- `data/processed/anomalies_injectees.csv` : 40 anomalies fabriquées.
- `data/docs/<id_projet>.txt` : un document par projet.
- 11 secteurs, 158 pays, approbations de 1987 à 2026.
- Contrôle d'apprenabilité avec un modèle linéaire : R² de 0,55 sur la rentabilité, 58 % d'exactitude sur le risque contre 43 % pour la classe majoritaire.

## Marche à suivre

1. Fait. `python scripts/telecharger_donnees.py` écrit le brut dans `data/raw/` sans le modifier.
2. Fait. `python scripts/preparer_donnees.py` écrit `data/processed/` et `data/docs/`.
3. Fait. `data/README.md` rempli.
4. À faire. Mettre à jour `docs/contrat.md` et `src/contrat.py` : nouvelles colonnes `region`, `annee_approbation`, `type_financement`, `statut` ; identifiants du type `P123456` ; cibles vides pour les projets en cours.
5. À faire. Prévenir les rôles 1, 2 et 3 que le fichier est disponible.

## À dire dans le rapport

Les cibles sont simulées : les résultats des modèles montrent que la méthode fonctionne, pas que la Banque mondiale obtient ces rentabilités.
