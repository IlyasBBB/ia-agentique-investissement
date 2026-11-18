# Plan du rôle 1 : apprentissage supervisé

Responsable : Salma Akazi. Module à livrer : `src/supervise.py` (`predict`), sur `main` le samedi 21 novembre 2026. Numérotation des tâches : feuille « Rôle 1 » de `docs/Plan_projet_ML_IA_agentique.xlsx`. Un commit par tâche au minimum, une branche `role1/tache-N` par pull request.

## Constats de départ (tâche 2)

- 2 000 projets réels de la Banque mondiale : 1 500 terminés (cibles connues) et 500 en cours (cibles vides, à évaluer par l'agent). Cibles `rentabilite` et `risque` **simulées**.
- Aucune valeur manquante ni doublon chez les projets terminés.
- Risque : 40 % `faible`, 40 % `moyen`, 20 % `eleve` ; référence « classe majoritaire » à 40 %.
- Coût très asymétrique (asymétrie 18,5, 0,06 après log10) : passer au log10.
- 158 pays, dont 42 avec moins de 5 projets ; 3 pays des projets en cours n'apparaissent jamais parmi les terminés.
- Projets en cours plus récents (médiane 2022 contre 2009) et plus chers (107 contre 46 millions de dollars).
- 25 des 40 anomalies fabriquées sont parmi les terminés ; elles font passer l'écart-type de la rentabilité de 3,1 à 5,1 points.

## État des tâches

| N° | Tâche | Échéance | État | Livrables |
|---|---|---|---|---|
| 2 | Analyse exploratoire | 19 nov. | fusionnée (PR n°12) | `notebooks/role1/01_exploration.ipynb`, `reports/role1/figures/` |
| 3 | Séparation 70 / 15 / 15 stratifiée | 19 nov. | fusionnée (PR n°13) | `scripts/separer_jeux_supervise.py`, `reports/role1/separation.csv` : 1 050 / 225 / 225 projets |
| 4 | Prétraitement (`Pipeline`, `ColumnTransformer`) | 19 nov. | fusionnée (PR n°14) | `src/supervise.py` : `creer_pretraitement`, `charger_jeux` (175 colonnes après encodage) |
| 5 | Modèles de référence | 20 nov. | fusionnée (PR n°15) | `notebooks/role1/02_modeles.ipynb` : R² 0 (moyenne) et 0,39 (linéaire, validation dans les bornes ; 0,08 sur la validation complète) ; risque 40 % / F1 macro 0,19 (classe majoritaire) |
| 6 | Ridge, Lasso, ElasticNet | 20 nov. | fusionnée (PR n°16) ; complément sur `role1/tache-7` : forêt aléatoire R² 0,40, LightGBM 0,37, moins bien que Ridge | notebook 02 : R² validation dans les bornes 0,44 (Ridge, ElasticNet), 0,42 (Lasso, 66 colonnes sur 175 mises à zéro) ; `reports/role1/figures/regularisation_alpha.png` |
| 7 | Classification du risque | 20 nov. | à faire | notebook 02 |
| 8 | Tableau comparatif, test ouvert une fois | 20 nov. | à faire | `reports/role1/` |
| 9 | Interprétabilité (SHAP, permutation, erreurs) | 21 nov. | à faire | `notebooks/role1/03_interpretabilite.ipynb`, `reports/role1/shap_importance.csv` |
| 10 | `predict`, modèles `joblib`, tests | **21 nov.** | à faire | `src/supervise.py`, `tests/test_supervise.py` |
| 11 | Partie du rapport et diapositives | brouillon 21, final 27 | à faire | `reports/role1/rapport_partie1.md` |

## Décisions

- Entrées : `secteur`, `pays`, `region`, `type_financement`, `cout_total` (en log10), `duree_mois`, `annee_approbation`. Exclus : `id_projet`, `nom`, `statut`, et la rentabilité comme entrée du modèle de risque (inconnue pour les projets en cours).
- Séparation stratifiée sur le risque, `random_state=42`, avant tout prétraitement : 1 050 projets d'entraînement, 225 de validation, 225 de test, chacun à 40 / 40 / 20 % de `faible` / `moyen` / `eleve`. Anomalies injectées : 17 en entraînement, 5 en validation, 3 en test.
- Prétraitement : log10 du coût puis médiane et standardisation ; durée et année : médiane et standardisation ; catégories : indicatrices avec `handle_unknown="ignore"`. Un manquant catégoriel devient « inconnu » (donc des zéros, comme une catégorie jamais vue) plutôt que la valeur la plus fréquente, qui lui donnerait à tort l'effet d'un pays précis. `None` et `pd.NA` sont ramenés à NaN avant l'imputation, car `SimpleImputer` ne les reconnaît pas.
- Rentabilités aberrantes (tâche 5, corrigée à la demande d'Ilyas) : règle apprise sur l'entraînement seulement, rentabilité hors de [Q1 − 3·IQR ; Q3 + 3·IQR] = [−6,1 % ; 23,3 %] (`supervise.bornes_rentabilite`), appliquée telle quelle à la validation et au test. Les modèles de rentabilité sont appris sur l'entraînement dans les bornes (1 044 projets) ; mesures données sur la validation complète et sur la validation dans les bornes. `anomalies_injectees.csv` ne sert qu'à vérifier après coup : la règle attrape les 10 rentabilités fabriquées et aucun autre projet. Effet sur la régression linéaire (validation dans les bornes) : R² de 0,22 à 0,39. Le risque garde tous les projets.
- Modèle de rentabilité : les modèles à arbres (forêt aléatoire, LightGBM) font moins bien que Ridge et ElasticNet (R² 0,40 et 0,37 contre 0,44 sur la validation dans les bornes ; même ordre en validation croisée). La rentabilité simulée est surtout une somme d'effets, dont un effet pays sur 158 catégories, que le modèle linéaire régularisé estime mieux.
