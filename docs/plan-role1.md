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
| 6 | Ridge, Lasso, ElasticNet | 20 nov. | fusionnée (PR n°16) ; complément fusionné (PR n°17) : forêt aléatoire R² 0,40, LightGBM 0,37, moins bien que Ridge | notebook 02 : R² validation dans les bornes 0,44 (Ridge, ElasticNet), 0,42 (Lasso, 66 colonnes sur 175 mises à zéro) ; `reports/role1/figures/regularisation_alpha.png` |
| 7 | Classification du risque | 20 nov. | fusionnée (PR n°17) | notebook 02 : F1 macro validation 0,53 (logistique), 0,57 (forêt), 0,59 (LightGBM) ; exactitude 55 à 59 % ; AUC élevé 0,86 ; `reports/role1/figures/confusion_risque.png` |
| 8 | Tableau comparatif, test ouvert une fois | 20 nov. | faite, branche `role1/tache-8` | `reports/role1/comparaison_rentabilite.csv`, `comparaison_risque.csv`, `comparaison_modeles.png`, `resultats_test.csv` ; test : Ridge R² 0,47 dans les bornes (0,12 complet), LightGBM 67 % / F1 0,66 |
| 9 | Interprétabilité (SHAP, permutation, erreurs) | 21 nov. | à faire | `notebooks/role1/03_interpretabilite.ipynb`, `reports/role1/shap_importance.csv` |
| 10 | `predict`, modèles `joblib`, tests | **21 nov.** | à faire | `src/supervise.py`, `tests/test_supervise.py` |
| 11 | Partie du rapport et diapositives | brouillon 21, final 27 | à faire | `reports/role1/rapport_partie1.md` |

## Décisions

- Entrées : `secteur`, `pays`, `region`, `type_financement`, `cout_total` (en log10), `duree_mois`, `annee_approbation`. Exclus : `id_projet`, `nom`, `statut`, et la rentabilité comme entrée du modèle de risque (inconnue pour les projets en cours).
- Séparation stratifiée sur le risque, `random_state=42`, avant tout prétraitement : 1 050 projets d'entraînement, 225 de validation, 225 de test, chacun à 40 / 40 / 20 % de `faible` / `moyen` / `eleve`. Anomalies injectées : 17 en entraînement, 5 en validation, 3 en test.
- Prétraitement : log10 du coût puis médiane et standardisation ; durée et année : médiane et standardisation ; catégories : indicatrices avec `handle_unknown="ignore"`. Un manquant catégoriel devient « inconnu » (donc des zéros, comme une catégorie jamais vue) plutôt que la valeur la plus fréquente, qui lui donnerait à tort l'effet d'un pays précis. `None` et `pd.NA` sont ramenés à NaN avant l'imputation, car `SimpleImputer` ne les reconnaît pas.
- Rentabilités aberrantes (tâche 5, corrigée à la demande d'Ilyas) : règle apprise sur l'entraînement seulement, rentabilité hors de [Q1 − 3·IQR ; Q3 + 3·IQR] = [−6,1 % ; 23,3 %] (`supervise.bornes_rentabilite`), appliquée telle quelle à la validation et au test. Les modèles de rentabilité sont appris sur l'entraînement dans les bornes (1 044 projets) ; mesures données sur la validation complète et sur la validation dans les bornes. `anomalies_injectees.csv` ne sert qu'à vérifier après coup : la règle attrape les 10 rentabilités fabriquées et aucun autre projet. Effet sur la régression linéaire (validation dans les bornes) : R² de 0,22 à 0,39. Le risque garde tous les projets.
- Modèle de rentabilité : les modèles à arbres (forêt aléatoire, LightGBM) font moins bien que Ridge et ElasticNet (R² 0,40 et 0,37 contre 0,44 sur la validation dans les bornes ; même ordre en validation croisée). La rentabilité simulée est surtout une somme d'effets, dont un effet pays sur 158 catégories, que le modèle linéaire régularisé estime mieux.
- Modèle de risque : validation croisée (logistique en tête, F1 0,65) et validation (LightGBM en tête, F1 0,59) ne donnent pas le même classement ; les écarts sont du même ordre que le bruit de mesure (écart-type entre plis 0,02 à 0,04 ; ± 6,5 points d'exactitude sur 225 projets). L'avance des arbres vient du rappel de la classe `eleve` (53 à 58 % contre 38 %), grâce à `class_weight="balanced"`. Choix à la tâche 8.
- Choix des modèles (tâche 8, règle fixée par Salma le 19 nov. avant d'ouvrir le test) : rentabilité = plus petite RMSE sur la validation dans les bornes, le plus simple à 0,01 près → Ridge (α ≈ 3,2) ; risque = meilleur F1 macro sur la validation → LightGBM. Test ouvert une seule fois, sur ces deux modèles appris sur l'entraînement : Ridge R² 0,47 (RMSE 2,24) dans les bornes, 0,12 sur le test complet ; LightGBM 67 % d'exactitude, F1 macro 0,66, AUC élevé 0,88. Le 67 % dépasse le plafond annoncé : vérifié, pas de fuite (aucun projet commun, aucune entrée identique) ; tirage favorable (environ 7 % de chances si la vraie exactitude est 62 %). Estimation honnête : environ 60 %, à quelques points près.
