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
| 5 | Modèles de référence | 20 nov. | faite, branche `role1/tache-5` | `notebooks/role1/02_modeles.ipynb` : R² 0 (moyenne) et 0,39 (linéaire) ; risque 40 % / F1 macro 0,19 (classe majoritaire) |
| 6 | Ridge, Lasso, ElasticNet | 20 nov. | à faire | notebook 02 |
| 7 | Classification du risque | 20 nov. | à faire | notebook 02 |
| 8 | Tableau comparatif, test ouvert une fois | 20 nov. | à faire | `reports/role1/` |
| 9 | Interprétabilité (SHAP, permutation, erreurs) | 21 nov. | à faire | `notebooks/role1/03_interpretabilite.ipynb`, `reports/role1/shap_importance.csv` |
| 10 | `predict`, modèles `joblib`, tests | **21 nov.** | à faire | `src/supervise.py`, `tests/test_supervise.py` |
| 11 | Partie du rapport et diapositives | brouillon 21, final 27 | à faire | `reports/role1/rapport_partie1.md` |

## Décisions

- Entrées : `secteur`, `pays`, `region`, `type_financement`, `cout_total` (en log10), `duree_mois`, `annee_approbation`. Exclus : `id_projet`, `nom`, `statut`, et la rentabilité comme entrée du modèle de risque (inconnue pour les projets en cours).
- Séparation stratifiée sur le risque, `random_state=42`, avant tout prétraitement : 1 050 projets d'entraînement, 225 de validation, 225 de test, chacun à 40 / 40 / 20 % de `faible` / `moyen` / `eleve`. Anomalies injectées : 17 en entraînement, 5 en validation, 3 en test.
- Prétraitement : log10 du coût puis médiane et standardisation ; durée et année : médiane et standardisation ; catégories : indicatrices avec `handle_unknown="ignore"`. Un manquant catégoriel devient « inconnu » (donc des zéros, comme une catégorie jamais vue) plutôt que la valeur la plus fréquente, qui lui donnerait à tort l'effet d'un pays précis. `None` et `pd.NA` sont ramenés à NaN avant l'imputation, car `SimpleImputer` ne les reconnaît pas.
- Anomalies (tâche 5) : les 6 rentabilités fabriquées de l'entraînement font tomber le R² de validation de la régression linéaire de 0,39 à 0,22 ; retirer en plus les 11 coûts ou durées modifiés ne change presque rien (0,389). Choix : apprendre la rentabilité sans les rentabilités fabriquées connues. La validation et le test contiennent chacun 2 rentabilités fabriquées, qui font plus que doubler la RMSE : on compare les modèles sur la validation sans cibles fabriquées, et on donne aussi la mesure sur la validation complète. Le risque n'est pas touché (calculé avant l'injection).
