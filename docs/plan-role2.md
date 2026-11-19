# Plan du rôle 2 : non supervisé et classement multicritère

Responsable : Zineb Baghzali. Module à livrer : `src/mcda.py` (`rank`, `detect_anomalies`), sur `main` le samedi 21 novembre 2026. Numérotation des tâches : feuille « Rôle 2 » de `docs/Plan_projet_ML_IA_agentique.xlsx`. Un commit par tâche au minimum, une branche `role2/tache-N` par pull request.

## État des tâches

| N° | Tâche | Échéance | État | Livrables |
|---|---|---|---|---|
| 1 | Préparation des données | au plus vite | à faire | `notebooks/role2/01_non_supervise.ipynb` |
| 2 | ACP | | à faire | notebook 01 |
| 3 | Clustering (K-means, hiérarchique, DBSCAN, mélange gaussien) | | à faire | notebook 01 |
| 4 | Détection d'anomalies (Isolation Forest) | | à faire | `detect_anomalies` dans `src/mcda.py` |
| 5 | Somme pondérée (référence) | | fait, branche `role2/tache-4-5-9` | `somme_ponderee`, `normaliser` dans `src/mcda.py` ; tests dans `tests/test_mcda.py` |
| 6 | Poids par AHP | | à faire | `ahp_poids` dans `src/mcda.py` |
| 7 | TOPSIS | | à faire | `topsis` dans `src/mcda.py` |
| 8 | Sensibilité du classement | | à faire | `notebooks/role2/02_mcda_sensibilite.ipynb` |
| 9 | `rank` et `detect_anomalies` au format du contrat, tests | **21 nov.** | `rank` fait (somme pondérée), branche `role2/tache-4-5-9` | `src/mcda.py`, `tests/test_mcda.py` ; `rank` : 3,9 ms sur 2 000 projets |
| 10 | Partie du rapport et diapositives | brouillon 21, final 27 | à faire | `reports/role2/rapport_partie2.md` |

## Décisions

- Somme pondérée (tâche 5) : chaque critère est ramené entre 0 et 1 par min-max **sur les projets reçus** (le meilleur de la sélection vaut 1, le moins bon 0), les critères à minimiser (`proba_risque`, `cout_total`, `duree_mois`) sont inversés. Colonne constante : 0,5 pour tous. Valeur manquante : 0, la pire note (un critère inconnu ne doit pas faire monter un projet). Calcul vectorisé avec numpy.
- `rank` (tâche 9), première version : somme pondérée. Égalités de score départagées par `id_projet` croissant, pour des rangs uniques et un classement qui ne dépend pas de l'ordre des lignes reçues. Colonnes en plus gardées, entrée non modifiée. TOPSIS pourra la remplacer après la tâche 7, selon une règle écrite à l'avance.
