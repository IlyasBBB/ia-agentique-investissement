# Plan du rôle 2 : non supervisé et classement multicritère

Responsable : Zineb Baghzali. Module à livrer : `src/mcda.py` (`rank`, `detect_anomalies`), sur `main` le samedi 21 novembre 2026. Numérotation des tâches : feuille « Rôle 2 » de `docs/Plan_projet_ML_IA_agentique.xlsx`. Un commit par tâche au minimum, une branche `role2/tache-N` par pull request.

## État des tâches

| N° | Tâche | Échéance | État | Livrables |
|---|---|---|---|---|
| 1 | Préparation des données | au plus vite | à faire | `notebooks/role2/01_non_supervise.ipynb` |
| 2 | ACP | | à faire | notebook 01 |
| 3 | Clustering (K-means, hiérarchique, DBSCAN, mélange gaussien) | | à faire | notebook 01 |
| 4 | Détection d'anomalies (Isolation Forest) | | fait (liste commentée des projets atypiques à reprendre dans le notebook 01), branche `role2/tache-4-5-9` | `detect_anomalies`, `score_anomalie` dans `src/mcda.py` ; `scripts/evaluer_anomalies_mcda.py` ; `reports/role2/anomalies_*.csv` |
| 5 | Somme pondérée (référence) | | fait, branche `role2/tache-4-5-9` | `somme_ponderee`, `normaliser` dans `src/mcda.py` ; tests dans `tests/test_mcda.py` |
| 6 | Poids par AHP | | à faire | `ahp_poids` dans `src/mcda.py` |
| 7 | TOPSIS | | fait, branche `role2/tache-7` | `topsis` dans `src/mcda.py` ; vérification à la main sur 4 projets dans `tests/test_mcda.py` ; 0,6 ms sur 2 000 projets |
| 8 | Sensibilité du classement | | à faire | `notebooks/role2/02_mcda_sensibilite.ipynb` |
| 9 | `rank` et `detect_anomalies` au format du contrat, tests | **21 nov.** | `rank` (somme pondérée) et `detect_anomalies` faits, branche `role2/tache-4-5-9` | `src/mcda.py`, `tests/test_mcda.py` ; `rank` : 3,9 ms sur 2 000 projets |
| 10 | Partie du rapport et diapositives | brouillon 21, final 27 | à faire | `reports/role2/rapport_partie2.md` |

## Décisions

- Somme pondérée (tâche 5) : chaque critère est ramené entre 0 et 1 par min-max **sur les projets reçus** (le meilleur de la sélection vaut 1, le moins bon 0), les critères à minimiser (`proba_risque`, `cout_total`, `duree_mois`) sont inversés. Colonne constante : 0,5 pour tous. Valeur manquante : 0, la pire note (un critère inconnu ne doit pas faire monter un projet). Calcul vectorisé avec numpy.
- `rank` (tâche 9), première version : somme pondérée. Égalités de score départagées par `id_projet` croissant, pour des rangs uniques et un classement qui ne dépend pas de l'ordre des lignes reçues. Colonnes en plus gardées, entrée non modifiée. TOPSIS pourra la remplacer après la tâche 7, selon une règle écrite à l'avance.
- **Contamination de l'Isolation Forest : 5 %**, valeur du plan commun (tâche 4), choisie par Zineb le 19 novembre **avant toute mesure sur `anomalies_injectees.csv`** (ce commit précède l'évaluation dans l'historique Git). Le taux de 2 % annoncé dans `data/README.md` n'est pas utilisé : il vient de la vérité terrain. D'autres contaminations seront mesurées, seulement comme analyse de sensibilité dans le rapport ; `detect_anomalies` garde 5 %.
- Détecteur (tâche 4) : Isolation Forest, 300 arbres, `random_state=42`, appris **une fois** sur les 2 000 projets de `projets.csv` (anomalies comprises, sans aucune étiquette), enregistré dans `models/mcda_anomalies.joblib` (hors Git), réappris s'il manque ou si la version de scikit-learn change. Appliqué tel quel à la sélection reçue (même 1 projet) : un projet reçoit la même réponse seul ou parmi les 2 000.
- Variables : log10 du coût, log10 de la durée, log10 du coût par mois, calculées à partir des seules colonnes obligatoires (présentes aussi dans un fichier importé). Secteur et pays non utilisés. Valeur manquante : médiane des 2 000 projets. Coût ou durée nul ou négatif : signalé d'office.
- Limites connues avant l'évaluation : les 10 rentabilités extrêmes sont indétectables (la rentabilité n'existe pas pour les projets en cours) ; un coût multiplié par 10 à 50 sur un petit projet peut tomber parmi les grands projets réels (34 projets dépassent 2 000 M$, le plus gros 37 707 M$) et passer inaperçu.
- Évaluation (tâche 4, faite après le commit de la décision des 5 %) sur les 2 000 projets : 100 signalés, 19 vraies anomalies, **81 faux positifs** ; précision 0,19, rappel 0,48 (0,60 sans les rentabilités extrêmes, indétectables). Par type : durées incohérentes 13 sur 13, coûts aberrants 5 sur 17, rentabilités extrêmes 1 sur 10 (par hasard). Projets en cours (ceux de l'agent) : 24 signalés sur 500, 9 vraies anomalies sur 15, 15 faux positifs (précision 0,38, rappel 0,60).
- Les faux positifs sont surtout de vrais projets rares : 37 durées courtes (6 à 8 mois, au bord du filtre de 6 mois de la préparation), 27 coûts extrêmes réels (de 1 à 22 000 M$), 17 coûts par mois inhabituels. Le détecteur repère des projets rares, pas des projets faux : en recommandation, un dossier signalé est à vérifier, pas forcément mauvais. À dire dans le rapport.
- Sensibilité à la contamination (analyse seulement, `detect_anomalies` garde 5 %) : de 1 % à 10 %, la précision baisse de 0,45 à 0,11 et le rappel monte de 0,23 à 0,55 (`reports/role2/anomalies_sensibilite.csv`).
- TOPSIS (tâche 7) : normalisation vectorielle, pondération, solutions idéale et anti-idéale, distances euclidiennes, score d- / (d+ + d-). Valeur manquante : pire valeur observée du critère. Colonne constante : ne compte pas (idéale = anti-idéale). Un seul projet ou projets identiques : 0,5. Une rentabilité négative ne pose pas de problème (la division par la norme garde l'ordre). Vérifiée à la main sur 4 projets (scores 0,746 ; 0,517 ; 0,373 ; 0,314). `rank` reste sur la somme pondérée : le passage à TOPSIS se décidera selon une règle écrite avant de comparer les deux méthodes.
