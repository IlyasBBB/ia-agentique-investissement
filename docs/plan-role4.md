# Plan du rôle 4

Le plan complet des quatre rôles est dans `docs/Plan_projet_ML_IA_agentique.xlsx`. Ce fichier ne reprend que les tâches d'Ilyas. Statuts possibles : à faire, en cours, terminé.

Dernière mise à jour : 16 novembre 2026.

## Tâche reprise au rôle 1

| Tâche | Échéance | Statut | Remarque |
|---|---|---|---|
| Choisir et préparer le jeu de données (`data/raw/projets.csv`, `data/README.md`) | lun. 16 nov. | à faire | Pas encore choisi. Les rôles 2 et 3 attendent ce fichier pour travailler sur de vraies données. Priorité numéro un. Voir `docs/donnees.md` |

## Tâches du rôle 4

| N° | Tâche | Dépend de | Échéance | Statut | Livrable |
|---|---|---|---|---|---|
| 1 | Créer le dépôt Git et inviter l'équipe | Réunion de lancement | dim. 15 nov. | à vérifier | Dépôt GitHub avec la structure de dossiers |
| 2 | Fausses fonctions | Contrat commun | lun. 16 nov. | terminé (version provisoire) | `src/mocks.py`, `src/contrat.py`, `tests/test_contrat.py` |
| 3 | Coder l'agent | Tâche 2 | jeu. 19 nov. | à faire | `src/agent.py`, `prompts/systeme.txt` |
| 4 | Coder l'application Streamlit | Tâches 2 et 3. Graphique SHAP du rôle 1 (factice en attendant) | ven. 20 nov. | à faire | `app/app.py` |
| 5 | Tester l'agent sur 5 scénarios | Tâche 3 | sam. 21 nov. | à faire | `tests/test_agent.py`, `reports/tests_agent.md` |
| 6 | Intégration des vrais modules | IMPOSSIBLE avant `predict` (rôle 1), `rank` (rôle 2), `extract` et `retrieve` (rôle 3), attendus le 21 | mar. 24 nov. | bloqué | Application complète en local |
| 7 | Déploiement ou démonstration locale | Tâche 6 | mar. 24 nov. | bloqué | Démonstration prête à filmer |
| 8 | README final | Tâche 6, sources des données | mer. 25 nov. | bloqué | `README.md` |
| 9 | Partie du rapport (LLM, agents, architecture, tests) | Tâche 5, puis tâche 6 pour les résultats | brouillon sam. 21, final ven. 27 | à faire | 4 à 6 pages, 3 à 4 diapositives |
| 10 | Assembler rapport et présentation | Parties des rôles 1, 2 et 3 | ven. 27 nov. | bloqué | Rapport et PowerPoint complets |

## Jalons communs

- Sam. 21 nov. : modules des trois autres rôles sur le dépôt.
- Dim. 22 nov. : réunion d'assemblage.
- Sam. 28 nov. : enregistrement de la vidéo.
- Dim. 29 nov. : relecture.
- Lun. 30 nov. : remise.

## Journal

- 16 nov. : structure du projet créée. Contrat, fausses fonctions et tests du contrat en place.
