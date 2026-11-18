# Plan du rôle 4

Le plan complet des quatre rôles est dans `docs/Plan_projet_ML_IA_agentique.xlsx`. Ce fichier ne reprend que les tâches d'Ilyas. Statuts possibles : à faire, en cours, terminé.

Dernière mise à jour : 18 novembre 2026.

## Tâche reprise au rôle 1

| Tâche | Échéance | Statut | Remarque |
|---|---|---|---|
| Choisir et préparer le jeu de données (`data/processed/projets.csv`, `data/README.md`) | lun. 16 nov. | terminé (reste à mettre le contrat à jour et à prévenir l'équipe) | Jeu hybride : 2 000 projets réels de la Banque mondiale, cibles `rentabilite` et `risque` simulées, 40 anomalies injectées, un document texte par projet. Voir `docs/donnees.md` et `data/README.md` |

## Tâches du rôle 4

| N° | Tâche | Dépend de | Échéance | Statut | Livrable |
|---|---|---|---|---|---|
| 1 | Créer le dépôt Git et inviter l'équipe | Réunion de lancement | dim. 15 nov. | en cours (dépôt public créé le 16 nov., équipe à inviter) | Dépôt GitHub avec la structure de dossiers |
| 2 | Fausses fonctions | Contrat commun | lun. 16 nov. | terminé (version provisoire) | `src/mocks.py`, `src/contrat.py`, `tests/test_contrat.py` |
| 3 | Coder l'agent | Tâche 2 | jeu. 19 nov. | terminé (16 nov. : essai réel réussi avec `qwen2.5:3b` en 58 s ; qualité à mesurer à la tâche 5) | `src/agent.py`, `prompts/systeme.txt`, `tests/test_agent.py`. Cinq outils sur tout le portefeuille, `pipeline_fixe` comme référence et secours |
| 4 | Coder l'application Streamlit | Tâches 2 et 3. Graphique SHAP du rôle 1 (factice en attendant) | ven. 20 nov. | terminé (16 nov. : application codée et redessinée ; graphique SHAP factice en attendant le rôle 1) | `app/app.py`, `app/composants.py`, `app/graphiques.py`, `app/style.css`, `.streamlit/config.toml`, `tests/test_app.py` |
| 5 | Tester l'agent sur 5 scénarios | Tâche 3 | sam. 21 nov. | terminé avec les fausses fonctions (16 nov. : 5 scénarios, 2 passages, agent et pipeline fixe ; à refaire après la tâche 6 avec `python scripts/evaluer_agent.py --passages 2`) | `tests/test_agent.py`, `reports/tests_agent.md`, `src/scenarios.py`, `scripts/evaluer_agent.py`, `reports/resultats_scenarios.csv` |
| 6 | Intégration des vrais modules | IMPOSSIBLE avant `predict` (rôle 1), `rank` (rôle 2), `extract` et `retrieve` (rôle 3), attendus le 21 | mar. 24 nov. | bloqué | Application complète en local |
| 7 | Déploiement ou démonstration locale | Tâche 6 | mar. 24 nov. | bloqué | Démonstration prête à filmer |
| 8 | README final | Tâche 6, sources des données | mer. 25 nov. | en cours (18 nov. : version complète, à vérifier après l'intégration) | `README.md` |
| 9 | Partie du rapport (LLM, agents, architecture, tests) | Tâche 5, puis tâche 6 pour les résultats | brouillon sam. 21, final ven. 27 | en cours (18 nov. : brouillon dans `reports/rapport/partie_role4.md`, résultats d'intégration à compléter) | 4 à 6 pages, 3 à 4 diapositives |
| 10 | Assembler rapport et présentation | Parties des rôles 1, 2 et 3 | ven. 27 nov. | bloqué | Rapport et PowerPoint complets |

## Jalons communs

- Sam. 21 nov. : modules des trois autres rôles sur le dépôt.
- Dim. 22 nov. : réunion d'assemblage.
- Sam. 28 nov. : enregistrement de la vidéo.
- Dim. 29 nov. : relecture.
- Lun. 30 nov. : remise.

## Journal

- 16 nov. : structure du projet créée. Contrat, fausses fonctions et tests du contrat en place.
- 16 nov. : dépôt public https://github.com/IlyasBBB/ia-agentique-investissement créé, premier commit poussé sur `main`.
- 16 nov. : jeu de données choisi et préparé (Banque mondiale, cibles simulées). Scripts `scripts/telecharger_donnees.py` et `scripts/preparer_donnees.py`, tests `tests/test_donnees.py`.
- 16 nov. : agent codé (`src/agent.py`, `prompts/systeme.txt`) avec son modèle de référence `pipeline_fixe`. Tests sans Ollama dans `tests/test_agent.py` (faux LLM). `pipeline_fixe` traite les 2 000 projets en moins d'une seconde avec les fausses fonctions.
- 16 nov. : essai réel avec `qwen3:4b` arrêté après deux tours de 10 minutes chacun. Le modèle réfléchit toujours (`think=False` sans effet) et produit environ 4 600 jetons de réflexion par tour. Détails et pistes dans `docs/architecture.md`.
- 16 nov. : `qwen2.5:3b` retenu : analyse complète en 58 s. Deux corrections après l'essai : arguments inventés par le LLM ignorés, contrôle automatique des projets et alertes oubliés dans la recommandation.
- 16 nov. : application redessinée (`app/app.py`, `app/composants.py`, `app/graphiques.py`, `app/style.css`, `.streamlit/config.toml`) : thème sombre de terminal financier, bandeau d'indicateurs, cartes des trois projets recommandés, journal en ligne du temps, onglet « Vue 3D » (nuage Plotly rentabilité, risque, impact). Fonctions inchangées, tests de `tests/test_app.py` adaptés.
- 16 nov. : agent évalué sur 5 scénarios et 30 projets en cours, 2 passages, face au pipeline fixe (`reports/tests_agent.md`). Poids cohérents avec la demande dans les 8 analyses menées par le LLM, aucun chiffre ni citation inventé dans l'état final. Deux corrections de `src/agent.py` : relance après une réponse vide du LLM, réponse sans outil gardée dans le journal. Deux reformulations du prompt essayées puis abandonnées. Reste ouvert : `chercher_passages` presque jamais demandé par le LLM, scénario 1 instable d'un passage à l'autre.
- 17 nov. : application refaite (branche `role4/app-v2`) : thème de quotidien financier sur papier saumon, relief 3D du portefeuille en Three.js (colonnes cliquables qui se réordonnent), poids modifiables en direct sans relancer l'agent, page découpée en fragment pour la fluidité, polices et Three.js copiés dans le dépôt (aucun appel externe).
- 18 nov. : brouillon de la partie 4 du rapport (`reports/rapport/partie_role4.md`) : théorie (LLM, prompt, appel d'outils, agents, RAG), architecture, choix du modèle, méthode et résultats de l'évaluation, analyse des erreurs, limites, plan des diapositives. Résultats avec les vrais modules à ajouter après la tâche 6.
- 18 nov. : README réécrit : présentation, rôles et modules, installation (Ollama, modèle d'embeddings), lancement de l'application, variables d'environnement, tests, reproduction des résultats (données, recherche, agent), données et mise en garde sur les cibles simulées, limites, équipe, licences.
