# Plan du rôle 3 : NLP et RAG

Responsable : Ayoub ASSEFFAR. Module à livrer : `src/nlp.py` (`extract`, `retrieve`), pour le samedi 21 novembre 2026.

## Constats de départ

- 2 000 documents en anglais dans `data/docs/`, de 58 à 1 310 mots (environ 270 en moyenne) : en-tête (pays, secteur, coût…), objectif de développement (PDO), résumé. 526 documents n'ont pas de résumé.
- L'agent (`src/agent.py`) appelle `extract` sur chaque projet analysé, et `retrieve(question, k)` sur tout le corpus, puis ne garde que les passages des projets analysés.

## `extract` : trois notes de 1 à 5

Pas de vérité terrain : on la construit.

1. **Grille de notation** (`reports/role3/grille_notation.md`) : définition de chaque note pour chaque critère.
2. **Jeu de test** : 100 documents tirés au hasard, stratifiés par secteur, notés selon la grille, indépendamment du LLM (`reports/role3/annotations/test_reference.csv`). Jamais utilisé pour l'entraînement.
3. **Jeu d'entraînement et de validation** : environ 800 autres documents notés automatiquement par `qwen2.5:3b` (Ollama, local) avec la même grille, une fois, hors démo (`reports/role3/annotations/entrainement_llm.csv`).

Modèles comparés sur le jeu de test (exactitude, erreur absolue moyenne, kappa pondéré quadratique) :

- référence : mots-clés par critère ;
- avancé : TF-IDF + régression logistique, un modèle par critère, vocabulaire appris sur l'entraînement seulement ;
- comparaison : le LLM lui-même (trop lent pour la démo).

Module livré : TF-IDF + régression logistique (quelques millisecondes par document). Justification = phrase du document qui pèse le plus dans la note. Texte vide ou très court : note neutre 3.

## `retrieve` : passages pertinents

- Index : PDO et résumé découpés en passages d'environ 3 phrases, construit au premier appel, gardé en mémoire et sur disque dans `models/` (ignoré par Git), reconstruit si les documents changent.
- Référence : TF-IDF + similarité cosinus. Avancé : embeddings `all-MiniLM-L6-v2` (local, processeur). Repli automatique sur TF-IDF si le modèle est absent.
- Évaluation : 40 requêtes écrites à la main avec leurs projets attendus (15 de réglage, 25 de test), précision@3, rappel@5, MRR.

## Calendrier et état

| Échéance | Tâche | État |
|---|---|---|
| 17 nov. | Environnement, lecture du contrat, plan | fait |
| 18 nov. | Analyse exploratoire des textes (`notebooks/role3/01_exploration.ipynb`), grille de notation | fait |
| 18 nov. | `retrieve` (TF-IDF, embeddings, hybride), 40 requêtes jugées, évaluation (`notebooks/role3/03_retrieve.ipynb`) | fait : embeddings retenus, précision@3 de 0,75 contre 0,55 pour TF-IDF sur 25 requêtes de test |
| 19 nov. | Jeu de test annoté (100 documents), annotations LLM (800 documents) | à faire |
| 19-20 nov. | `extract` : mots-clés, TF-IDF, comparaison, mots influents, analyse des erreurs | à faire |
| 20 nov. | `tests/test_nlp.py` (contrat, texte vide, vitesse), essai avec `USE_MOCKS=0` | à faire |
| 21 nov. | Branche poussée, pull request préparée | à faire |
| 23-27 nov. | Partie théorique et résultats pour le rapport (`reports/role3/`) | à faire |

## Décisions

- Annotations rangées dans `reports/role3/annotations/`.
- Intégration avec l'agent : le contrat de `retrieve` ne change pas. Proposition transmise à Ilyas : l'agent appelle `retrieve` avec un `k` plus grand (par exemple 50), puis garde les 3 premiers passages des projets analysés.
