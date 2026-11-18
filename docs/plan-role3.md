# Plan du rôle 3 : NLP et RAG

Responsable : Ayoub ASSEFFAR. Module à livrer : `src/nlp.py` (`extract`, `retrieve`), pour le samedi 21 novembre 2026. Numérotation des tâches : feuille « Rôle 3 » de `docs/Plan_projet_ML_IA_agentique.xlsx`. Un commit par tâche, une branche `role3/tache-N` par pull request.

## Constats de départ

- 2 000 documents en anglais dans `data/docs/`, de 58 à 1 310 mots : en-tête (pays, secteur, coût…), objectif de développement (PDO), résumé. 526 documents n'ont pas de résumé.
- L'agent (`src/agent.py`) appelle `extract` sur chaque projet analysé, et `retrieve(question, 50)` sur tout le corpus, puis garde les `k` premiers passages des projets analysés.

## État des tâches

| N° | Tâche | État | Livrables |
|---|---|---|---|
| 1 | Constituer le corpus | fusionnée (PR n°1) | `data/docs/README.md` |
| 2 | Fiches de vérification | fusionnée (PR n°2) ; validation notée à la main ajoutée (PR n°8) | `reports/role3/grille_notation.md`, `reports/role3/annotations/jeux_extract.csv`, `test_reference.csv`, `validation_reference.csv` |
| 3 | Préparer le texte | fusionnée (PR n°3) | `src/nlp.py` (lecture, nettoyage, découpage), `notebooks/role3/01_exploration.ipynb` |
| 4 | Recherche TF-IDF | fusionnée (PR n°4) | `IndexTfidf`, `chercher` |
| 5 | Embeddings | fusionnée (PR n°5, correction en PR n°6) | `IndexEmbeddings`, `retrieve` |
| 6 | Comparer les deux recherches | fusionnée (PR n°6) | `notebooks/role3/03_retrieve.ipynb` : précision@3 de 0,75 (embeddings) contre 0,55 (TF-IDF) sur 25 requêtes de test |
| 7 | Extraction par LLM | fusionnée (PR n°7) | `scripts/annoter_llm_nlp.py`, `reports/role3/annotations/notes_llm.csv` (900 documents) |
| 8 | Analyse des erreurs | fusionnée (PR n°9) | `scripts/evaluer_extract_nlp.py`, `notebooks/role3/02_extract.ipynb` |
| 9 | Fonctions `extract` et `retrieve` | fusionnée (PR n°8 ; `retrieve` avec `documents` en PR n°10) : module complet sur `main` | `src/nlp.py`, `tests/test_nlp.py` |
| 10 | Partie du rapport | brouillon en relecture (final le 27) | `reports/role3/rapport_partie3.md` |

## `extract` : méthode retenue

- **Données** : 640 documents d'entraînement notés par le LLM local `qwen2.5:3b` ; 160 de validation et 100 de test notés à la main selon la grille. Séparation fixée avant tout modèle.
- **Modèle livré** : combinaison du solde de mots-clés de la grille et de la note attendue d'une régression logistique sur TF-IDF (apprise sur les notes du LLM), chaque signal centré-réduit sur la validation, puis recalé sur l'échelle des notes de référence.
- **Choix** : règle fixée avant d'ouvrir le test (meilleur kappa moyen sur la validation, le plus simple en cas d'égalité à 0,02 près). Combinaison : 0,649 en validation.
- **Test** (100 documents, ouvert une fois) : kappa pondéré moyen de 0,66, contre 0,62 pour les mots-clés seuls (écart non démontré), 0,52 pour le LLM et 0,49 pour TF-IDF seul. Environ 10 s pour les 2 000 documents.

## Décisions

- Annotations rangées dans `reports/role3/annotations/`.
- `retrieve` : embeddings par défaut, TF-IDF en repli ; un passage par projet ; question vide : liste vide sans charger l'index.
- Intégration avec l'agent : d'abord, l'agent demandait 50 passages et gardait les `k` premiers des projets analysés (commit `53814c6`). Le 18 novembre, le contrat a été étendu avec l'accord d'Ilyas : `retrieve(question, k, documents)` ne cherche que dans les documents reçus ; s'ils sont déjà dans l'index de `data/docs/`, l'index existant est réutilisé (environ 24 ms pour 500 projets) ; seuls les documents inconnus (fichiers importés) reçoivent un index construit à la volée.
- `extract` : la validation est notée à la main (option A, 18 novembre), car une validation notée par le LLM favorisait le modèle qui imite le LLM.
- `extract` : le modèle peut donner la note 1 par extrapolation alors que l'annotateur ne l'a jamais utilisée. Choix du 18 novembre (option 1) : ne pas corriger après l'ouverture du test, et signaler la limite dans le rapport.
