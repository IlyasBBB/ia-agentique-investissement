# Contrat commun

Format des données et des fonctions partagé par les quatre modules. Les sorties sont vérifiées par les modèles de `src/contrat.py` et par `tests/test_contrat.py`. Toute modification se fait ici et dans `src/contrat.py` en même temps. Dernière modification : 18 novembre 2026 (paramètre `documents` de `retrieve`).

## Fiche projet

| Colonne | Type | Signification | Exemple |
|---|---|---|---|
| `id_projet` | texte | Identifiant Banque mondiale, repris dans le nom du document associé (`data/docs/<id_projet>.txt`) | `P171266` |
| `nom` | texte | Nom du projet (en anglais) | Agriculture Resilience and Competitiveness Project |
| `secteur` | catégorie | Un des 11 secteurs (voir `data/README.md`) | Agriculture |
| `pays` | catégorie | Pays ou zone | Bosnia and Herzegovina |
| `region` | catégorie | Région de la Banque mondiale | Europe et Asie centrale |
| `cout_total` | nombre | Investissement en millions de dollars | 120 |
| `duree_mois` | nombre | Durée prévue en mois | 36 |
| `annee_approbation` | entier | Année d'approbation | 2022 |
| `type_financement` | catégorie | `IBRD`, `IDA`, `Dons` ou `Autre` | `IBRD` |
| `statut` | catégorie | `termine` ou `en_cours` (projets à évaluer, cibles vides) | `en_cours` |
| `rentabilite` | nombre | Cible de la régression, en %, **simulée**. Connue pour les projets terminés seulement | 11.5 |
| `risque` | catégorie | Cible de la classification, **simulée** : `faible`, `moyen`, `eleve`. Connue pour les projets terminés seulement | `moyen` |
| `impact_env_social` | note 1 à 5 | Critère qualitatif extrait par le rôle 3 | 4 |
| `alignement_strategique` | note 1 à 5 | Critère qualitatif extrait par le rôle 3 | 3 |
| `faisabilite` | note 1 à 5 | Critère qualitatif extrait par le rôle 3 | 4 |

## Critères du classement

| Critère | Source | Sens |
|---|---|---|
| `rentabilite_predite` | `predict` | à maximiser |
| `proba_risque` | `predict` | à minimiser |
| `cout_total` | fiche projet | à minimiser |
| `duree_mois` | fiche projet | à minimiser |
| `impact_env_social` | `extract` | à maximiser |
| `alignement_strategique` | `extract` | à maximiser |
| `faisabilite` | `extract` | à maximiser |

## Fonctions

### `predict(projet: dict) -> dict` (rôle 1, `src/supervise.py`)

Entrée : une fiche projet sans les cibles.
Sortie : `{"rentabilite_predite": float, "risque": "faible" | "moyen" | "eleve", "proba_risque": float entre 0 et 1}`.
`proba_risque` est la probabilité que le projet soit à risque élevé.

### `rank(matrice: DataFrame, poids: dict) -> DataFrame` (rôle 2, `src/mcda.py`)

Entrée : un projet par ligne avec `id_projet` et une colonne par critère. `poids` associe chaque critère à un poids, la somme vaut 1.
Sortie : le même tableau trié, avec `score` (entre 0 et 1) et `rang` (1 pour le meilleur).

### `detect_anomalies(df: DataFrame) -> DataFrame` (rôle 2, `src/mcda.py`)

Sortie : le même tableau avec une colonne `anomalie` (booléen).

### `extract(document: str) -> dict` (rôle 3, `src/nlp.py`)

Entrée : le texte d'un document de projet.
Sortie : `{"impact_env_social": int, "alignement_strategique": int, "faisabilite": int, "justifications": {critère: phrase}}`. Notes de 1 à 5.

### `retrieve(question: str, k: int = 3, documents: dict | None = None) -> list[dict]` (rôle 3, `src/nlp.py`)

Sortie : `k` éléments `{"id_projet": str, "passage": str, "score": float}`, du plus pertinent au moins pertinent.

`documents` (ajouté le 18 novembre, facultatif) associe `id_projet` au texte du document, comme pour `run_agent`. Sans `documents`, la recherche se fait dans les 2 000 documents de `data/docs/`. Avec `documents`, elle ne rend que des passages de ces documents-là : c'est ce que fait l'agent, pour ne citer que les projets analysés, y compris des fichiers importés dans l'application. Si les documents reçus sont déjà dans l'index de `data/docs/` (même identifiant, même texte), la recherche réutilise cet index sans le reconstruire.

### `run_agent(demande: str, projets: DataFrame, documents: dict | None = None, poids: dict | None = None) -> dict` (rôle 4, `src/agent.py`)

`documents` associe `id_projet` au texte du document.
Sortie : `{"classement": DataFrame, "recommandation": str, "etapes": list[dict]}`.
Chaque étape : `{"outil": str, "arguments": dict, "resultat": str, "duree_s": float}`.
