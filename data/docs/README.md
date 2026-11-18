# Corpus de documents (rôle 3)

Un fichier texte par projet, nommé par son identifiant : `data/docs/<id_projet>.txt`, par exemple `P171266.txt`. Les 2 000 identifiants sont ceux de `data/processed/projets.csv`.

## Source

Projets réels de la Banque mondiale, « Projects & Operations » (https://projects.worldbank.org), licence Creative Commons Attribution 4.0 (CC BY 4.0). Téléchargés le 16 novembre 2026 avec `scripts/telecharger_donnees.py`, mis en forme avec `scripts/preparer_donnees.py`. Le détail (filtres, tirage, citation) est dans `data/README.md`.

Le texte de l'objectif et du résumé est recopié tel quel de la Banque mondiale, en anglais. L'en-tête est rédigé par l'équipe à partir de `projets.csv`.

## Format d'un document

```
Projet : <nom du projet>
Identifiant : <id_projet>
Pays : <pays> (<région>)
Secteur : <secteur>
Coût total : <millions de dollars>
Durée prévue : <mois>
Année d'approbation : <année>
Statut : termine | en_cours

Objectif de développement (PDO) :
<objectif de développement du projet, en anglais>

Résumé :
<résumé du projet, en anglais, ou « Non disponible. »>
```

## Chiffres

| | Valeur |
|---|---|
| Documents | 2 000 (1 500 projets terminés, 500 en cours) |
| Taille totale | 3,9 Mo |
| Documents sans résumé | 526 (26 %), surtout des projets anciens |
| Objectifs entièrement en majuscules | 285 (14 %), surtout des projets anciens |
| Longueur de l'objectif et du résumé | 211 mots en médiane, de 11 à 1 259 |
| Passages pour la recherche | 8 293 (3 phrases, 180 mots au plus) |

Analyse détaillée : `notebooks/role3/01_exploration.ipynb`.

## Utilisation par le rôle 3

- Seuls l'objectif et le résumé sont analysés (`nlp.corps`). L'en-tête répète des chiffres déjà utilisés par les autres rôles.
- Le texte est nettoyé avant usage (`nlp.nettoyer`) : mots coupés par un tiret de fin de ligne, espaces en trop, phrases en majuscules.
- `retrieve` cherche dans des passages de 3 phrases découpés dans ces textes (`nlp.decouper_passages`).
- `extract` note chaque texte sur trois critères qualitatifs. Les notes de référence et les jeux d'entraînement et de test sont dans `reports/role3/annotations/`.

## Mises en garde

- Le coût et la durée de l'en-tête viennent de `projets.csv` : pour les 40 projets à anomalie fabriquée, ils sont modifiés (voir `data/README.md`). Le texte, lui, n'est jamais modifié.
- Les textes ne dépendent pas des cibles simulées (`rentabilite`, `risque`) : les notes qualitatives n'aident pas à les prédire.
