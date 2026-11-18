"""Sépare les documents en jeux de test, d'entraînement et de validation pour extract.

La séparation est faite **avant** toute annotation et tout modèle, une fois pour toutes (graine 42) :

- test : 100 documents tirés au hasard, stratifiés par secteur, notés à la main selon la grille.
  Ils ne servent qu'à la mesure finale ;
- entrainement (640) et validation (160) : notés par le LLM local, servent à apprendre
  et à régler le modèle TF-IDF ;
- reste : documents non annotés.

Usage : python scripts/separer_jeux_nlp.py   (écrit reports/role3/annotations/jeux_extract.csv)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parents[1]
GRAINE = 42
N_TEST, N_ENTRAINEMENT, N_VALIDATION = 100, 640, 160
SORTIE = RACINE / "reports" / "role3" / "annotations" / "jeux_extract.csv"


def main() -> None:
    """Écrit la répartition id_projet -> jeu."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if SORTIE.exists():
        raise SystemExit(f"{SORTIE} existe déjà : la séparation est fixée, on ne la refait pas.")
    projets = pd.read_csv(RACINE / "data" / "processed" / "projets.csv")[["id_projet", "secteur"]]
    reste, test = train_test_split(projets, test_size=N_TEST, stratify=projets.secteur, random_state=GRAINE)
    reste, annotes = train_test_split(reste, test_size=N_ENTRAINEMENT + N_VALIDATION,
                                      stratify=reste.secteur, random_state=GRAINE)
    entrainement, validation = train_test_split(annotes, test_size=N_VALIDATION,
                                                stratify=annotes.secteur, random_state=GRAINE)
    jeux = pd.concat([test.assign(jeu="test"), entrainement.assign(jeu="entrainement"),
                      validation.assign(jeu="validation"), reste.assign(jeu="non_annote")])
    jeux.sort_values("id_projet").to_csv(SORTIE, index=False)
    logging.info("Répartition :\n%s", jeux.jeu.value_counts().to_string())


if __name__ == "__main__":
    main()
