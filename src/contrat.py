"""Contrat commun : formats échangés entre les quatre modules.

Ce fichier est la version code de docs/contrat.md. Les deux doivent rester identiques.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

NIVEAUX_RISQUE = ("faible", "moyen", "eleve")

# Critère -> True s'il faut le maximiser, False s'il faut le minimiser.
CRITERES: dict[str, bool] = {
    "rentabilite_predite": True,
    "proba_risque": False,
    "cout_total": False,
    "duree_mois": False,
    "impact_env_social": True,
    "alignement_strategique": True,
    "faisabilite": True,
}

POIDS_PAR_DEFAUT: dict[str, float] = {
    "rentabilite_predite": 0.25,
    "proba_risque": 0.20,
    "cout_total": 0.10,
    "duree_mois": 0.05,
    "impact_env_social": 0.15,
    "alignement_strategique": 0.15,
    "faisabilite": 0.10,
}


class Projet(BaseModel):
    """Fiche d'un projet, sans les cibles."""

    id_projet: str
    nom: str
    secteur: str
    pays: str
    cout_total: float = Field(gt=0)
    duree_mois: float = Field(gt=0)


class Prediction(BaseModel):
    """Sortie de predict()."""

    rentabilite_predite: float
    risque: Literal["faible", "moyen", "eleve"]
    proba_risque: float = Field(ge=0, le=1)


class Extraction(BaseModel):
    """Sortie de extract()."""

    impact_env_social: int = Field(ge=1, le=5)
    alignement_strategique: int = Field(ge=1, le=5)
    faisabilite: int = Field(ge=1, le=5)
    justifications: dict[str, str]


class Passage(BaseModel):
    """Un élément de la sortie de retrieve()."""

    id_projet: str
    passage: str
    score: float


class Etape(BaseModel):
    """Une ligne du journal de l'agent."""

    outil: str
    arguments: dict
    resultat: str
    duree_s: float = Field(ge=0)


def verifier_poids(poids: dict[str, float]) -> None:
    """Lève ValueError si les poids ne correspondent pas au contrat."""
    inconnus = set(poids) - set(CRITERES)
    if inconnus:
        raise ValueError(f"Critères inconnus : {sorted(inconnus)}")
    if any(p < 0 for p in poids.values()):
        raise ValueError("Un poids ne peut pas être négatif.")
    if abs(sum(poids.values()) - 1) > 1e-6:
        raise ValueError(f"La somme des poids vaut {sum(poids.values()):.3f} au lieu de 1.")


def verifier_classement(df: pd.DataFrame) -> None:
    """Lève ValueError si la sortie de rank() ne respecte pas le contrat."""
    for colonne in ("id_projet", "score", "rang"):
        if colonne not in df.columns:
            raise ValueError(f"Colonne manquante dans le classement : {colonne}")
    if not df["score"].between(0, 1).all():
        raise ValueError("Les scores doivent être compris entre 0 et 1.")
    if sorted(df["rang"]) != list(range(1, len(df) + 1)):
        raise ValueError("Les rangs doivent aller de 1 au nombre de projets, sans doublon.")
    if not df["score"].is_monotonic_decreasing:
        raise ValueError("Le classement doit être trié du meilleur score au moins bon.")
