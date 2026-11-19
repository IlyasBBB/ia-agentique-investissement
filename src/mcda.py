"""Rôle 2 : classement multicritère et détection d'anomalies.

Fonctions du contrat (docs/contrat.md) : rank(matrice, poids) et detect_anomalies(df).
Ce module contient aussi la méthode de référence du classement, somme_ponderee().
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.contrat import CRITERES, verifier_poids

journal = logging.getLogger(__name__)


def normaliser(matrice: pd.DataFrame, criteres: list[str]) -> np.ndarray:
    """Ramène chaque critère entre 0 et 1, 1 étant toujours la meilleure valeur.

    Min-max sur les projets reçus : le meilleur projet du critère vaut 1, le moins bon 0.
    Les critères à minimiser (coût, durée, risque) sont inversés. Une colonne constante
    vaut 0,5 pour tous (aucun projet n'y est meilleur qu'un autre, pas de division par zéro).
    Une valeur manquante reçoit 0, la pire note : un critère inconnu ne doit pas aider un projet.
    """
    valeurs = matrice[criteres].to_numpy(dtype=float)
    if valeurs.shape[0] == 0:
        return valeurs
    with np.errstate(all="ignore"):  # colonnes entièrement vides : NaN traités plus bas
        mini = np.nanmin(valeurs, axis=0)
        maxi = np.nanmax(valeurs, axis=0)
    etendue = maxi - mini
    constante = ~(etendue > 0)  # vrai aussi pour une colonne entièrement vide
    normalise = (valeurs - mini) / np.where(constante, 1.0, etendue)
    normalise[:, constante] = 0.5
    a_minimiser = np.array([not CRITERES[c] for c in criteres])
    normalise[:, a_minimiser] = 1.0 - normalise[:, a_minimiser]
    return np.nan_to_num(normalise, nan=0.0)


def somme_ponderee(matrice: pd.DataFrame, poids: dict[str, float]) -> pd.Series:
    """Score de référence : somme pondérée des critères ramenés entre 0 et 1.

    Un critère absent de `poids` compte pour 0. Le score est entre 0 et 1 car les poids
    somment à 1. Rend une série alignée sur l'index de `matrice`.
    """
    verifier_poids(poids)
    criteres = [c for c in CRITERES if poids.get(c, 0) > 0]
    manquants = [c for c in criteres if c not in matrice.columns]
    if manquants:
        raise ValueError(f"Critères absents de la matrice : {manquants}")
    w = np.array([poids[c] for c in criteres], dtype=float)
    score = normaliser(matrice, criteres) @ w if criteres else np.zeros(len(matrice))
    return pd.Series(np.clip(score, 0.0, 1.0), index=matrice.index, name="score")
