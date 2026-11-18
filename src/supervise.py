"""Rôle 1 : apprentissage supervisé (rentabilité et risque des projets).

Ce module contient :
- le chargement des jeux d'entraînement, de validation et de test (séparation fixée
  par scripts/separer_jeux_supervise.py, dans reports/role1/separation.csv) ;
- le prétraitement commun à tous les modèles (`creer_pretraitement`).

Le prétraitement est un objet scikit-learn à ajuster sur le jeu d'entraînement
seulement : placé dans un `Pipeline` avec le modèle, `fit` n'apprend les moyennes,
écarts-types et catégories que sur les données qu'on lui donne, ce qui évite les fuites.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

journal = logging.getLogger(__name__)

RACINE = Path(__file__).resolve().parents[1]
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
FICHIER_SEPARATION = RACINE / "reports" / "role1" / "separation.csv"

# Entrées du modèle. Exclus : id_projet, nom (identifiants), statut, et les cibles.
VARIABLES_COUT = ["cout_total"]  # passé au log10 (distribution très asymétrique)
VARIABLES_NUMERIQUES = ["duree_mois", "annee_approbation"]
VARIABLES_CATEGORIELLES = ["secteur", "pays", "region", "type_financement"]
VARIABLES = VARIABLES_COUT + VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES

CIBLE_REGRESSION = "rentabilite"
CIBLE_CLASSIFICATION = "risque"


def en_nombres(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convertit en nombres décimaux ; None, pd.NA ou un texte non numérique deviennent NaN."""
    return pd.DataFrame(x).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def en_textes(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convertit en textes ; None et pd.NA deviennent NaN, reconnu par SimpleImputer."""
    tableau = pd.DataFrame(x).astype(object)
    return tableau.where(tableau.notna(), np.nan).to_numpy(dtype=object)


def log10_positif(x: np.ndarray) -> np.ndarray:
    """log10 des valeurs strictement positives ; une valeur nulle ou négative devient manquante."""
    x = en_nombres(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), np.nan)


def creer_pretraitement() -> ColumnTransformer:
    """Prétraitement non ajusté : log10 du coût, imputation, standardisation, indicatrices.

    Une valeur manquante peut arriver sous forme de NaN, None ou pd.NA : elle est d'abord
    ramenée à NaN, que SimpleImputer sait reconnaître.

    - coût : log10, puis valeur manquante remplacée par la médiane, puis centrage-réduction ;
    - durée et année : médiane pour les manquants, puis centrage-réduction ;
    - secteur, pays, région, financement : une colonne 0/1 par catégorie vue à
      l'entraînement. Une catégorie jamais vue (pays inconnu, par exemple) donne des
      zéros partout au lieu de faire planter. Un manquant est remplacé par « inconnu »,
      donc traité de la même façon : on n'invente pas de pays (remplacer par le pays le
      plus fréquent lui donnerait à tort l'effet de ce pays).
    """
    cout = Pipeline([
        ("log10", FunctionTransformer(log10_positif, feature_names_out="one-to-one")),
        ("imputation", SimpleImputer(strategy="median")),
        ("standardisation", StandardScaler()),
    ])
    numeriques = Pipeline([
        ("nombres", FunctionTransformer(en_nombres, feature_names_out="one-to-one")),
        ("imputation", SimpleImputer(strategy="median")),
        ("standardisation", StandardScaler()),
    ])
    categorielles = Pipeline([
        ("textes", FunctionTransformer(en_textes, feature_names_out="one-to-one")),
        ("imputation", SimpleImputer(strategy="constant", fill_value="inconnu")),
        ("indicatrices", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("cout", cout, VARIABLES_COUT),
        ("numeriques", numeriques, VARIABLES_NUMERIQUES),
        ("categorielles", categorielles, VARIABLES_CATEGORIELLES),
    ], verbose_feature_names_out=False)


def bornes_rentabilite(rentabilite: pd.Series, k: float = 3.0) -> tuple[float, float]:
    """Bornes [Q1 − k·IQR ; Q3 + k·IQR] de la rentabilité, à calculer sur l'entraînement seulement.

    IQR = Q3 − Q1 (écart interquartile). Les quartiles bougent très peu à cause de
    quelques valeurs extrêmes, donc la règle n'est pas faussée par ce qu'elle doit repérer.
    """
    q1, q3 = rentabilite.quantile([0.25, 0.75])
    ecart = q3 - q1
    return float(q1 - k * ecart), float(q3 + k * ecart)


def dans_les_bornes(projets: pd.DataFrame, bornes: tuple[float, float]) -> pd.Series:
    """Vrai pour les projets dont la rentabilité est dans les bornes (règle apprise ailleurs)."""
    return projets[CIBLE_REGRESSION].between(*bornes)


def charger_jeux(fichier_projets: Path = FICHIER_PROJETS,
                 fichier_separation: Path = FICHIER_SEPARATION) -> dict[str, pd.DataFrame]:
    """Rend les projets terminés de chaque jeu : {"entrainement": ..., "validation": ..., "test": ...}.

    Chaque tableau garde toutes les colonnes de projets.csv (entrées, cibles, identifiant).
    """
    projets = pd.read_csv(fichier_projets)
    separation = pd.read_csv(fichier_separation)
    projets = projets.merge(separation, on="id_projet", how="inner")
    jeux = {nom: projets[projets["jeu"] == nom].drop(columns="jeu").reset_index(drop=True)
            for nom in ("entrainement", "validation", "test")}
    journal.info("Jeux chargés : %s", {nom: len(jeu) for nom, jeu in jeux.items()})
    return jeux
