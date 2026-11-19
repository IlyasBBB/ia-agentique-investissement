"""Rôle 2 : classement multicritère et détection d'anomalies.

Fonctions du contrat (docs/contrat.md) : rank(matrice, poids) et detect_anomalies(df).
Ce module contient aussi la méthode de référence du classement, somme_ponderee().

Le détecteur d'anomalies (Isolation Forest) est appris une fois sur les 2 000 projets de
data/processed/projets.csv, enregistré dans models/mcda_anomalies.joblib (hors Git) et
réappris automatiquement s'il manque. `python -m src.mcda` le réapprend.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from src.contrat import CRITERES, verifier_poids

journal = logging.getLogger(__name__)

RACINE = Path(__file__).resolve().parents[1]
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
FICHIER_DETECTEUR = RACINE / "models" / "mcda_anomalies.joblib"

# Part des projets de référence signalés. Fixée à 5 % (plan commun, tâche 4) avant toute
# mesure sur anomalies_injectees.csv, qui ne sert qu'à évaluer.
CONTAMINATION = 0.05
VARIABLES_ANOMALIES = ["log_cout", "log_duree", "log_cout_par_mois"]


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
    criteres, w = _criteres_et_poids(matrice, poids)
    score = normaliser(matrice, criteres) @ w if criteres else np.zeros(len(matrice))
    return pd.Series(np.clip(score, 0.0, 1.0), index=matrice.index, name="score")


def _criteres_et_poids(matrice: pd.DataFrame, poids: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """Vérifie les poids et rend les critères de poids non nul, avec leurs poids."""
    verifier_poids(poids)
    criteres = [c for c in CRITERES if poids.get(c, 0) > 0]
    manquants = [c for c in criteres if c not in matrice.columns]
    if manquants:
        raise ValueError(f"Critères absents de la matrice : {manquants}")
    return criteres, np.array([poids[c] for c in criteres], dtype=float)


def topsis(matrice: pd.DataFrame, poids: dict[str, float]) -> pd.Series:
    """Score TOPSIS : proximité relative à la solution idéale, entre 0 et 1.

    Les cinq étapes :
    1. normalisation vectorielle : chaque colonne est divisée par sa norme (racine de la
       somme des carrés), pour comparer des critères d'unités différentes ;
    2. pondération : chaque colonne est multipliée par son poids ;
    3. solution idéale (meilleure valeur de chaque critère : maximum ou minimum selon le
       sens) et solution anti-idéale (la pire) ;
    4. distance euclidienne de chaque projet à l'idéale (d+) et à l'anti-idéale (d-) ;
    5. score = d- / (d+ + d-) : 1 si le projet est l'idéal, 0 s'il est l'anti-idéal.

    Comme pour somme_ponderee, une valeur manquante reçoit la pire valeur du critère.
    Projets tous identiques (ou un seul projet) : d+ = d- = 0, score 0,5.
    Rend une série alignée sur l'index de `matrice`.
    """
    criteres, w = _criteres_et_poids(matrice, poids)
    n = len(matrice)
    if not criteres or n == 0:
        return pd.Series(np.full(n, 0.5), index=matrice.index, name="score")
    x = matrice[criteres].to_numpy(dtype=float)
    maximiser = np.array([CRITERES[c] for c in criteres])

    # Valeur manquante : pire valeur observée du critère (0 si la colonne est vide).
    with np.errstate(all="ignore"):
        pire = np.where(maximiser, np.nanmin(x, axis=0), np.nanmax(x, axis=0))
    pire = np.nan_to_num(pire, nan=0.0)
    x = np.where(np.isnan(x), pire, x)

    norme = np.sqrt((x ** 2).sum(axis=0))                      # étape 1
    r = x / np.where(norme > 0, norme, 1.0)
    v = r * w                                                  # étape 2
    ideale = np.where(maximiser, v.max(axis=0), v.min(axis=0))  # étape 3
    anti_ideale = np.where(maximiser, v.min(axis=0), v.max(axis=0))
    d_plus = np.sqrt(((v - ideale) ** 2).sum(axis=1))          # étape 4
    d_moins = np.sqrt(((v - anti_ideale) ** 2).sum(axis=1))
    total = d_plus + d_moins
    score = np.divide(d_moins, total, out=np.full(n, 0.5), where=total > 0)  # étape 5
    return pd.Series(np.clip(score, 0.0, 1.0), index=matrice.index, name="score")


def rank(matrice: pd.DataFrame, poids: dict[str, float]) -> pd.DataFrame:
    """Classe les projets du meilleur au moins bon (contrat : docs/contrat.md).

    Rend le même tableau (colonnes en plus gardées) avec `score` entre 0 et 1 et `rang`
    (1 pour le meilleur), trié par score décroissant. En cas d'égalité de score, l'ordre
    suit `id_projet`, pour que le classement soit toujours le même et les rangs uniques.
    Méthode actuelle : somme pondérée (somme_ponderee).
    """
    score = somme_ponderee(matrice, poids).to_numpy()
    ordre = np.lexsort((matrice["id_projet"].astype(str).to_numpy(), -score))
    sortie = matrice.iloc[ordre].copy()
    sortie["score"] = score[ordre]
    sortie["rang"] = np.arange(1, len(sortie) + 1)
    return sortie.reset_index(drop=True)


# --- Détection d'anomalies ------------------------------------------------------------

def variables_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Variables du détecteur, calculées à partir des seules colonnes obligatoires.

    log10 du coût, de la durée et du coût par mois : le log rend comparables un projet de
    5 et un de 500 millions. Un coût ou une durée nul, négatif ou manquant donne NaN.
    """
    manquantes = [c for c in ("cout_total", "duree_mois") if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes obligatoires absentes : {manquantes}")
    cout = pd.to_numeric(df["cout_total"], errors="coerce").where(lambda s: s > 0)
    duree = pd.to_numeric(df["duree_mois"], errors="coerce").where(lambda s: s > 0)
    return pd.DataFrame({
        "log_cout": np.log10(cout),
        "log_duree": np.log10(duree),
        "log_cout_par_mois": np.log10(cout / duree),
    }, index=df.index)


def entrainer_detecteur(projets: pd.DataFrame | None = None) -> dict:
    """Apprend l'Isolation Forest sur les projets de référence (les 2 000 de projets.csv).

    Aucune information sur les anomalies injectées n'est utilisée. Rend un dictionnaire
    prêt à enregistrer : le modèle, les médianes (pour remplacer les manquants) et la
    version de scikit-learn.
    """
    if projets is None:
        projets = pd.read_csv(FICHIER_PROJETS)
    x = variables_anomalies(projets)
    medianes = x.median()
    modele = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
    modele.fit(x.fillna(medianes))
    journal.info("Détecteur d'anomalies appris sur %d projets.", len(x))
    return {"modele": modele, "medianes": medianes, "variables": VARIABLES_ANOMALIES,
            "contamination": CONTAMINATION, "version_sklearn": sklearn.__version__}


_DETECTEUR: dict | None = None  # chargé une seule fois, au premier appel


def charger_detecteur(fichier: Path = FICHIER_DETECTEUR) -> dict:
    """Charge le détecteur enregistré, ou l'apprend et l'enregistre s'il manque ou est périmé."""
    global _DETECTEUR
    if _DETECTEUR is not None:
        return _DETECTEUR
    detecteur = None
    if fichier.exists():
        try:
            detecteur = joblib.load(fichier)
        except Exception as erreur:  # fichier abîmé : on réapprend
            journal.warning("Détecteur illisible (%s), réapprentissage.", erreur)
    if (detecteur is None or detecteur.get("version_sklearn") != sklearn.__version__
            or detecteur.get("contamination") != CONTAMINATION
            or detecteur.get("variables") != VARIABLES_ANOMALIES):
        detecteur = entrainer_detecteur()
        fichier.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(detecteur, fichier)
        journal.info("Détecteur enregistré dans %s.", fichier.name)
    _DETECTEUR = detecteur
    return detecteur


def score_anomalie(df: pd.DataFrame) -> pd.Series:
    """Score d'anomalie de chaque projet : plus il est grand, plus le projet est atypique.

    Positif au-delà du seuil fixé par la contamination (projet signalé).
    """
    detecteur = charger_detecteur()
    x = variables_anomalies(df)
    if len(x) == 0:
        return pd.Series(dtype=float, index=df.index, name="score_anomalie")
    score = -detecteur["modele"].decision_function(x.fillna(detecteur["medianes"]))
    return pd.Series(score, index=df.index, name="score_anomalie")


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne booléenne `anomalie` (contrat : docs/contrat.md).

    Le détecteur est appris une fois sur les 2 000 projets de référence puis appliqué tel
    quel, même à un seul projet. Un coût ou une durée nul ou négatif est signalé d'office.
    """
    sortie = df.copy()
    impossible = pd.Series(False, index=df.index)
    for colonne in ("cout_total", "duree_mois"):
        if colonne in df.columns:
            impossible |= pd.to_numeric(df[colonne], errors="coerce") <= 0
    sortie["anomalie"] = (score_anomalie(df) > 0) | impossible
    return sortie


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FICHIER_DETECTEUR.unlink(missing_ok=True)
    charger_detecteur()
