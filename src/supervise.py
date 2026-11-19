"""Rôle 1 : apprentissage supervisé (rentabilité et risque des projets).

Ce module contient :
- le chargement des jeux d'entraînement, de validation et de test (séparation fixée
  par scripts/separer_jeux_supervise.py, dans reports/role1/separation.csv) ;
- le prétraitement commun à tous les modèles (`creer_pretraitement`) ;
- la fonction du contrat, `predict(projet)`, et l'apprentissage des modèles livrés.

Modèles livrés (choisis à la tâche 8, notebooks/role1/02_modeles.ipynb) : Ridge pour la
rentabilité, LightGBM pour le risque. Ils sont réappris avec les mêmes réglages sur les
1 500 projets terminés, enregistrés dans models/supervise.joblib (hors Git) et chargés
une seule fois. Pour les réapprendre : python -m src.supervise

Rapidité : prétraiter une seule ligne coûte environ 15 ms (surtout les vérifications de
scikit-learn), soit 30 s pour les 2 000 projets appelés un par un par l'agent. Au
chargement, on calcule donc en une fois (moins d'une seconde) les prédictions des projets de
projets.csv, rangées selon les valeurs de leurs entrées ; `predict` les relit. Un projet
inconnu (fichier importé, valeur modifiée) passe par le calcul complet, puis est gardé.

Le prétraitement est un objet scikit-learn à ajuster sur le jeu d'entraînement
seulement : placé dans un `Pipeline` avec le modèle, `fit` n'apprend les moyennes,
écarts-types et catégories que sur les données qu'on lui donne, ce qui évite les fuites.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

journal = logging.getLogger(__name__)

RACINE = Path(__file__).resolve().parents[1]
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
FICHIER_SEPARATION = RACINE / "reports" / "role1" / "separation.csv"
FICHIER_MODELES = RACINE / "models" / "supervise.joblib"

# Entrées du modèle. Exclus : id_projet, nom (identifiants), statut, et les cibles.
VARIABLES_COUT = ["cout_total"]  # passé au log10 (distribution très asymétrique)
VARIABLES_NUMERIQUES = ["duree_mois", "annee_approbation"]
VARIABLES_CATEGORIELLES = ["secteur", "pays", "region", "type_financement"]
VARIABLES = VARIABLES_COUT + VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES

CIBLE_REGRESSION = "rentabilite"
CIBLE_CLASSIFICATION = "risque"
NIVEAUX_RISQUE = ("faible", "moyen", "eleve")

# Réglages choisis par validation croisée sur l'entraînement (tâches 6 et 7), retenus à la tâche 8.
RIDGE_ALPHA = 10 ** 0.5  # environ 3,16
LGBM_REGLAGES = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 7,
                 "min_child_samples": 30, "class_weight": "balanced"}
K_IQR = 3.0
GRAINE = 42

# LightGBM signale à tort des noms de colonnes différents entre fit et predict dans un Pipeline.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


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


def creer_modele_rentabilite() -> Pipeline:
    """Pipeline non ajusté du modèle de rentabilité : prétraitement puis Ridge."""
    return Pipeline([("pretraitement", creer_pretraitement()), ("modele", Ridge(alpha=RIDGE_ALPHA))])


def creer_modele_risque() -> Pipeline:
    """Pipeline non ajusté du modèle de risque : prétraitement puis LightGBM."""
    return Pipeline([("pretraitement", creer_pretraitement()),
                     ("modele", LGBMClassifier(**LGBM_REGLAGES, random_state=GRAINE, verbose=-1, n_jobs=1))])


def entrainer(projets: pd.DataFrame | None = None) -> dict:
    """Apprend les deux modèles livrés sur les projets terminés (les 1 500 par défaut).

    La règle des rentabilités aberrantes (bornes IQR) est recalculée sur ces projets ;
    le modèle de rentabilité apprend sur les projets dans les bornes, celui du risque
    sur tous. Rend un dictionnaire prêt à enregistrer avec joblib.
    """
    if projets is None:
        projets = pd.read_csv(FICHIER_PROJETS)
    termines = projets[projets[CIBLE_REGRESSION].notna() & projets[CIBLE_CLASSIFICATION].notna()]
    bornes = bornes_rentabilite(termines[CIBLE_REGRESSION], K_IQR)
    dans = termines[dans_les_bornes(termines, bornes)]
    rentabilite = creer_modele_rentabilite().fit(dans[VARIABLES], dans[CIBLE_REGRESSION])
    risque = creer_modele_risque().fit(termines[VARIABLES], termines[CIBLE_CLASSIFICATION])
    journal.info("Modèles appris : rentabilité sur %d projets (bornes %.1f ; %.1f), risque sur %d projets.",
                 len(dans), *bornes, len(termines))
    return {"rentabilite": rentabilite, "risque": risque, "bornes": bornes,
            "projets_rentabilite": len(dans), "projets_risque": len(termines),
            "version_sklearn": sklearn.__version__}


def enregistrer(modeles: dict, fichier: Path = FICHIER_MODELES) -> None:
    """Enregistre les modèles avec joblib."""
    fichier.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modeles, fichier)
    journal.info("Modèles enregistrés : %s", fichier.name)


_MODELES: dict | None = None  # chargés une seule fois, au premier appel de predict
_PREDICTIONS: dict[tuple, dict] = {}  # entrées du projet -> prédiction déjà calculée


def charger_modeles(fichier: Path = FICHIER_MODELES) -> dict:
    """Charge les modèles enregistrés ; s'ils sont absents ou illisibles, les réapprend et les enregistre."""
    global _MODELES
    if _MODELES is not None:
        return _MODELES
    modeles = None
    if fichier.exists():
        try:
            modeles = joblib.load(fichier)
            if modeles.get("version_sklearn") != sklearn.__version__:
                journal.warning("Modèles enregistrés avec scikit-learn %s (installé : %s) : réapprentissage.",
                                modeles.get("version_sklearn"), sklearn.__version__)
                modeles = None
        except Exception as erreur:  # fichier abîmé ou incompatible : on réapprend
            journal.warning("Lecture de %s impossible (%s) : réapprentissage.", fichier.name, erreur)
            modeles = None
    if modeles is None:
        journal.info("Apprentissage des modèles du rôle 1 (quelques secondes)...")
        modeles = entrainer()
        enregistrer(modeles, fichier)
    _MODELES = modeles
    _PREDICTIONS.clear()
    if FICHIER_PROJETS.exists():
        projets = pd.read_csv(FICHIER_PROJETS)
        tableau = projets.reindex(columns=VARIABLES)
        for ligne, prediction in zip(tableau.to_dict("records"), _predire_tableau(modeles, tableau)):
            _PREDICTIONS[_cle(ligne)] = prediction
        journal.info("Prédictions calculées d'avance pour %d projets.", len(_PREDICTIONS))
    return modeles


def _fiche_en_tableau(projet: dict) -> pd.DataFrame:
    """Une ligne avec les seules entrées du modèle ; une entrée absente devient manquante."""
    return pd.DataFrame([{variable: projet.get(variable, np.nan) for variable in VARIABLES}])


def _cle(projet: dict) -> tuple:
    """Clé de rangement d'une prédiction : les valeurs des entrées, écrites en texte.

    Deux fiches de même clé ont les mêmes entrées, donc la même prédiction. Une clé qui
    ne correspond à rien (par exemple 106 au lieu de 106.0) coûte seulement un calcul complet.
    """
    return tuple(str(projet.get(variable)) for variable in VARIABLES)


def _predire_tableau(modeles: dict, tableau: pd.DataFrame) -> list[dict]:
    """Prédictions du contrat pour chaque ligne d'un tableau d'entrées."""
    rentabilites = modeles["rentabilite"].predict(tableau)
    probas = modeles["risque"].predict_proba(tableau)
    classes = list(modeles["risque"].classes_)
    colonne_eleve = classes.index("eleve")
    return [{
        "rentabilite_predite": round(float(rentabilite), 2),
        "risque": str(classes[int(np.argmax(proba))]),
        "proba_risque": float(min(1.0, max(0.0, proba[colonne_eleve]))),
    } for rentabilite, proba in zip(rentabilites, probas)]


def predict(projet: dict) -> dict:
    """Prédit la rentabilité (en %) et le risque d'un projet.

    Entrée : fiche projet (colonnes de projets.csv) ; seules les entrées du modèle sont lues,
    les cibles éventuelles sont ignorées. Une valeur manquante ou une catégorie jamais vue
    (pays, secteur) est acceptée.
    Sortie : {"rentabilite_predite": float, "risque": "faible" | "moyen" | "eleve",
    "proba_risque": probabilité du risque élevé, entre 0 et 1}.
    """
    modeles = charger_modeles()
    cle = _cle(projet)
    if cle not in _PREDICTIONS:
        _PREDICTIONS[cle] = _predire_tableau(modeles, _fiche_en_tableau(projet))[0]
    return dict(_PREDICTIONS[cle])


if __name__ == "__main__":
    # Passer par le module importé (et non __main__) : joblib enregistre les fonctions du
    # prétraitement sous le nom src.supervise, sinon elles ne se rechargeraient pas.
    from src import supervise as module

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    module.enregistrer(module.entrainer())
