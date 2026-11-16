"""Vérifie que les outils branchés dans src/outils.py respectent le contrat.

Avec USE_MOCKS=1 on teste les fausses fonctions, avec USE_MOCKS=0 les vrais modules.
"""
import pandas as pd
import pytest

from src import outils
from src.contrat import (
    POIDS_PAR_DEFAUT,
    Extraction,
    Passage,
    Prediction,
    Projet,
    verifier_classement,
    verifier_poids,
)
from src.mocks import DOCUMENTS_EXEMPLE, PROJETS_EXEMPLE


def matrice_criteres() -> pd.DataFrame:
    """Construit la matrice des critères comme le fera l'agent."""
    lignes = []
    for projet in PROJETS_EXEMPLE.to_dict("records"):
        prediction = outils.predict(projet)
        extraction = outils.extract(DOCUMENTS_EXEMPLE[projet["id_projet"]])
        lignes.append({
            "id_projet": projet["id_projet"],
            "cout_total": projet["cout_total"],
            "duree_mois": projet["duree_mois"],
            "rentabilite_predite": prediction["rentabilite_predite"],
            "proba_risque": prediction["proba_risque"],
            **{c: extraction[c] for c in ("impact_env_social", "alignement_strategique", "faisabilite")},
        })
    return pd.DataFrame(lignes)


def test_projets_exemple_conformes():
    for projet in PROJETS_EXEMPLE.to_dict("records"):
        Projet(**projet)


def test_predict_respecte_le_contrat():
    for projet in PROJETS_EXEMPLE.to_dict("records"):
        Prediction(**outils.predict(projet))


def test_predict_est_stable():
    projet = PROJETS_EXEMPLE.iloc[0].to_dict()
    assert outils.predict(projet) == outils.predict(projet)


def test_extract_respecte_le_contrat():
    for texte in DOCUMENTS_EXEMPLE.values():
        Extraction(**outils.extract(texte))


def test_retrieve_respecte_le_contrat():
    resultats = outils.retrieve("étude d'impact du parc éolien", k=2)
    assert len(resultats) == 2
    for resultat in resultats:
        Passage(**resultat)
    assert resultats[0]["score"] >= resultats[1]["score"]


def test_detect_anomalies_ajoute_la_colonne():
    sortie = outils.detect_anomalies(PROJETS_EXEMPLE)
    assert len(sortie) == len(PROJETS_EXEMPLE)
    assert sortie["anomalie"].dtype == bool


def test_rank_respecte_le_contrat():
    classement = outils.rank(matrice_criteres(), POIDS_PAR_DEFAUT)
    verifier_classement(classement)
    assert len(classement) == len(PROJETS_EXEMPLE)


def test_poids_par_defaut_valides():
    verifier_poids(POIDS_PAR_DEFAUT)


def test_poids_invalides_refuses():
    with pytest.raises(ValueError):
        verifier_poids({"rentabilite_predite": 0.5})
    with pytest.raises(ValueError):
        verifier_poids({"critere_inconnu": 1.0})
