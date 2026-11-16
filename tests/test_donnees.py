"""Vérifie le jeu de données préparé par scripts/preparer_donnees.py.

Ces tests sont sautés tant que data/processed/projets.csv n'existe pas.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.contrat import NIVEAUX_RISQUE, Projet

RACINE = Path(__file__).resolve().parents[1]
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
FICHIER_ANOMALIES = RACINE / "data" / "processed" / "anomalies_injectees.csv"
DOSSIER_DOCS = RACINE / "data" / "docs"


@pytest.fixture(scope="module")
def projets() -> pd.DataFrame:
    if not FICHIER_PROJETS.exists():
        pytest.skip("data/processed/projets.csv absent : lancer scripts/preparer_donnees.py")
    return pd.read_csv(FICHIER_PROJETS)


def test_taille_et_identifiants(projets):
    assert len(projets) >= 500
    assert projets["id_projet"].is_unique


def test_colonnes_et_types(projets):
    for colonne in ("id_projet", "nom", "secteur", "pays", "cout_total", "duree_mois", "rentabilite", "risque", "statut"):
        assert colonne in projets.columns
    fiche = ["id_projet", "nom", "secteur", "pays", "cout_total", "duree_mois"]
    assert projets[fiche].notna().all().all()
    for projet in projets[fiche].to_dict("records"):
        Projet(**projet)  # lève une erreur si un type ou une valeur ne respecte pas le contrat
    assert set(projets["statut"]) == {"termine", "en_cours"}


def test_cibles_connues_pour_les_projets_termines_seulement(projets):
    en_cours = projets["statut"] == "en_cours"
    assert (projets["rentabilite"].isna() == en_cours).all()
    assert (projets["risque"].isna() == en_cours).all()
    assert set(projets["risque"].dropna()) == set(NIVEAUX_RISQUE)


def test_un_document_par_projet(projets):
    manquants = [i for i in projets["id_projet"] if not (DOSSIER_DOCS / f"{i}.txt").exists()]
    assert manquants == []


def test_anomalies_referencent_des_projets_existants(projets):
    anomalies = pd.read_csv(FICHIER_ANOMALIES)
    assert len(anomalies) > 0
    assert anomalies["id_projet"].isin(projets["id_projet"]).all()
