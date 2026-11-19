"""Tests du rôle 1 : séparation des jeux, prétraitement et fonction predict."""
import time

import numpy as np
import pandas as pd
import pytest

from src import supervise
from src.contrat import NIVEAUX_RISQUE, Prediction

# Avertissement sans conséquence de LightGBM dans un Pipeline (déjà filtré dans src/supervise.py,
# mais pytest remet les filtres à zéro pour chaque test).
pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

PROJETS = pd.read_csv(supervise.FICHIER_PROJETS)


def fiches(statut: str | None = None) -> list[dict]:
    """Fiches projet comme l'agent les envoie : toutes les colonnes sauf les cibles."""
    projets = PROJETS if statut is None else PROJETS[PROJETS["statut"] == statut]
    return [{k: v for k, v in r.items() if k not in ("rentabilite", "risque")}
            for r in projets.to_dict("records")]


def verifier_sortie(sortie: dict) -> None:
    """La sortie passe le modèle Prediction du contrat et a des valeurs plausibles."""
    Prediction(**sortie)
    assert set(sortie) == {"rentabilite_predite", "risque", "proba_risque"}
    assert sortie["risque"] in NIVEAUX_RISQUE
    assert 0 <= sortie["proba_risque"] <= 1
    assert np.isfinite(sortie["rentabilite_predite"])


# --- Séparation et prétraitement -------------------------------------------------------

def test_separation_couvre_les_termines_sans_chevauchement():
    jeux = supervise.charger_jeux()
    assert [len(jeux[j]) for j in ("entrainement", "validation", "test")] == [1050, 225, 225]
    ids = [set(jeu["id_projet"]) for jeu in jeux.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert set().union(*ids) == set(PROJETS.loc[PROJETS["statut"] == "termine", "id_projet"])
    for jeu in jeux.values():  # stratification : 40 / 40 / 20 % dans chaque jeu
        parts = jeu["risque"].value_counts(normalize=True)
        assert parts["faible"] == pytest.approx(0.4) and parts["eleve"] == pytest.approx(0.2)


def test_pretraitement_appris_sur_l_entrainement_seulement():
    jeux = supervise.charger_jeux()
    entrainement = jeux["entrainement"]
    pretraitement = supervise.creer_pretraitement().fit(entrainement[supervise.VARIABLES])
    moyenne = pretraitement.named_transformers_["cout"]["standardisation"].mean_[0]
    assert moyenne == pytest.approx(np.log10(entrainement["cout_total"]).mean())
    tous = pd.concat(jeux.values())
    assert moyenne != pytest.approx(np.log10(tous["cout_total"]).mean())


def test_pretraitement_supporte_inconnus_et_manquants():
    pretraitement = supervise.creer_pretraitement().fit(PROJETS[supervise.VARIABLES])
    lignes = pd.DataFrame([
        {"cout_total": None, "duree_mois": pd.NA, "annee_approbation": "2025", "secteur": "Spatial",
         "pays": "Atlantide", "region": None, "type_financement": "IDA"},
        {"cout_total": -3, "duree_mois": "abc", "annee_approbation": np.nan, "secteur": pd.NA,
         "pays": np.nan, "region": pd.NA, "type_financement": None},
    ])
    sortie = pretraitement.transform(lignes)
    assert not np.isnan(sortie).any()
    # catégorie inconnue ou manquante : aucune indicatrice allumée pour cette variable
    noms = pretraitement.get_feature_names_out()
    pays = [i for i, nom in enumerate(noms) if nom.startswith("pays_")]
    assert sortie[:, pays].sum() == 0


def test_bornes_rentabilite():
    rentabilite = pd.Series([6.0, 7.0, 8.0, 9.0, 10.0, 55.0])
    bas, haut = supervise.bornes_rentabilite(rentabilite, k=3)
    q1, q3 = rentabilite.quantile([0.25, 0.75])
    assert (bas, haut) == pytest.approx((q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)))
    assert supervise.dans_les_bornes(pd.DataFrame({"rentabilite": rentabilite}), (bas, haut)).tolist() == \
        [True] * 5 + [False]


# --- predict -----------------------------------------------------------------------------

def test_predict_respecte_le_contrat_sur_les_projets_en_cours():
    for fiche in fiches("en_cours"):
        verifier_sortie(supervise.predict(fiche))


def test_predict_ignore_les_cibles():
    fiche = fiches("termine")[0]
    avec_cibles = {**fiche, "rentabilite": 99.0, "risque": "eleve"}
    vides = {**fiche, "rentabilite": np.nan, "risque": None}
    assert supervise.predict(fiche) == supervise.predict(avec_cibles) == supervise.predict(vides)


@pytest.mark.parametrize("modification", [
    {"pays": "Atlantide"},
    {"secteur": "Spatial", "region": "Lune"},
    {"cout_total": None},
    {"duree_mois": np.nan},
    {"annee_approbation": pd.NA, "type_financement": None},
    {"cout_total": "beaucoup", "duree_mois": -5},
])
def test_predict_categorie_inconnue_et_valeur_manquante(modification):
    verifier_sortie(supervise.predict({**fiches("en_cours")[0], **modification}))


def test_predict_fiche_incomplete_ou_vide():
    verifier_sortie(supervise.predict({"id_projet": "X1", "nom": "Projet minimal",
                                       "secteur": "Énergie", "pays": "Morocco",
                                       "cout_total": 120, "duree_mois": 36}))
    verifier_sortie(supervise.predict({}))


def test_predict_precalcul_identique_au_calcul_complet():
    modeles = supervise.charger_modeles()
    for fiche in fiches()[::40]:
        calcul = supervise._predire_tableau(modeles, supervise._fiche_en_tableau(fiche))[0]
        assert supervise.predict(fiche) == calcul


def test_predict_rapide_sur_2000_projets():
    toutes = fiches()
    supervise.predict(toutes[0])  # chargement des modèles hors chronométrage
    debut = time.perf_counter()
    for fiche in toutes:
        supervise.predict(fiche)
    assert time.perf_counter() - debut < 2.0


def test_predict_risque_coherent_avec_les_donnees():
    """Sur les projets terminés, le modèle retrouve nettement mieux que le hasard la classe connue."""
    termines = PROJETS[PROJETS["statut"] == "termine"]
    predits = [supervise.predict(f)["risque"] for f in fiches("termine")]
    assert np.mean(np.array(predits) == termines["risque"].to_numpy()) > 0.5


def test_charger_modeles_reapprend_si_absent_puis_relit_le_fichier(tmp_path, monkeypatch):
    fichier = tmp_path / "supervise.joblib"
    monkeypatch.setattr(supervise, "_MODELES", None)
    modeles = supervise.charger_modeles(fichier)
    assert fichier.exists()
    assert modeles["projets_risque"] == 1500 and modeles["projets_rentabilite"] == 1490

    # Deuxième chargement : lu depuis le fichier, sans réapprendre.
    monkeypatch.setattr(supervise, "_MODELES", None)
    monkeypatch.setattr(supervise, "entrainer", lambda *a, **k: pytest.fail("réapprentissage inutile"))
    relus = supervise.charger_modeles(fichier)
    fiche = fiches("en_cours")[0]
    ligne = supervise._fiche_en_tableau(fiche)
    assert relus["rentabilite"].predict(ligne) == pytest.approx(modeles["rentabilite"].predict(ligne))
