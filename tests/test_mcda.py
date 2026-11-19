"""Tests du rôle 2 : src/mcda.py."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.contrat import CRITERES, POIDS_PAR_DEFAUT, verifier_classement
from src.mcda import normaliser, rank, somme_ponderee


def matrice_exemple() -> pd.DataFrame:
    """Quatre projets, calculables à la main."""
    return pd.DataFrame({
        "id_projet": ["A", "B", "C", "D"],
        "rentabilite_predite": [10.0, 5.0, 15.0, 10.0],
        "proba_risque": [0.2, 0.1, 0.9, 0.5],
        "cout_total": [100.0, 50.0, 200.0, 150.0],
        "duree_mois": [24, 12, 48, 36],
        "impact_env_social": [3, 5, 1, 3],
        "alignement_strategique": [4, 2, 5, 3],
        "faisabilite": [4, 4, 4, 4],
    })


# --- Tâche 5 : somme pondérée -------------------------------------------------------

def test_normaliser_entre_0_et_1_et_inverse_les_criteres_a_minimiser():
    m = matrice_exemple()
    n = normaliser(m, ["rentabilite_predite", "cout_total"])
    assert n[:, 0].tolist() == [0.5, 0.0, 1.0, 0.5]  # à maximiser
    assert n[:, 1].tolist() == pytest.approx([2 / 3, 1.0, 0.0, 1 / 3])  # à minimiser : le moins cher vaut 1


def test_normaliser_colonne_constante_sans_division_par_zero():
    n = normaliser(matrice_exemple(), ["faisabilite"])
    assert n[:, 0].tolist() == [0.5] * 4


def test_normaliser_valeur_manquante_recoit_la_pire_note():
    m = matrice_exemple()
    m.loc[0, "proba_risque"] = np.nan
    n = normaliser(m, ["proba_risque"])
    assert n[0, 0] == 0.0
    assert not np.isnan(n).any()


def test_somme_ponderee_calcul_a_la_main():
    poids = {"rentabilite_predite": 0.5, "cout_total": 0.5}
    score = somme_ponderee(matrice_exemple(), poids)
    # A : 0.5*0.5 + 0.5*2/3 ; B : 0 + 0.5*1 ; C : 0.5*1 + 0 ; D : 0.5*0.5 + 0.5*1/3
    assert score.tolist() == pytest.approx([0.25 + 1 / 3, 0.5, 0.5, 0.25 + 1 / 6])


def test_somme_ponderee_un_seul_critere_donne_l_ordre_de_ce_critere():
    score = somme_ponderee(matrice_exemple(), {"proba_risque": 1.0})
    assert score.idxmax() == 1  # B, le moins risqué
    assert score.idxmin() == 2  # C, le plus risqué


def test_somme_ponderee_scores_entre_0_et_1_avec_les_poids_par_defaut():
    score = somme_ponderee(matrice_exemple(), POIDS_PAR_DEFAUT)
    assert score.between(0, 1).all()
    assert list(score.index) == list(matrice_exemple().index)


def test_somme_ponderee_poids_nul_et_colonne_absente_acceptes():
    m = matrice_exemple().drop(columns="duree_mois")
    poids = {**POIDS_PAR_DEFAUT, "duree_mois": 0.0, "rentabilite_predite": 0.30}
    assert somme_ponderee(m, poids).between(0, 1).all()


def test_somme_ponderee_refuse_des_poids_hors_contrat():
    with pytest.raises(ValueError):
        somme_ponderee(matrice_exemple(), {"rentabilite_predite": 0.5})  # somme 0,5
    with pytest.raises(ValueError):
        somme_ponderee(matrice_exemple(), {"inconnu": 1.0})
    with pytest.raises(ValueError):
        somme_ponderee(matrice_exemple().drop(columns="cout_total"), POIDS_PAR_DEFAUT)


def test_somme_ponderee_tous_les_criteres_du_contrat():
    assert set(POIDS_PAR_DEFAUT) == set(CRITERES)
    assert len(somme_ponderee(matrice_exemple(), POIDS_PAR_DEFAUT)) == 4


# --- Tâche 9 : rank au format du contrat ----------------------------------------------

def test_rank_respecte_le_contrat():
    sortie = rank(matrice_exemple(), POIDS_PAR_DEFAUT)
    verifier_classement(sortie)
    assert sortie["rang"].tolist() == [1, 2, 3, 4]
    assert sorted(sortie["id_projet"]) == ["A", "B", "C", "D"]


def test_rank_ordre_attendu_et_entree_non_modifiee():
    m = matrice_exemple()
    copie = m.copy()
    sortie = rank(m, {"proba_risque": 1.0})
    assert sortie["id_projet"].tolist() == ["B", "A", "D", "C"]
    pd.testing.assert_frame_equal(m, copie)


def test_rank_un_seul_projet():
    sortie = rank(matrice_exemple().iloc[[2]], POIDS_PAR_DEFAUT)
    verifier_classement(sortie)
    assert sortie["rang"].tolist() == [1]
    assert sortie["score"].iloc[0] == pytest.approx(0.5)


def test_rank_egalites_rangs_uniques_et_stables():
    m = pd.concat([matrice_exemple()] * 3, ignore_index=True)
    m["id_projet"] = [f"P{i:02d}" for i in range(len(m))][::-1]
    sortie = rank(m, POIDS_PAR_DEFAUT)
    verifier_classement(sortie)
    # A, A', A'' ont le même score : départagés par id_projet croissant
    egaux = sortie[sortie["score"] == sortie["score"].iloc[0]]
    assert len(egaux) == 3
    assert egaux["id_projet"].tolist() == sorted(egaux["id_projet"])
    assert rank(m.sample(frac=1, random_state=42), POIDS_PAR_DEFAUT)["id_projet"].tolist() == sortie["id_projet"].tolist()


def test_rank_tous_projets_identiques():
    m = pd.concat([matrice_exemple().iloc[[0]]] * 5, ignore_index=True)
    m["id_projet"] = list("EDCBA")
    sortie = rank(m, POIDS_PAR_DEFAUT)
    verifier_classement(sortie)
    assert sortie["id_projet"].tolist() == list("ABCDE")


def test_rank_garde_les_colonnes_en_plus():
    m = matrice_exemple().assign(nom=["a", "b", "c", "d"], secteur="Énergie")
    sortie = rank(m, POIDS_PAR_DEFAUT)
    assert {"nom", "secteur", "score", "rang"} <= set(sortie.columns)
    assert sortie.set_index("id_projet").loc["B", "nom"] == "b"


def test_rank_tableau_vide():
    sortie = rank(matrice_exemple().iloc[0:0], POIDS_PAR_DEFAUT)
    verifier_classement(sortie)
    assert len(sortie) == 0


def test_rank_rapide_sur_2000_projets():
    rng = np.random.default_rng(42)
    m = pd.DataFrame({c: rng.random(2000) for c in CRITERES})
    m.insert(0, "id_projet", [f"P{i:04d}" for i in range(2000)])
    rank(m, POIDS_PAR_DEFAUT)  # premier appel : échauffement
    debut = time.perf_counter()
    for _ in range(10):
        sortie = rank(m, POIDS_PAR_DEFAUT)
    duree_ms = (time.perf_counter() - debut) / 10 * 1000
    verifier_classement(sortie)
    assert duree_ms < 50, f"rank trop lent : {duree_ms:.1f} ms"
