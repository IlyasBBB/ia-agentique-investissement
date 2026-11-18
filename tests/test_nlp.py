"""Tests du module NLP (rôle 3) : format du contrat, cas limites, vitesse."""
import time

import pytest

from src import nlp
from src.contrat import Extraction, Passage

DOC_EXEMPLE = """Projet : Agriculture Resilience and Competitiveness Project
Identifiant : P171266
Pays : Bosnia and Herzegovina (Europe et Asie centrale)
Secteur : Agriculture

Objectif de développement (PDO) :
The Project Development Objective is to increase climate resilience and competitiveness of the agriculture sector.

Résumé :
This project has four components. The first component supports climate-resilient agriculture. \
The second component improves irrigation and drainage systems for climate change adaptation. \
The third component enhances food quality and safety. The fourth component supports a Project Implementation Unit.
"""

PETIT_CORPUS = {
    "P1": "Projet : Solar\nObjectif de développement (PDO) :\nExpand solar mini-grids for rural electrification.\n"
          "Résumé :\nNon disponible.",
    "P2": "Projet : Schools\nObjectif de développement (PDO) :\nImprove learning outcomes in primary schools.\n"
          "Résumé :\nTeachers are trained. Textbooks are distributed. Classrooms are built.",
    "P3": "Projet : Roads\nObjectif de développement (PDO) :\nRehabilitate rural roads to connect farmers to markets.\n"
          "Résumé :\nNon disponible.",
}


# --- Lecture et nettoyage ---------------------------------------------------


def test_nettoyer_repare_majuscules_et_mots_coupes():
    assert nlp.nettoyer("CONSERVATION OF WILDLIFE UTILIZA- TION.  IN  AFRICA.") == "Conservation of wildlife utilization. In africa."


def test_corps_garde_pdo_et_resume_sans_en_tete():
    texte = nlp.corps(DOC_EXEMPLE)
    assert texte.startswith("The Project Development Objective")
    assert "Implementation Unit" in texte
    assert "Bosnia" not in texte


def test_corps_ignore_resume_absent_et_accepte_texte_libre():
    assert nlp.corps(PETIT_CORPUS["P1"]) == "Expand solar mini-grids for rural electrification."
    assert nlp.corps("A free text without markers.") == "A free text without markers."
    assert nlp.corps("") == ""


def test_decouper_passages_couvre_tout_et_respecte_la_longueur():
    passages = nlp.decouper_passages({"P171266": DOC_EXEMPLE, **PETIT_CORPUS})
    assert {p["id_projet"] for p in passages} == {"P171266", "P1", "P2", "P3"}
    assert all(len(p["passage"].split()) <= nlp.PHRASES_PAR_PASSAGE * nlp.MOTS_MAX_PHRASE for p in passages)
    textes = " ".join(p["passage"] for p in passages if p["id_projet"] == "P171266")
    assert "Implementation Unit" in textes  # la fin du document n'est pas perdue


def test_vrais_documents_tous_decoupes():
    documents = nlp.charger_documents()
    assert len(documents) == 2000
    passages = nlp.decouper_passages(documents)
    assert {p["id_projet"] for p in passages} == set(documents)


# --- retrieve ---------------------------------------------------------------


def test_chercher_tfidf_format_ordre_et_un_passage_par_projet():
    index = nlp.IndexTfidf(nlp.decouper_passages(PETIT_CORPUS))
    resultats = nlp.chercher(index, "solar electricity for rural villages", k=3)
    assert [r["id_projet"] for r in resultats][0] == "P1"
    assert len(resultats) == 3
    assert len({r["id_projet"] for r in resultats}) == 3
    for r in resultats:
        Passage(**r)
    scores = [r["score"] for r in resultats]
    assert scores == sorted(scores, reverse=True)


def test_chercher_cas_limites():
    index = nlp.IndexTfidf(nlp.decouper_passages(PETIT_CORPUS))
    assert nlp.chercher(index, "", 3) == []
    assert nlp.chercher(index, "   ", 3) == []
    assert nlp.chercher(index, "schools", 0) == []
    assert len(nlp.chercher(index, "schools", 50)) == 3  # pas plus de projets qu'il n'en existe


def test_retrieve_question_vide_sans_charger_l_index(monkeypatch):
    def interdit(*args, **kwargs):
        raise AssertionError("l'index ne doit pas être chargé pour une question vide")

    monkeypatch.setattr(nlp, "index_par_defaut", interdit)
    assert nlp.retrieve("", 3) == []
    assert nlp.retrieve("   ", 3) == []
    assert nlp.retrieve("schools", 0) == []


def modele_embeddings_disponible() -> bool:
    """Vrai si le modèle d'embeddings est déjà téléchargé (pas de réseau pendant les tests)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(nlp.MODELE_EMBEDDINGS, "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not modele_embeddings_disponible(), reason="modèle d'embeddings non téléchargé")
def test_retrieve_vrai_corpus_trouve_le_projet_attendu():
    resultats = nlp.retrieve("rural electrification with solar mini-grids", k=5)
    assert len(resultats) == 5
    for r in resultats:
        Passage(**r)
    assert "P171742" in [r["id_projet"] for r in resultats[:3]]  # projet ADELE, Éthiopie
    debut = time.perf_counter()
    nlp.retrieve("access to finance for small businesses", k=50)
    assert time.perf_counter() - debut < 1.0  # index déjà en mémoire : réponse rapide


def projets_en_cours() -> dict[str, str]:
    """Documents des 500 projets en cours, lus comme le fait l'application."""
    import pandas as pd

    projets = pd.read_csv(nlp.RACINE / "data" / "processed" / "projets.csv")
    ids = projets.loc[projets.statut == "en_cours", "id_projet"]
    return {i: (nlp.DOSSIER_DOCS / f"{i}.txt").read_text(encoding="utf-8") for i in ids}


@pytest.mark.skipif(not modele_embeddings_disponible(), reason="modèle d'embeddings non téléchargé")
def test_retrieve_documents_connus_sans_nouvel_index(monkeypatch):
    documents = projets_en_cours()
    nlp.retrieve("rural electrification", 3)  # index par défaut déjà chargé, comme au démarrage

    def interdit(*args, **kwargs):
        raise AssertionError("aucun index ne doit être construit pour des documents déjà connus")

    monkeypatch.setattr(nlp, "index_a_la_volee", interdit)
    debut = time.perf_counter()
    resultats = nlp.retrieve("access to finance for small businesses", 3, documents=documents)
    assert time.perf_counter() - debut < 1.0
    assert len(resultats) == 3
    assert all(r["id_projet"] in documents for r in resultats)
    for r in resultats:
        Passage(**r)


@pytest.mark.skipif(not modele_embeddings_disponible(), reason="modèle d'embeddings non téléchargé")
def test_retrieve_documents_inconnus_index_a_la_volee_sans_fichier():
    avant = set(nlp.DOSSIER_MODELES.glob("*"))
    resultats = nlp.retrieve("solar electricity for rural villages", 3, documents=PETIT_CORPUS)
    assert [r["id_projet"] for r in resultats][0] == "P1"
    assert {r["id_projet"] for r in resultats} <= set(PETIT_CORPUS)
    assert set(nlp.DOSSIER_MODELES.glob("*")) == avant  # rien d'écrit sur disque


@pytest.mark.skipif(not modele_embeddings_disponible(), reason="modèle d'embeddings non téléchargé")
def test_retrieve_melange_et_texte_modifie():
    vrai = nlp.charger_documents()["P171742"]  # projet solaire en Éthiopie, dans l'index
    modifie = "Objectif de développement (PDO) :\nBuild new school classrooms and train primary teachers in rural areas."
    documents = {"P171742": vrai, "P2": PETIT_CORPUS["P2"], "P171266": modifie}
    solaire = nlp.retrieve("solar mini-grids for rural electrification", 1, documents=documents)
    assert solaire[0]["id_projet"] == "P171742"
    ecoles = nlp.retrieve("training primary school teachers", 3, documents=documents)
    assert {r["id_projet"] for r in ecoles} == set(documents)
    # P171266 a un texte modifié : c'est ce texte qui est cherché, pas celui de l'index
    assert any(r["id_projet"] == "P171266" and "classrooms" in r["passage"] for r in ecoles)


def test_retrieve_documents_vides():
    assert nlp.retrieve("schools", 3, documents={}) == []
    assert nlp.retrieve("schools", 3, documents={"X1": ""}) == []


# --- extract ----------------------------------------------------------------


@pytest.mark.parametrize("texte", ["", "   ", "Short text.", DOC_EXEMPLE, PETIT_CORPUS["P2"]])
def test_extract_respecte_le_contrat(texte):
    sortie = Extraction(**nlp.extract(texte))
    assert set(sortie.justifications) == set(nlp.CRITERES_TEXTE)
    assert all(isinstance(j, str) and j for j in sortie.justifications.values())


@pytest.mark.parametrize("texte", ["", "Short text.", "Objectif de développement (PDO) :\nImprove roads.\nRésumé :\nNon disponible."])
def test_extract_note_neutre_si_texte_trop_court(texte):
    sortie = nlp.extract(texte)
    assert all(sortie[c] == nlp.NOTE_NEUTRE for c in nlp.CRITERES_TEXTE)


def test_mots_cles_note_neutre_sans_aucun_mot_cle():
    texte = "Objectif de développement (PDO) :\n" + "The road will be built between the two towns by the ministry. " * 3
    assert all(nlp.solde_mots_cles(nlp.corps(texte), c) == 0 for c in nlp.CRITERES_TEXTE)
    sortie = nlp.extract_mots_cles(texte)
    assert all(sortie[c] == nlp.NOTE_NEUTRE for c in nlp.CRITERES_TEXTE)


def test_le_jeu_de_test_ne_sert_jamais_a_apprendre():
    import pandas as pd

    jeux = pd.read_csv(nlp.FICHIER_JEUX)
    test = set(jeux.loc[jeux.jeu == "test", "id_projet"])
    apprentissage = set(jeux.loc[jeux.jeu == "entrainement", "id_projet"])
    validation = set(pd.read_csv(nlp.FICHIER_VALIDATION).id_projet)
    assert len(test) == 100 and not test & apprentissage and not test & validation


def test_extract_utilise_le_modele_et_justifie_par_une_phrase_du_texte():
    assert nlp.modele_extract() is not None  # annotations présentes : pas de repli sur les mots-clés
    sortie = nlp.extract(DOC_EXEMPLE)
    texte = nlp.corps(DOC_EXEMPLE)
    for critere in nlp.CRITERES_TEXTE:
        assert sortie["justifications"][critere] in texte
    assert sortie["faisabilite"] >= 4  # quatre composantes et une unité de gestion du projet


def test_extract_rapide_sur_les_2000_documents():
    documents = nlp.charger_documents()
    nlp.modele_extract()  # chargement du modèle hors chronomètre, comme au démarrage de l'application
    debut = time.perf_counter()
    for document in documents.values():
        Extraction(**nlp.extract(document))
    assert time.perf_counter() - debut < 60  # environ 10 s sur un portable sans GPU
