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


def test_retrieve_question_vide():
    assert nlp.retrieve("", 3) == []


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
