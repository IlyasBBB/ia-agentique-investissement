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
