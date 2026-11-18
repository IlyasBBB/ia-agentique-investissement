"""Rôle 3 : NLP et RAG.

Deux fonctions du contrat (docs/contrat.md) :

- extract(document) : trois notes qualitatives de 1 à 5 et une justification par critère ;
- retrieve(question, k) : les k passages des documents de projets les plus proches de la question.

Les documents sont les fichiers data/docs/<id_projet>.txt. Seuls l'objectif de développement
(PDO) et le résumé sont analysés : l'en-tête (pays, coût, durée...) est déjà dans les chiffres.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

journal = logging.getLogger(__name__)

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DOCS = RACINE / "data" / "docs"
DOSSIER_MODELES = RACINE / "models" / "nlp"  # ignoré par Git

CRITERES_TEXTE = ("impact_env_social", "alignement_strategique", "faisabilite")
NOTE_NEUTRE = 3
MODELE_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
PHRASES_PAR_PASSAGE = 3
PAS_PASSAGE = 2  # fenêtre de 3 phrases avançant de 2 : une phrase de recouvrement
MOTS_MAX_PHRASE = 60  # au-delà, une phrase est coupée : un passage fait au plus 180 mots

MARQUEUR_PDO = "Objectif de développement (PDO) :"
MARQUEUR_RESUME = "Résumé :"
ABSENT = "Non disponible."


# --- Lecture et nettoyage des documents -------------------------------------


def nettoyer(texte: str) -> str:
    """Répare les défauts fréquents des textes de la Banque mondiale.

    Mots coupés par un tiret de fin de ligne (« UTILIZA- TION »), espaces multiples,
    phrases entièrement en majuscules (anciens PDO) remises en minuscules.
    """
    texte = re.sub(r"(\w)- (\w)", r"\1\2", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    lettres = [c for c in texte if c.isalpha()]
    if lettres and sum(c.isupper() for c in lettres) / len(lettres) > 0.8:
        texte = ". ".join(p.strip().capitalize() for p in texte.lower().split(". "))
    return texte


def corps(document: str) -> str:
    """Rend le texte utile d'un document : PDO puis résumé, nettoyés.

    Si le document ne suit pas le format de data/docs/, tout le texte est gardé.
    """
    if not document:
        return ""
    if MARQUEUR_PDO not in document:
        return nettoyer(document)
    apres_pdo = document.split(MARQUEUR_PDO, 1)[1]
    pdo, _, resume = apres_pdo.partition(MARQUEUR_RESUME)
    morceaux = [nettoyer(m) for m in (pdo, resume) if m.strip() and m.strip() != ABSENT]
    return " ".join(m for m in morceaux if m)


def titre(document: str) -> str:
    """Nom du projet lu dans l'en-tête (chaîne vide s'il manque)."""
    trouve = re.search(r"^Projet : (.+)$", document or "", flags=re.MULTILINE)
    return trouve.group(1).strip() if trouve else ""


def phrases(texte: str) -> list[str]:
    """Découpe un texte en phrases (règle simple : ponctuation forte suivie d'un espace).

    Une « phrase » de plus de MOTS_MAX_PHRASE mots (énumération sans point) est coupée en morceaux.
    """
    resultat = []
    for phrase in re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9(])", texte):
        mots = phrase.split()
        for debut in range(0, len(mots), MOTS_MAX_PHRASE):
            morceau = " ".join(mots[debut:debut + MOTS_MAX_PHRASE])
            if len(morceau) > 2:
                resultat.append(morceau)
    return resultat


def charger_documents(dossier: Path = DOSSIER_DOCS) -> dict[str, str]:
    """Lit tous les documents : {id_projet: texte brut}."""
    return {f.stem: f.read_text(encoding="utf-8") for f in sorted(dossier.glob("*.txt"))}


def decouper_passages(documents: dict[str, str]) -> list[dict]:
    """Découpe chaque document en passages de quelques phrases qui se recouvrent.

    Chaque passage garde le nom du projet à part : il est ajouté au texte indexé
    (il aide à situer un passage isolé) mais pas au passage rendu à l'agent.
    """
    passages = []
    for id_projet, document in documents.items():
        liste = phrases(corps(document))
        nom = titre(document)
        if not liste:
            continue
        dernier = max(len(liste) - PHRASES_PAR_PASSAGE, 0)
        debuts = list(range(0, dernier + 1, PAS_PASSAGE))
        if debuts[-1] != dernier:
            debuts.append(dernier)  # dernière fenêtre, pour ne pas perdre la fin du texte
        for debut in debuts:
            texte = " ".join(liste[debut:debut + PHRASES_PAR_PASSAGE])
            passages.append({"id_projet": id_projet, "passage": texte, "indexe": f"{nom}. {texte}"})
    return passages
