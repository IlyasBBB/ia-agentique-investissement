"""Fausses fonctions au format du contrat.

Elles renvoient des valeurs inventées mais stables (un même projet donne toujours
le même résultat), pour développer l'agent et l'application sans attendre les
modules des rôles 1, 2 et 3. Aucune de ces valeurs n'a de sens métier.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from src.contrat import CRITERES, NIVEAUX_RISQUE, verifier_poids

PROJETS_EXEMPLE = pd.DataFrame(
    [
        {"id_projet": "P001", "nom": "Centrale solaire", "secteur": "Énergie", "pays": "Maroc", "cout_total": 120.0, "duree_mois": 36},
        {"id_projet": "P002", "nom": "Ligne de tramway", "secteur": "Transport", "pays": "Maroc", "cout_total": 310.0, "duree_mois": 60},
        {"id_projet": "P003", "nom": "Usine de dessalement", "secteur": "Eau", "pays": "Maroc", "cout_total": 85.0, "duree_mois": 30},
        {"id_projet": "P004", "nom": "Plateforme logistique", "secteur": "Logistique", "pays": "Sénégal", "cout_total": 45.0, "duree_mois": 18},
        {"id_projet": "P005", "nom": "Parc éolien", "secteur": "Énergie", "pays": "Tunisie", "cout_total": 2500.0, "duree_mois": 6},
    ]
)

DOCUMENTS_EXEMPLE = {
    "P001": "Projet de centrale solaire de 100 MW. Réduction attendue des émissions de CO2. Terrain déjà acquis.",
    "P002": "Extension du réseau de tramway sur 12 km. Forte attente des habitants. Expropriations à prévoir.",
    "P003": "Usine de dessalement pour alimenter 200 000 habitants. Technologie éprouvée. Consommation d'énergie élevée.",
    "P004": "Plateforme logistique près du port. Partenaire privé identifié. Études techniques terminées.",
    "P005": "Parc éolien en mer. Calendrier très court annoncé par le promoteur. Aucune étude d'impact fournie.",
}


def _alea(texte: str, sel: str) -> float:
    """Nombre entre 0 et 1, toujours le même pour un même texte."""
    empreinte = hashlib.md5(f"{sel}:{texte}".encode("utf-8")).hexdigest()
    return int(empreinte[:8], 16) / 0xFFFFFFFF


def predict(projet: dict) -> dict:
    """Fausse prédiction de rentabilité et de risque."""
    cle = str(projet.get("id_projet", projet.get("nom", "")))
    proba = round(_alea(cle, "risque"), 2)
    niveau = NIVEAUX_RISQUE[0 if proba < 0.33 else 1 if proba < 0.66 else 2]
    return {
        "rentabilite_predite": round(4 + 12 * _alea(cle, "rentabilite"), 1),
        "risque": niveau,
        "proba_risque": proba,
    }


def extract(document: str) -> dict:
    """Fausse extraction des critères qualitatifs."""
    notes = {c: 1 + int(_alea(document, c) * 4.999) for c in ("impact_env_social", "alignement_strategique", "faisabilite")}
    return {**notes, "justifications": {c: "Justification factice (mock)." for c in notes}}


def retrieve(question: str, k: int = 3) -> list[dict]:
    """Faux moteur de recherche : classe les documents d'exemple par mots en commun."""
    mots = set(question.lower().split())
    resultats = []
    for id_projet, texte in DOCUMENTS_EXEMPLE.items():
        communs = len(mots & set(texte.lower().split()))
        resultats.append({"id_projet": id_projet, "passage": texte, "score": round(communs / (len(mots) or 1), 2)})
    return sorted(resultats, key=lambda r: r["score"], reverse=True)[:k]


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Fausse détection : signale un coût par mois dix fois supérieur à la médiane."""
    sortie = df.copy()
    cout_par_mois = sortie["cout_total"] / sortie["duree_mois"]
    sortie["anomalie"] = cout_par_mois > 10 * cout_par_mois.median()
    return sortie


def rank(matrice: pd.DataFrame, poids: dict) -> pd.DataFrame:
    """Faux classement : somme pondérée des critères ramenés entre 0 et 1."""
    verifier_poids(poids)
    sortie = matrice.copy()
    score = pd.Series(0.0, index=sortie.index)
    for critere, p in poids.items():
        colonne = sortie[critere].astype(float)
        etendue = colonne.max() - colonne.min()
        normalise = (colonne - colonne.min()) / etendue if etendue else pd.Series(0.5, index=sortie.index)
        score += p * (normalise if CRITERES[critere] else 1 - normalise)
    sortie["score"] = score.clip(0, 1).round(4)
    sortie = sortie.sort_values("score", ascending=False).reset_index(drop=True)
    sortie["rang"] = range(1, len(sortie) + 1)
    return sortie
