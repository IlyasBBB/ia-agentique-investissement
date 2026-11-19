"""Scénarios de test de l'agent et vérifications automatiques de ses réponses.

Ce module ne lance rien lui-même. Il fournit :
- SCENARIOS : les cinq demandes de test ;
- charger_projets_scenarios : les 30 projets, toujours les mêmes, sur lesquels on teste ;
- verifier_sortie : les contrôles faits sur une réponse de run_agent ou de pipeline_fixe.

Il est utilisé par tests/test_agent.py (tests marqués llm) et par scripts/evaluer_agent.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.agent import NB_RECOMMANDES, normaliser_poids
from src.contrat import CRITERES, POIDS_PAR_DEFAUT, Etape, verifier_classement, verifier_poids

RACINE = Path(__file__).resolve().parent.parent
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
DOSSIER_DOCUMENTS = RACINE / "data" / "docs"
NB_PROJETS = 30
OUTILS_AVANT_CLASSER = ("extraire_criteres", "predire", "detecter_anomalies")
TITRE_CONTROLE = "Contrôle automatique"

# criteres_attendus : les critères qui doivent recevoir les plus gros poids.
# Liste vide : la demande ne donne aucune priorité, on attend les poids par défaut.
# poids_utilisateur : poids fixés à la main, que l'agent doit reprendre tels quels.
SCENARIOS = [
    {"numero": 1, "nom": "Peu risqué, fort impact environnemental",
     "demande": "Je veux des projets peu risqués avec un fort impact environnemental",
     "criteres_attendus": ["proba_risque", "impact_env_social"], "poids_utilisateur": None},
    {"numero": 2, "nom": "Le plus rentable, même risqué",
     "demande": "Je cherche les projets les plus rentables, même s'ils sont risqués",
     "criteres_attendus": ["rentabilite_predite"], "poids_utilisateur": None},
    {"numero": 3, "nom": "Rapide, peu coûteux, facile",
     "demande": "Je veux des projets rapides à réaliser, peu coûteux et faciles à mettre en œuvre",
     "criteres_attendus": ["cout_total", "duree_mois", "faisabilite"], "poids_utilisateur": None},
    {"numero": 4, "nom": "Poids fixés par l'utilisateur",
     "demande": "Classe les projets avec les poids que j'ai fixés",
     "criteres_attendus": [],
     "poids_utilisateur": {"faisabilite": 0.5, "cout_total": 0.3, "rentabilite_predite": 0.2}},
    {"numero": 5, "nom": "Demande hors sujet",
     "demande": "Quel temps fera-t-il demain ?",
     "criteres_attendus": [], "poids_utilisateur": None},
]


def charger_projets_scenarios(nombre: int = NB_PROJETS) -> tuple[pd.DataFrame, dict[str, str]]:
    """Les `nombre` premiers projets en cours, triés par identifiant, avec leur document.

    La règle est volontairement simple : elle redonne toujours les mêmes projets.
    """
    projets = pd.read_csv(FICHIER_PROJETS)
    projets = projets[projets["statut"] == "en_cours"].sort_values("id_projet").head(nombre)
    projets = projets.reset_index(drop=True)
    documents = {}
    for id_projet in projets["id_projet"]:
        fichier = DOSSIER_DOCUMENTS / f"{id_projet}.txt"
        if fichier.exists():
            documents[id_projet] = fichier.read_text(encoding="utf-8")
    return projets, documents


# --- Petites fonctions de lecture d'une sortie ------------------------------------


def outils_appeles(sortie: dict) -> list[str]:
    """Noms des outils du journal, dans l'ordre des appels."""
    return [etape["outil"] for etape in sortie["etapes"]]


def poids_du_classement(sortie: dict, poids_utilisateur: dict | None = None) -> dict[str, float]:
    """Poids réellement utilisés : ceux de l'utilisateur, sinon ceux du dernier appel à classer."""
    if poids_utilisateur:
        return normaliser_poids(poids_utilisateur)
    appels = [e for e in sortie["etapes"] if e["outil"] == "classer"]
    return normaliser_poids(appels[-1]["arguments"].get("poids") if appels else None)


def trois_premiers(classement: pd.DataFrame) -> list[str]:
    """Identifiants des trois premiers projets, dossiers anormaux écartés (règle de l'agent)."""
    if "anomalie" in classement:
        classement = classement[~classement["anomalie"].astype(bool)]
    return classement["id_projet"].head(NB_RECOMMANDES).tolist()


def dossiers_anormaux(classement: pd.DataFrame) -> list[str]:
    """Identifiants des dossiers signalés anormaux dans le classement."""
    if "anomalie" not in classement:
        return []
    return classement.loc[classement["anomalie"].astype(bool), "id_projet"].tolist()


def texte_du_llm(recommandation: str) -> str:
    """La recommandation sans le bloc « Contrôle automatique » ajouté par le code."""
    return recommandation.split(TITRE_CONTROLE)[0]


# --- Les vérifications --------------------------------------------------------------


def contrat_respecte(sortie: dict) -> bool:
    """Vrai si la sortie a les trois clés du contrat, un classement et un journal valides."""
    try:
        assert set(sortie) == {"classement", "recommandation", "etapes"}
        assert isinstance(sortie["recommandation"], str) and sortie["recommandation"].strip()
        verifier_classement(sortie["classement"])
        for etape in sortie["etapes"]:
            Etape(**etape)
    except Exception:
        return False
    return True


def ordre_respecte(outils: list[str]) -> bool:
    """Vrai si les trois outils de préparation ont été demandés avant le premier classer."""
    if "classer" not in outils:
        return False
    avant = outils[:outils.index("classer")]
    return all(outil in avant for outil in OUTILS_AVANT_CLASSER)


def poids_coherents(poids: dict[str, float], scenario: dict) -> bool:
    """Vrai si les poids vont dans le sens de la demande du scénario."""
    if scenario["poids_utilisateur"]:
        attendus = normaliser_poids(scenario["poids_utilisateur"])
        return all(abs(poids[c] - attendus[c]) < 1e-6 for c in CRITERES)
    if not scenario["criteres_attendus"]:  # aucune priorité dans la demande : poids par défaut
        return all(abs(poids[c] - POIDS_PAR_DEFAUT[c]) < 1e-6 for c in CRITERES)
    # Chaque critère attendu doit peser au moins autant que tous les autres critères.
    plus_petit_attendu = min(poids[c] for c in scenario["criteres_attendus"])
    plus_grand_autre = max(p for c, p in poids.items() if c not in scenario["criteres_attendus"])
    return plus_petit_attendu > 0 and plus_petit_attendu >= plus_grand_autre


def chiffres_non_retrouves(texte: str, classement: pd.DataFrame, poids: dict[str, float]) -> list[str]:
    """Nombres à virgule du texte qui ne correspondent à aucune valeur rendue par les outils.

    Valeurs admises : score, rentabilité, probabilité de risque et coût des projets cités
    dans le texte, et les poids. Tolérance : une unité du dernier chiffre écrit (arrondi
    ou troncature). Les nombres entre guillemets (passage cité) ne sont pas comptés.
    Limite : les nombres entiers (notes sur 5, rangs, pourcentages
    arrondis) ne sont pas contrôlés, et un chiffre juste attribué au mauvais projet
    cité passe le contrôle.
    """
    cites = classement[classement["id_projet"].apply(lambda i: i in texte)]
    admis = list(poids.values()) + [100 * p for p in poids.values()]
    for colonne in ("score", "rentabilite_predite", "proba_risque", "cout_total"):
        if colonne in cites:
            admis += cites[colonne].astype(float).tolist()
    if "proba_risque" in cites:  # une probabilité peut être écrite en pourcentage
        admis += (100 * cites["proba_risque"].astype(float)).tolist()

    # Les nombres d'un passage cité viennent du document, pas des outils : ils sont vérifiés par
    # citation_verifiee (la citation doit figurer mot pour mot dans un passage rendu).
    hors_citations = re.sub(r"«[^»]*»|\"[^\"]*\"|“[^”]*”", " ", texte)
    inconnus = []
    for ecrit in re.findall(r"\d+[.,]\d+", hors_citations):
        nombre = float(ecrit.replace(",", "."))
        tolerance = 10 ** -len(re.split(r"[.,]", ecrit)[1]) + 1e-9
        if not any(abs(nombre - valeur) <= tolerance for valeur in admis):
            inconnus.append(ecrit)
    return inconnus


def identifiants_inconnus(texte: str, classement: pd.DataFrame) -> list[str]:
    """Identifiants de projet (P suivi de chiffres) écrits dans le texte mais absents du portefeuille."""
    connus = set(classement["id_projet"])
    return sorted({i for i in re.findall(r"\bP\d{3,}\b", texte) if i not in connus})


def citation_verifiee(texte: str, sortie: dict) -> bool:
    """Vrai si le texte ne cite aucun passage, ou seulement des passages rendus par chercher_passages."""
    apres = re.split(r"passage cité", texte, flags=re.I)
    if len(apres) < 2:
        return True
    citation = apres[1].split("Limites")[0]
    morceaux = re.findall(r"«\s*(.+?)\s*»|\"(.+?)\"|“(.+?)”", citation, flags=re.S)
    morceaux = [next(m for m in groupe if m) for groupe in morceaux]
    if not morceaux:  # pas de guillemets : honnête seulement si le texte dit « aucun »
        return "aucun" in citation.lower()
    rendus = " ".join(e["resultat"] for e in sortie["etapes"] if e["outil"] == "chercher_passages")
    rendus = " ".join(rendus.split())
    return all(" ".join(m.split()) in rendus for m in morceaux)


def verifier_sortie(sortie: dict, scenario: dict) -> dict:
    """Toutes les vérifications d'un scénario, rendues dans un dictionnaire (une ligne du tableau)."""
    recommandation = sortie["recommandation"]
    llm = texte_du_llm(recommandation)
    classement = sortie["classement"]
    outils = outils_appeles(sortie)
    poids = poids_du_classement(sortie, scenario["poids_utilisateur"])
    verifier_poids(poids)
    top3 = trois_premiers(classement)
    anomalies = dossiers_anormaux(classement)
    return {
        "contrat_ok": contrat_respecte(sortie),
        "outils": " > ".join(outils),
        "ordre_ok": ordre_respecte(outils),
        "passages_cherches": "chercher_passages" in outils,
        "poids": ", ".join(f"{c} {p:.2f}" for c, p in poids.items() if p > 0),
        "poids_ok": poids_coherents(poids, scenario),
        "top3": ", ".join(top3),
        "top3_cite_par_llm": all(i in llm for i in top3),
        "top3_dans_sortie": all(i in recommandation for i in top3),
        "anomalies": ", ".join(anomalies),
        "anomalies_citees_par_llm": all(i in llm for i in anomalies),
        "anomalies_dans_sortie": all(i in recommandation for i in anomalies),
        "chiffres_non_retrouves": ", ".join(chiffres_non_retrouves(llm, classement, poids)),
        "identifiants_inconnus": ", ".join(identifiants_inconnus(llm, classement)),
        "citation_ok": citation_verifiee(llm, sortie),
        "controle_auto": TITRE_CONTROLE in recommandation,
        "secours": "sans LLM" in recommandation,
    }
