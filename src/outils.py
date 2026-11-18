"""Point d'accès unique aux outils de l'agent.

USE_MOCKS=1 (par défaut) : fausses fonctions de src/mocks.py.
USE_MOCKS=0 : vrais modules des rôles 1, 2 et 3.
MODULES_REELS="nlp" (ou "nlp,supervise"...) : seulement ces modules en vrai, les autres en
fausses fonctions. Sert à essayer un module dès qu'il est livré, sans attendre les autres.
Elle passe avant USE_MOCKS. Un module demandé mais absent provoque une erreur claire plutôt
qu'un retour silencieux aux fausses fonctions.
"""
from __future__ import annotations

import importlib
import os

# Module de chaque rôle et les fonctions du contrat qu'il fournit.
MODULES = {
    "supervise": ("predict",),
    "mcda": ("rank", "detect_anomalies"),
    "nlp": ("extract", "retrieve"),
}


def modules_reels() -> set[str]:
    """Les modules branchés en vrai, d'après MODULES_REELS ou, à défaut, USE_MOCKS."""
    liste = os.getenv("MODULES_REELS", "").strip()
    if liste:
        noms = {n.strip() for n in liste.split(",") if n.strip()}
        inconnus = noms - set(MODULES)
        if inconnus:
            raise ValueError(f"MODULES_REELS : module inconnu {sorted(inconnus)}. "
                             f"Modules possibles : {', '.join(MODULES)}.")
        return noms
    return set(MODULES) if os.getenv("USE_MOCKS", "1") == "0" else set()


def utilise_mocks() -> bool:
    """Vrai si au moins une fausse fonction est branchée."""
    return modules_reels() != set(MODULES)


def description_modules() -> str:
    """Phrase courte qui dit quels modules sont vrais, pour l'application et les journaux."""
    reels = modules_reels()
    if not reels:
        return "fausses fonctions (USE_MOCKS=1)"
    if reels == set(MODULES):
        return "vrais modules des rôles 1 à 3"
    return f"vrais modules : {', '.join(sorted(reels))} ; fausses fonctions pour les autres"


def _charger() -> dict:
    """Associe chaque fonction du contrat à sa version vraie ou fausse."""
    fonctions = {}
    for module, noms in MODULES.items():
        source = f"src.{module}" if module in modules_reels() else "src.mocks"
        try:
            charge = importlib.import_module(source)
        except ImportError as erreur:
            raise ImportError(
                f"Le module {source} est demandé mais ne se charge pas : {erreur}. "
                "Retirer ce module de MODULES_REELS (ou remettre USE_MOCKS=1), "
                "ou attendre sa livraison."
            ) from erreur
        fonctions.update({nom: getattr(charge, nom) for nom in noms})
    return fonctions


globals().update(_charger())

__all__ = ["predict", "rank", "detect_anomalies", "extract", "retrieve",
           "utilise_mocks", "modules_reels", "description_modules"]
