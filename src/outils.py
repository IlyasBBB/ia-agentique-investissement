"""Point d'accès unique aux outils de l'agent.

USE_MOCKS=1 (par défaut) : fausses fonctions de src/mocks.py.
USE_MOCKS=0 : vrais modules des rôles 1, 2 et 3. Un module absent provoque une
erreur claire plutôt qu'un retour silencieux aux fausses fonctions.
"""
from __future__ import annotations

import os


def utilise_mocks() -> bool:
    """Vrai si les fausses fonctions sont branchées."""
    return os.getenv("USE_MOCKS", "1") != "0"


if utilise_mocks():
    from src.mocks import detect_anomalies, extract, predict, rank, retrieve
else:
    try:
        from src.mcda import detect_anomalies, rank
        from src.nlp import extract, retrieve
        from src.supervise import predict
    except ImportError as erreur:
        raise ImportError(
            f"USE_MOCKS=0 mais un module manque : {erreur}. "
            "Remettre USE_MOCKS=1 ou attendre la livraison du module."
        ) from erreur

__all__ = ["predict", "rank", "detect_anomalies", "extract", "retrieve", "utilise_mocks"]
