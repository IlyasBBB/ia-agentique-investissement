"""Évalue l'agent sur les cinq scénarios de test et le compare au pipeline fixe.

Pour chaque passage et chaque scénario, on lance sur les mêmes 30 projets :
- pipeline_fixe : le modèle de référence, sans LLM ;
- run_agent : l'agent piloté par le LLM (Ollama doit tourner).

Fichiers écrits dans reports/ :
- resultats_scenarios.csv : une ligne par exécution, avec toutes les vérifications ;
- sorties_scenarios.json : les textes complets et le journal des étapes, pour l'analyse des erreurs.

Exemples :
    python scripts/evaluer_agent.py --passages 2
    python scripts/evaluer_agent.py --scenarios 5 --suffixe apres_correction
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src import outils  # noqa: E402
from src.agent import MODELE_PAR_DEFAUT, pipeline_fixe, run_agent  # noqa: E402
from src.scenarios import SCENARIOS, charger_projets_scenarios, verifier_sortie  # noqa: E402

journal = logging.getLogger("evaluer_agent")
DOSSIER_RAPPORTS = RACINE / "reports"
COLONNES_RESUME = ["passage", "scenario", "mode", "duree_s", "ordre_ok", "poids_ok", "top3_cite_par_llm",
                   "top3_dans_sortie", "anomalies_citees_par_llm", "citation_ok", "chiffres_non_retrouves",
                   "controle_auto", "secours", "meme_top3_que_pipeline"]


def executer(mode: str, scenario: dict, projets: pd.DataFrame, documents: dict) -> tuple[dict, float]:
    """Lance l'agent ou le pipeline fixe sur un scénario et mesure la durée en secondes."""
    fonction = run_agent if mode == "agent" else pipeline_fixe
    debut = time.perf_counter()
    sortie = fonction(scenario["demande"], projets, documents, scenario["poids_utilisateur"])
    return sortie, round(time.perf_counter() - debut, 1)


def evaluer(passages: int, numeros: list[int], suffixe: str) -> pd.DataFrame:
    """Lance les scénarios demandés et écrit les deux fichiers de résultats au fur et à mesure."""
    projets, documents = charger_projets_scenarios()
    journal.info("%d projets, %d documents, USE_MOCKS=%s, modèle %s", len(projets), len(documents),
                 "1" if outils.utilise_mocks() else "0", os.getenv("OLLAMA_MODEL") or MODELE_PAR_DEFAUT)
    DOSSIER_RAPPORTS.mkdir(exist_ok=True)
    fin = f"_{suffixe}" if suffixe else ""
    fichier_csv = DOSSIER_RAPPORTS / f"resultats_scenarios{fin}.csv"
    fichier_json = DOSSIER_RAPPORTS / f"sorties_scenarios{fin}.json"

    lignes, textes = [], []
    for passage in range(1, passages + 1):
        for scenario in [s for s in SCENARIOS if s["numero"] in numeros]:
            top3_pipeline = None
            for mode in ("pipeline_fixe", "agent"):  # le pipeline d'abord : il sert de point de comparaison
                journal.info("Passage %d, scénario %d, %s...", passage, scenario["numero"], mode)
                sortie, duree = executer(mode, scenario, projets, documents)
                verifications = verifier_sortie(sortie, scenario)
                if mode == "pipeline_fixe":
                    top3_pipeline = verifications["top3"]
                lignes.append({"passage": passage, "scenario": scenario["numero"], "nom": scenario["nom"],
                               "mode": mode, "duree_s": duree, **verifications,
                               "meme_top3_que_pipeline": verifications["top3"] == top3_pipeline})
                textes.append({"passage": passage, "scenario": scenario["numero"], "mode": mode,
                               "demande": scenario["demande"], "duree_s": duree,
                               "recommandation": sortie["recommandation"], "etapes": sortie["etapes"]})
                # Écriture après chaque exécution : rien n'est perdu si on arrête le script.
                pd.DataFrame(lignes).to_csv(fichier_csv, index=False, encoding="utf-8")
                fichier_json.write_text(json.dumps(textes, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
                journal.info("  terminé en %.1f s", duree)
    journal.info("Résultats écrits dans %s et %s", fichier_csv, fichier_json)
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    analyseur = argparse.ArgumentParser(description="Évalue l'agent sur les scénarios de test.")
    analyseur.add_argument("--passages", type=int, default=1, help="nombre de passages complets (1 par défaut)")
    analyseur.add_argument("--scenarios", default="1,2,3,4,5", help="numéros des scénarios, séparés par des virgules")
    analyseur.add_argument("--suffixe", default="", help="ajouté au nom des fichiers, pour garder plusieurs essais")
    options = analyseur.parse_args()
    resultats = evaluer(options.passages, [int(n) for n in options.scenarios.split(",")], options.suffixe)
    # Affichage du tableau final : c'est le résultat du script, pas un message de suivi.
    print(resultats[COLONNES_RESUME].to_string(index=False))
