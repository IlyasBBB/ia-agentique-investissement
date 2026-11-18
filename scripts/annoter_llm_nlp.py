"""Fait noter les documents par un LLM local (Ollama) selon la grille de notation.

Les notes du LLM servent :
- d'étiquettes d'entraînement et de validation pour le modèle TF-IDF (jeux entrainement et validation) ;
- de modèle de comparaison sur le jeu de test (jamais pour l'entraînement).

Le LLM est trop lent pour la démo (plusieurs secondes par document) : il n'est appelé qu'ici, une fois.
Le script reprend où il s'est arrêté si on le relance.

Usage : python scripts/annoter_llm_nlp.py [--modele qwen2.5:3b] [--limite N]
        (écrit reports/role3/annotations/notes_llm.csv)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import nlp  # noqa: E402

DOSSIER = nlp.RACINE / "reports" / "role3" / "annotations"
SORTIE = DOSSIER / "notes_llm.csv"
JEUX_A_NOTER = ("test", "entrainement", "validation")

# Traduction en anglais de reports/role3/grille_notation.md (les documents sont en anglais).
CONSIGNE = """You rate World Bank project descriptions on three criteria, each an integer from 1 to 5.
Use ONLY the text. 3 means neutral: not enough evidence either way (always 3 for a very short or empty text).

impact (expected environmental and social benefits):
5 = main objective is explicitly social or environmental with precise targets or beneficiaries (poverty, vulnerable groups, women, health, access to water or energy, climate, biodiversity)
4 = explicit social or environmental benefits, but secondary or without precise target
3 = nothing clear, or text too short
2 = purely financial, administrative or infrastructure project with no social or environmental benefit mentioned
1 = negative effects mentioned (displacement, pollution, deforestation) without mitigation

alignment (consistency with development priorities):
5 = explicitly part of a national or sector strategy, a government reform program or an international framework, AND targets a priority issue (human capital, climate, energy transition, digital, jobs)
4 = clearly targets a priority issue, OR explicitly linked to a government program, but not both
3 = general development objective with no clear link or priority, or text too short
2 = vague, very narrow or one-off objective with no link to a broader policy
1 = objective contradicts stated priorities or is incoherent

feasibility (ability to be implemented as planned):
5 = detailed components, precise institutional set-up (project management unit, implementing agency), monitoring and evaluation, capacity building, phasing or previous experience (second phase, successful pilot)
4 = components described and at least one implementation element (management, monitoring, capacity)
3 = objective only, no description of implementation, or text too short
2 = difficult context mentioned (conflict, crisis, emergency, fragility, weak institutions) with few implementation elements
1 = very difficult context and very ambitious objectives with no implementation arrangements

Answer with JSON only: {"impact": n, "alignment": n, "feasibility": n}"""

# Clés courtes en anglais dans la réponse du LLM (il écorche les noms français) -> noms du contrat.
CLES = {"impact": "impact_env_social", "alignment": "alignement_strategique", "feasibility": "faisabilite"}


def noter(client, modele: str, texte: str, graine: int = 42) -> dict[str, int]:
    """Demande les trois notes au LLM ; lève ValueError si la réponse est inexploitable."""
    reponse = client.chat(
        model=modele,
        messages=[{"role": "system", "content": CONSIGNE}, {"role": "user", "content": f"Text:\n{texte or '(empty)'}"}],
        format="json",
        options={"temperature": 0, "seed": graine, "num_ctx": 4096},
    )
    notes = json.loads(reponse["message"]["content"])
    sortie = {CLES[c]: int(notes[c]) for c in CLES}
    if not all(1 <= n <= 5 for n in sortie.values()):
        raise ValueError(f"note hors de 1 à 5 : {sortie}")
    return sortie


def main() -> None:
    """Note les documents des jeux test, entrainement et validation qui ne le sont pas encore."""
    import ollama

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--modele", default="qwen2.5:3b")
    parseur.add_argument("--limite", type=int, default=None, help="nombre maximal de documents à noter")
    args = parseur.parse_args()

    jeux = pd.read_csv(DOSSIER / "jeux_extract.csv")
    a_noter = jeux[jeux.jeu.isin(JEUX_A_NOTER)]
    # Le jeu de test d'abord : c'est lui qui sert à comparer le LLM aux autres modèles.
    a_noter = a_noter.assign(ordre=a_noter.jeu.map({"test": 0, "entrainement": 1, "validation": 2})).sort_values(["ordre", "id_projet"])
    deja = set(pd.read_csv(SORTIE).id_projet) if SORTIE.exists() else set()
    restants = [i for i in a_noter.id_projet if i not in deja][: args.limite]
    logging.info("%d documents déjà notés, %d à noter avec %s", len(deja), len(restants), args.modele)

    client = ollama.Client()
    debut = time.perf_counter()
    for n, id_projet in enumerate(restants, start=1):
        texte = nlp.corps((nlp.DOSSIER_DOCS / f"{id_projet}.txt").read_text(encoding="utf-8"))
        for essai in range(3):
            try:
                notes = noter(client, args.modele, texte, graine=42 + essai)  # autre graine si la 1re réponse est inexploitable
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as erreur:
                logging.warning("%s, essai %d : réponse inexploitable (%s)", id_projet, essai + 1, erreur)
        else:
            continue  # trois échecs : document laissé de côté, il sera retenté au prochain lancement
        ligne = pd.DataFrame([{"id_projet": id_projet, **notes, "modele": args.modele}])
        ligne.to_csv(SORTIE, mode="a", header=not SORTIE.exists(), index=False)
        if n % 25 == 0 or n == len(restants):
            ecoule = time.perf_counter() - debut
            logging.info("%d / %d notés (%.1f s par document)", n, len(restants), ecoule / n)


if __name__ == "__main__":
    main()
