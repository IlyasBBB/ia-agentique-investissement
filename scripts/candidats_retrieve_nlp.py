"""Rassemble les projets candidats de chaque requête de test, à juger ensuite à la main.

Méthode de mise en commun (« pooling », comme dans les campagnes TREC) : pour chaque requête,
on réunit les 10 premiers projets de TF-IDF, les 10 premiers des embeddings et jusqu'à 10 projets
dont le titre contient un synonyme choisi à la main. Chaque candidat est ensuite jugé pertinent
ou non. Limite : un projet pertinent qu'aucune des trois sources ne trouve n'est pas jugé.

Usage : python scripts/candidats_retrieve_nlp.py   (écrit reports/role3/annotations/candidats_retrieve.csv)
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import nlp  # noqa: E402

DOSSIER = nlp.RACINE / "reports" / "role3" / "annotations"
PROFONDEUR = 10

# Synonymes cherchés dans le titre des projets, écrits avant de regarder les résultats des moteurs.
MOTIFS_TITRE = {
    "R01": r"electrif|off-grid|mini-?grid|solar|energy access",
    "R02": r"primary|basic education|learning|quality education",
    "R03": r"nutrition|stunting|malnutrition",
    "R04": r"flood|urban resilien|climate resilien",
    "R05": r"sme|msme|access to finance|small and medium|enterprise",
    "R06": r"rural road|rural access|feeder road|rural transport",
    "R07": r"cash transfer|safety net|social assistance|social protection",
    "R08": r"tax|revenue|customs",
    "R09": r"broadband|digital|ict|telecom|connectivity",
    "R10": r"irrigation|water resources|agricultural productivity",
    "R11": r"water supply|sanitation|sewer|wastewater|urban water",
    "R12": r"covid|pandemic|disease|epidemic|ebola|hiv|aids|malaria|tuberculosis",
    "R13": r"skills|vocational|tvet|youth employment|technical education",
    "R14": r"port|maritime|trade logistics|shipping",
    "R15": r"power sector|electricity sector|utility|utilities|power company",
    "R16": r"drought|climate.smart|resilien|dryland",
    "R17": r"girl|secondary education|secondary school",
    "R18": r"maternal|mother|newborn|child health|reproductive",
    "R19": r"renewable|clean energy|emission|energy efficiency|wind|solar|geothermal",
    "R20": r"bank|financial sector|financial crisis|crisis response",
    "R21": r"bus|transit|metro|urban transport|brt",
    "R22": r"public works|labor.intensive|labour.intensive|employment",
    "R23": r"public financial|public expenditure|budget|fiscal",
    "R24": r"e-government|egov|digital government|public service delivery|govtech",
    "R25": r"forest|biodiversity|protected area|wildlife|conservation",
    "R26": r"solid waste|waste management|landfill",
    "R27": r"hospital|health system|health service|quality of care",
    "R28": r"higher education|universit|tertiary|research",
    "R29": r"investment climate|business environment|competitiveness|private sector",
    "R30": r"highway|road|expressway|corridor",
    "R31": r"emergency|earthquake|cyclone|hurricane|recovery|reconstruction",
    "R32": r"land|cadastr|property registr|titling",
    "R33": r"livestock|animal|veterinar|dairy",
    "R34": r"fisher|coastal|marine|aquaculture",
    "R35": r"pension|social insurance|social security",
    "R36": r"hydro|dam",
    "R37": r"microfinance|micro.finance|women entrepreneur|women's|gender",
    "R38": r"rail",
    "R39": r"refugee|displaced|host communit",
    "R40": r"trade|export|border|customs",
}


def main() -> None:
    """Écrit la liste des candidats à juger, avec le titre et le début du texte de chaque projet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    requetes = pd.read_csv(DOSSIER / "requetes_retrieve.csv")
    documents = nlp.charger_documents()
    titres = {i: nlp.titre(d) for i, d in documents.items()}
    index = {m: nlp.construire_index(m, documents) for m in ("tfidf", "embeddings")}

    lignes = []
    for _, r in requetes.iterrows():
        sources: dict[str, list[str]] = {}
        for methode, idx in index.items():
            for res in nlp.chercher(idx, r["requete"], PROFONDEUR):
                sources.setdefault(res["id_projet"], []).append(methode)
        motif = re.compile(MOTIFS_TITRE[r["id_requete"]], flags=re.IGNORECASE)
        for id_projet in [i for i, t in titres.items() if motif.search(t)][:PROFONDEUR]:
            sources.setdefault(id_projet, []).append("titre")
        for id_projet, src in sources.items():
            lignes.append({
                "id_requete": r["id_requete"], "id_projet": id_projet, "sources": "+".join(src),
                "titre": titres[id_projet], "extrait": nlp.corps(documents[id_projet])[:300],
            })
    sortie = pd.DataFrame(lignes)
    sortie.to_csv(DOSSIER / "candidats_retrieve.csv", index=False, encoding="utf-8")
    logging.info("%d candidats pour %d requêtes (%.1f par requête).",
                 len(sortie), len(requetes), len(sortie) / len(requetes))


if __name__ == "__main__":
    main()
