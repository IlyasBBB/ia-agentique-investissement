"""Évalue retrieve sur les 40 requêtes jugées à la main : TF-IDF, embeddings, hybride.

Les 15 requêtes « reglage » servent à choisir le poids alpha du moteur hybride ;
les 25 requêtes « test » ne servent qu'à la mesure finale.

Mesures (au niveau des projets, un passage par projet) :
- précision@k : part de projets pertinents parmi les k premiers ;
- MRR : moyenne de 1 / rang du premier projet pertinent ;
- rappel@10 : part des projets pertinents connus retrouvés dans les 10 premiers.

Usage : python scripts/evaluer_retrieve_nlp.py   (écrit reports/role3/resultats_retrieve.csv)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import nlp  # noqa: E402

DOSSIER = nlp.RACINE / "reports" / "role3"
ALPHAS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRAINE = 42

journal = logging.getLogger("evaluer_retrieve")


def mesures(classes: list[str], pertinents: set[str]) -> dict[str, float]:
    """Précision@3, précision@5, MRR et rappel@10 pour une requête."""
    vrais = [i in pertinents for i in classes]
    premier = next((rang for rang, v in enumerate(vrais, start=1) if v), None)
    return {
        "precision@3": float(np.mean(vrais[:3])),
        "precision@5": float(np.mean(vrais[:5])),
        "mrr": 1 / premier if premier else 0.0,
        "rappel@10": sum(vrais[:10]) / len(pertinents),
    }


def evaluer(index, requetes: pd.DataFrame, jugements: dict[str, set[str]]) -> pd.DataFrame:
    """Une ligne par requête avec ses mesures."""
    lignes = []
    for _, r in requetes.iterrows():
        classes = [p["id_projet"] for p in nlp.chercher(index, r["requete"], 10)]
        lignes.append({"id_requete": r["id_requete"], "jeu": r["jeu"], **mesures(classes, jugements[r["id_requete"]])})
    return pd.DataFrame(lignes)


def intervalle_difference(a: pd.Series, b: pd.Series, n: int = 5000) -> tuple[float, float]:
    """Intervalle de confiance à 95 % (bootstrap sur les requêtes) de la différence moyenne a - b."""
    rng = np.random.default_rng(GRAINE)
    diff = (a - b).to_numpy()
    moyennes = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)]
    return float(np.percentile(moyennes, 2.5)), float(np.percentile(moyennes, 97.5))


def main() -> None:
    """Choisit alpha sur le jeu de réglage, puis mesure les trois moteurs sur le jeu de test."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    requetes = pd.read_csv(DOSSIER / "annotations" / "requetes_retrieve.csv")
    juges = pd.read_csv(DOSSIER / "annotations" / "jugements_retrieve.csv")
    jugements = juges[juges.pertinent == 1].groupby("id_requete")["id_projet"].apply(set).to_dict()

    documents = nlp.charger_documents()
    tfidf = nlp.construire_index("tfidf", documents)
    emb = nlp.construire_index("embeddings", documents)

    reglage = requetes[requetes.jeu == "reglage"]
    choix = []
    for alpha in ALPHAS:
        res = evaluer(nlp.IndexHybride(tfidf, emb, alpha), reglage, jugements)
        choix.append({"alpha": alpha, **res[["precision@3", "mrr"]].mean().to_dict()})
    choix = pd.DataFrame(choix)
    journal.info("Réglage de alpha (15 requêtes) :\n%s", choix.round(3).to_string(index=False))
    meilleur = choix.sort_values(["precision@3", "mrr", "alpha"], ascending=[False, False, True]).iloc[0]
    alpha = float(meilleur["alpha"])
    journal.info("alpha retenu : %.1f", alpha)

    moteurs = {"tfidf": tfidf, "embeddings": emb, "hybride": nlp.IndexHybride(tfidf, emb, alpha)}
    detail = pd.concat([evaluer(idx, requetes, jugements).assign(moteur=nom) for nom, idx in moteurs.items()])
    detail.to_csv(DOSSIER / "resultats_retrieve.csv", index=False)
    choix.to_csv(DOSSIER / "reglage_alpha_retrieve.csv", index=False)

    test = detail[detail.jeu == "test"]
    resume = test.groupby("moteur")[["precision@3", "precision@5", "mrr", "rappel@10"]].mean()
    journal.info("Jeu de test (25 requêtes) :\n%s", resume.loc[list(moteurs)].round(3).to_string())
    pivot = test.pivot(index="id_requete", columns="moteur", values="precision@3")
    for a, b in (("embeddings", "tfidf"), ("hybride", "embeddings"), ("hybride", "tfidf")):
        bas, haut = intervalle_difference(pivot[a], pivot[b])
        gagne, perd = int((pivot[a] > pivot[b]).sum()), int((pivot[a] < pivot[b]).sum())
        journal.info("précision@3 %s - %s : %+.3f [%+.3f ; %+.3f], %d requêtes gagnées, %d perdues",
                     a, b, (pivot[a] - pivot[b]).mean(), bas, haut, gagne, perd)


if __name__ == "__main__":
    main()
