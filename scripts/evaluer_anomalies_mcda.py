"""Évalue detect_anomalies avec la vérité terrain (rôle 2, tâche 4).

anomalies_injectees.csv n'est lu qu'ici, après la détection : il ne sert ni à apprendre
ni à régler le seuil (contamination fixée à 5 % avant cette évaluation).

Écrit dans reports/role2/ :
- anomalies_evaluation.csv : précision, rappel, faux positifs (tous les projets, projets en
  cours, par type d'anomalie) ;
- anomalies_sensibilite.csv : mêmes mesures pour d'autres contaminations (analyse de
  sensibilité seulement, detect_anomalies garde 5 %) ;
- anomalies_detectees.csv : les projets signalés, avec la variable la plus atypique.

    .venv\\Scripts\\python.exe scripts/evaluer_anomalies_mcda.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from src.mcda import CONTAMINATION, FICHIER_PROJETS, detect_anomalies, score_anomalie, variables_anomalies  # noqa: E402

journal = logging.getLogger("evaluer_anomalies_mcda")
FICHIER_VERITE = RACINE / "data" / "processed" / "anomalies_injectees.csv"
DOSSIER_SORTIE = RACINE / "reports" / "role2"
CONTAMINATIONS_SENSIBILITE = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
NOMS_VARIABLES = {"log_cout": "coût", "log_duree": "durée", "log_cout_par_mois": "coût par mois"}


def mesures(signale: pd.Series, vrai: pd.Series, perimetre: str) -> dict:
    """Précision, rappel et nombre de faux positifs sur un ensemble de projets."""
    vp = int((signale & vrai).sum())
    fp = int((signale & ~vrai).sum())
    fn = int((~signale & vrai).sum())
    return {"perimetre": perimetre, "projets": len(signale), "anomalies_reelles": int(vrai.sum()),
            "signales": int(signale.sum()), "vrais_positifs": vp, "faux_positifs": fp, "manquees": fn,
            "precision": round(vp / (vp + fp), 3) if vp + fp else float("nan"),
            "rappel": round(vp / (vp + fn), 3) if vp + fn else float("nan")}


def evaluer(projets: pd.DataFrame, signale: pd.Series, verite: pd.DataFrame) -> list[dict]:
    """Mesures sur tous les projets, les projets en cours, et rappel par type d'anomalie."""
    vrai = projets["id_projet"].isin(verite["id_projet"])
    detectables = projets["id_projet"].isin(verite.loc[verite["type_anomalie"] != "rentabilite_extreme", "id_projet"])
    en_cours = projets["statut"] == "en_cours"
    lignes = [mesures(signale, vrai, "tous les projets"),
              mesures(signale, detectables, "tous, sans les rentabilités extrêmes"),
              mesures(signale[en_cours], vrai[en_cours], "projets en cours")]
    for type_anomalie, groupe in verite.groupby("type_anomalie"):
        dans_type = projets["id_projet"].isin(groupe["id_projet"])
        ligne = mesures(signale[dans_type], dans_type[dans_type], f"type : {type_anomalie}")
        lignes.append({**ligne, "precision": float("nan"), "faux_positifs": float("nan")})
    return lignes


def expliquer(projets: pd.DataFrame) -> pd.Series:
    """Pour chaque projet, la variable la plus éloignée de la normale et dans quel sens.

    Écart robuste : (valeur - médiane) / écart interquartile, calculé sur les 2 000 projets.
    """
    x = variables_anomalies(projets)
    ecart = (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25))
    variable = ecart.abs().idxmax(axis=1)
    return pd.Series([f"{NOMS_VARIABLES[v]} {'très élevé' if ecart.at[i, v] > 0 else 'très faible'} "
                      f"({ecart.at[i, v]:+.1f} écarts interquartiles)" for i, v in variable.items()],
                     index=projets.index)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    projets = pd.read_csv(FICHIER_PROJETS)

    # 1. Détection, sans la vérité terrain.
    sortie = detect_anomalies(projets)
    signale = sortie["anomalie"]

    # 2. Seulement maintenant : lecture de la vérité terrain, pour mesurer.
    verite = pd.read_csv(FICHIER_VERITE)
    evaluation = pd.DataFrame(evaluer(projets, signale, verite))
    evaluation.insert(0, "contamination", CONTAMINATION)
    evaluation.to_csv(DOSSIER_SORTIE / "anomalies_evaluation.csv", index=False)
    journal.info("Évaluation (contamination %.0f %%) :\n%s", 100 * CONTAMINATION, evaluation.to_string(index=False))

    # 3. Liste des projets signalés, avec une explication.
    types = verite.set_index("id_projet")["type_anomalie"]
    liste = projets.loc[signale, ["id_projet", "nom", "secteur", "pays", "statut", "cout_total", "duree_mois"]].copy()
    liste["score_anomalie"] = score_anomalie(projets)[signale].round(3)
    liste["explication"] = expliquer(projets)[signale]
    liste["anomalie_injectee"] = liste["id_projet"].map(types).fillna("non (faux positif)")
    liste.sort_values("score_anomalie", ascending=False).to_csv(DOSSIER_SORTIE / "anomalies_detectees.csv", index=False)

    # 4. Sensibilité à la contamination (mêmes variables, mêmes graines).
    x = variables_anomalies(projets)
    x = x.fillna(x.median())
    lignes = []
    for c in CONTAMINATIONS_SENSIBILITE:
        modele = IsolationForest(n_estimators=300, contamination=c, random_state=42).fit(x)
        signale_c = pd.Series(modele.predict(x) == -1, index=projets.index)
        for ligne in evaluer(projets, signale_c, verite)[:3]:
            lignes.append({"contamination": c, **ligne})
    sensibilite = pd.DataFrame(lignes)
    sensibilite.to_csv(DOSSIER_SORTIE / "anomalies_sensibilite.csv", index=False)
    journal.info("Sensibilité (tous les projets) :\n%s",
                 sensibilite[sensibilite["perimetre"] == "tous les projets"].to_string(index=False))


if __name__ == "__main__":
    main()
