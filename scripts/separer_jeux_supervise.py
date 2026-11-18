"""Sépare les projets terminés en jeux d'entraînement, de validation et de test (rôle 1).

Seuls les 1 500 projets terminés ont des cibles connues : ce sont eux qu'on sépare.
Les 500 projets en cours ne servent qu'à la prédiction finale par l'agent.

Proportions 70 / 15 / 15, tirage stratifié sur la classe de risque (mêmes parts de
`faible`, `moyen` et `eleve` dans chaque jeu), graine 42. La séparation est faite
AVANT tout prétraitement appris, pour éviter les fuites de données.

On n'écrit que les identifiants (pas de copie des données) :
reports/role1/separation.csv, colonnes `id_projet` et `jeu`
(`entrainement`, `validation` ou `test`).

Utilisation : python scripts/separer_jeux_supervise.py
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

GRAINE = 42
PART_VALIDATION = 0.15
PART_TEST = 0.15

RACINE = Path(__file__).resolve().parents[1]
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
FICHIER_ANOMALIES = RACINE / "data" / "processed" / "anomalies_injectees.csv"
FICHIER_SEPARATION = RACINE / "reports" / "role1" / "separation.csv"

journal = logging.getLogger(__name__)


def separer(projets: pd.DataFrame) -> pd.DataFrame:
    """Rend un tableau `id_projet`, `jeu` pour les projets terminés, stratifié sur le risque."""
    termines = projets[projets["statut"] == "termine"]
    # 1er tirage : on met de côté 30 % (validation + test), stratifié sur le risque.
    entrainement, reste = train_test_split(
        termines, test_size=PART_VALIDATION + PART_TEST,
        stratify=termines["risque"], random_state=GRAINE,
    )
    # 2e tirage : on coupe ces 30 % en deux moitiés égales (15 % + 15 %).
    validation, test = train_test_split(
        reste, test_size=PART_TEST / (PART_VALIDATION + PART_TEST),
        stratify=reste["risque"], random_state=GRAINE,
    )
    separation = pd.concat([
        pd.DataFrame({"id_projet": jeu["id_projet"], "jeu": nom})
        for nom, jeu in [("entrainement", entrainement), ("validation", validation), ("test", test)]
    ])
    return separation.sort_values("id_projet").reset_index(drop=True)


def main() -> None:
    """Calcule la séparation, la vérifie et l'écrit dans reports/role1/separation.csv."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    projets = pd.read_csv(FICHIER_PROJETS)
    separation = separer(projets)

    # Vérifications : chaque projet terminé dans exactement un jeu, aucun projet en cours.
    termines = set(projets.loc[projets["statut"] == "termine", "id_projet"])
    assert set(separation["id_projet"]) == termines
    assert separation["id_projet"].is_unique

    FICHIER_SEPARATION.parent.mkdir(parents=True, exist_ok=True)
    separation.to_csv(FICHIER_SEPARATION, index=False)
    journal.info("Écrit : %s", FICHIER_SEPARATION.relative_to(RACINE).as_posix())

    detail = separation.merge(projets[["id_projet", "risque"]], on="id_projet")
    journal.info("\nProjets par jeu et par classe de risque :\n%s",
                 pd.crosstab(detail["jeu"], detail["risque"], margins=True).to_string())

    # Information seulement (le fichier des anomalies n'entre jamais dans un modèle).
    anomalies = set(pd.read_csv(FICHIER_ANOMALIES)["id_projet"])
    detail["anomalie"] = detail["id_projet"].isin(anomalies)
    journal.info("\nAnomalies injectées par jeu :\n%s",
                 detail.groupby("jeu")["anomalie"].sum().to_string())


if __name__ == "__main__":
    main()
