"""Télécharge les projets de la Banque mondiale et les écrit tels quels dans data/raw/.

Source : API publique « Projects & Operations » de la Banque mondiale (licence CC BY 4.0).
Le fichier produit n'est pas modifié ensuite : toute la préparation se fait dans
scripts/preparer_donnees.py.

Utilisation :
    python scripts/telecharger_donnees.py            # ne fait rien si le fichier existe déjà
    python scripts/telecharger_donnees.py --force    # retélécharge
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

# --- Constantes ---------------------------------------------------------------
URL_API = "https://search.worldbank.org/api/v3/projects"
# Champs demandés à l'API (noms vérifiés dans la réponse de la version 3).
CHAMPS = [
    "id", "project_name", "countryshortname", "countryname", "countrycode", "regionname",
    "major_sectors", "major_sector_name", "projectfinancialtype",
    "lendprojectcost", "totalamt", "curr_project_cost",
    "boardapprovaldate", "closingdate", "fiscalyear", "status",
    "pdo", "project_abstract",
]
PROJETS_PAR_PAGE = 500
PAUSE_ENTRE_PAGES_S = 1.0
ESSAIS_MAX = 6

RACINE = Path(__file__).resolve().parents[1]
FICHIER_BRUT = RACINE / "data" / "raw" / "banque_mondiale_projets.json"


def telecharger_page(decalage: int) -> dict:
    """Demande une page de projets. En cas d'erreur (429, coupure), attend puis réessaie."""
    parametres = urllib.parse.urlencode(
        {"format": "json", "rows": PROJETS_PAR_PAGE, "os": decalage, "fl": ",".join(CHAMPS)}
    )
    requete = urllib.request.Request(f"{URL_API}?{parametres}", headers={"User-Agent": "python-urllib"})
    for essai in range(1, ESSAIS_MAX + 1):
        try:
            with urllib.request.urlopen(requete, timeout=120) as reponse:
                return json.loads(reponse.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as erreur:
            attente = 5 * 2 ** (essai - 1)  # 5 s, 10 s, 20 s, 40 s...
            print(f"  Erreur ({erreur}), nouvel essai dans {attente} s")
            time.sleep(attente)
    raise RuntimeError(f"Page à partir de {decalage} : échec après {ESSAIS_MAX} essais.")


def main() -> None:
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--force", action="store_true", help="retélécharger même si le fichier existe")
    options = analyseur.parse_args()

    if FICHIER_BRUT.exists() and not options.force:
        print(f"{FICHIER_BRUT} existe déjà. Ajouter --force pour retélécharger.")
        return

    projets = {}
    decalage = 0
    total = None
    while total is None or decalage < total:
        page = telecharger_page(decalage)
        total = int(page["total"])
        projets.update(page["projects"])  # dictionnaire {identifiant: projet}
        decalage += PROJETS_PAR_PAGE
        print(f"{min(decalage, total)} / {total} projets lus")
        time.sleep(PAUSE_ENTRE_PAGES_S)

    contenu = {
        "source": "Banque mondiale, Projects & Operations",
        "url": URL_API,
        "licence": "CC BY 4.0",
        "date_telechargement": date.today().isoformat(),
        "champs": CHAMPS,
        "projets": list(projets.values()),
    }
    FICHIER_BRUT.parent.mkdir(parents=True, exist_ok=True)
    FICHIER_BRUT.write_text(json.dumps(contenu, ensure_ascii=False), encoding="utf-8")
    print(f"{len(projets)} projets écrits dans {FICHIER_BRUT}")


if __name__ == "__main__":
    main()
