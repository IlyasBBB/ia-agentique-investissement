"""Prépare le jeu de données du projet à partir du fichier brut de la Banque mondiale.

Étapes :
  1. lire data/raw/banque_mondiale_projets.json (écrit par scripts/telecharger_donnees.py) ;
  2. nettoyer et filtrer les projets ;
  3. tirer au hasard 1 500 projets terminés et 500 projets en cours ;
  4. simuler les deux cibles (rentabilite, risque) avec une formule connue ;
  5. injecter quelques anomalies ;
  6. écrire data/processed/projets.csv, data/processed/anomalies_injectees.csv
     et un document texte par projet dans data/docs/.

ATTENTION : rentabilite et risque sont SIMULÉS. Ce ne sont pas des résultats réels
de la Banque mondiale. Les formules sont dans ce fichier et dans data/README.md.

Utilisation : python scripts/preparer_donnees.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- Constantes ---------------------------------------------------------------
GRAINE = 42  # même graine = mêmes fichiers à chaque exécution

RACINE = Path(__file__).resolve().parents[1]
FICHIER_BRUT = RACINE / "data" / "raw" / "banque_mondiale_projets.json"
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
FICHIER_ANOMALIES = RACINE / "data" / "processed" / "anomalies_injectees.csv"
DOSSIER_DOCS = RACINE / "data" / "docs"

# Filtres
LONGUEUR_MIN_PDO = 80          # caractères : écarte les textes vides ou « TBD »
COUT_MIN_MUSD = 1.0            # millions de dollars
DUREE_MIN_MOIS, DUREE_MAX_MOIS = 6, 180

# Échantillon
NB_TERMINES, NB_EN_COURS = 1500, 500

# Regroupement des grands secteurs de la Banque mondiale en 11 secteurs en français.
# Les préfixes « FY17 - » et « (Historic) » sont retirés avant la recherche.
SECTEURS = {
    "Agriculture, Fishing and Forestry": "Agriculture",
    "Energy and Extractives": "Énergie",
    "Electric Power & Other Energy": "Énergie",
    "Energy and Mineral Resources": "Énergie",
    "Oil & Gas": "Énergie",
    "Mining": "Énergie",
    "Transportation": "Transport",
    "Water, Sanitation and Waste Management": "Eau et assainissement",
    "Health": "Santé",
    "Health and other social services": "Santé",
    "Education": "Éducation",
    "Financial Sector": "Finance",
    "Information and Communications Technologies": "Numérique",
    "Digital Development": "Numérique",
    "Public Administration": "Gouvernance",
    "Economic Policy": "Gouvernance",
    "Social Protection": "Protection sociale",
    "Social Sustainability and Inclusion": "Protection sociale",
    "Industry, Trade and Services": "Industrie et commerce",
    "Industry": "Industrie et commerce",
    "Tourism": "Industrie et commerce",
    "Private Sector Development": "Industrie et commerce",
}

REGIONS = {
    "Latin America and Caribbean": "Amérique latine et Caraïbes",
    "Eastern and Southern Africa": "Afrique de l'Est et australe",
    "Western and Central Africa": "Afrique de l'Ouest et centrale",
    "Africa": "Afrique",
    "Europe and Central Asia": "Europe et Asie centrale",
    "East Asia and Pacific": "Asie de l'Est et Pacifique",
    "South Asia": "Asie du Sud",
    "Middle East, North Africa, Afghanistan, and Pakistan": "Moyen-Orient, Afrique du Nord, Afghanistan et Pakistan",
    "Middle East and North Africa": "Moyen-Orient, Afrique du Nord, Afghanistan et Pakistan",
}

# --- Paramètres de la simulation des cibles ------------------------------------
RENTABILITE_DE_BASE = 9.0  # en %
EFFET_SECTEUR = {
    "Énergie": 2.5, "Numérique": 2.0, "Transport": 1.5, "Finance": 1.0,
    "Industrie et commerce": 1.0, "Agriculture": 0.0, "Eau et assainissement": -0.5,
    "Éducation": -1.0, "Santé": -1.5, "Gouvernance": -2.0, "Protection sociale": -2.0,
}
EFFET_FINANCEMENT = {"IBRD": 1.0, "IDA": 0.0, "Dons": -1.0, "Autre": 0.0}
SECTEURS_INFRASTRUCTURE = ["Énergie", "Transport"]  # les gros projets y sont plus rentables
ECART_TYPE_EFFET_PAYS = 1.5
ECART_TYPE_BRUIT_RENTABILITE = 1.8
ECART_TYPE_EFFET_REGION = 0.4
ECART_TYPE_BRUIT_RISQUE = 0.8
PART_RISQUE_FAIBLE, PART_RISQUE_MOYEN = 0.40, 0.40  # le reste (20 %) est « eleve »

# Anomalies : environ 2 % des projets
PART_ANOMALIES = 0.02


def secteur_principal(projet: dict) -> str | None:
    """Renvoie le secteur français du grand secteur qui a le plus gros pourcentage."""
    meilleur, meilleur_pourcentage = None, -1.0
    for element in projet.get("major_sectors") or []:
        grand_secteur = element["major_sector"]
        pourcentage = sum(float(s.get("sector_percent") or 0) for s in grand_secteur.get("sectors", []))
        if pourcentage > meilleur_pourcentage:
            meilleur, meilleur_pourcentage = grand_secteur["major_sector_name"], pourcentage
    if meilleur is None:
        return None
    nom = meilleur.replace("FY17 - ", "").replace("(Historic)", "").strip()
    return SECTEURS.get(nom)


def type_financement(projet: dict) -> str:
    """IBRD (prêt aux pays à revenu intermédiaire), IDA (pays pauvres), Dons, ou Autre."""
    types = projet.get("projectfinancialtype") or []
    for cle, nom in (("IBRD", "IBRD"), ("IDA", "IDA"), ("Grants", "Dons")):
        if cle in types:
            return nom
    return "Autre"


def nombre(valeur) -> float:
    """Convertit « 1,250,000 » ou « 1250000 » en nombre. Renvoie 0 si c'est impossible."""
    try:
        return float(str(valeur).replace(",", ""))
    except ValueError:
        return 0.0


def texte_propre(valeur) -> str:
    """Retire les retours à la ligne et les espaces en double."""
    return " ".join(str(valeur or "").split())


def lire_et_filtrer() -> tuple[pd.DataFrame, str]:
    """Lit le brut, construit les colonnes du contrat et applique les filtres."""
    brut = json.loads(FICHIER_BRUT.read_text(encoding="utf-8"))
    lignes = []
    for p in brut["projets"]:
        cout = nombre(p.get("lendprojectcost")) or nombre(p.get("totalamt"))  # en dollars
        lignes.append({
            "id_projet": p.get("id"),
            "nom": texte_propre(p.get("project_name")),
            "secteur": secteur_principal(p),
            "pays": texte_propre(p.get("countryshortname")),
            "region": REGIONS.get(p.get("regionname")),
            "cout_total": round(cout / 1e6, 2),
            "approbation": pd.to_datetime(p.get("boardapprovaldate"), errors="coerce", utc=True),
            "cloture": pd.to_datetime(p.get("closingdate"), errors="coerce", utc=True),
            "type_financement": type_financement(p),
            "statut": {"Closed": "termine", "Active": "en_cours"}.get(p.get("status")),
            "pdo": texte_propre(p.get("pdo")),
            "resume": texte_propre(p.get("project_abstract")),
        })
    df = pd.DataFrame(lignes)
    print(f"Brut : {len(df)} projets")

    df["duree_mois"] = ((df["cloture"] - df["approbation"]).dt.days / 30.44).round()
    df["annee_approbation"] = df["approbation"].dt.year
    garder = (
        df["statut"].notna()
        & df["secteur"].notna()
        & df["region"].notna()
        & (df["pays"] != "")
        & (df["nom"] != "")
        & (df["pdo"].str.len() >= LONGUEUR_MIN_PDO)
        & (df["cout_total"] >= COUT_MIN_MUSD)
        & df["duree_mois"].between(DUREE_MIN_MOIS, DUREE_MAX_MOIS)  # faux si une date manque
    )
    df = df[garder].drop_duplicates("id_projet").sort_values("id_projet").reset_index(drop=True)
    df["duree_mois"] = df["duree_mois"].astype(int)
    df["annee_approbation"] = df["annee_approbation"].astype(int)
    print(f"Après filtres : {len(df)} projets ({(df['statut'] == 'termine').sum()} terminés)")
    return df.drop(columns=["approbation", "cloture"]), brut["date_telechargement"]


def centrer_reduire(colonne: pd.Series) -> pd.Series:
    """Ramène une colonne à moyenne 0 et écart-type 1."""
    return (colonne - colonne.mean()) / colonne.std()


def simuler_cibles(df: pd.DataFrame, alea: np.random.Generator) -> pd.DataFrame:
    """Ajoute rentabilite (%) et risque. Les formules sont recopiées dans data/README.md."""
    z_cout = centrer_reduire(np.log10(df["cout_total"]))
    z_duree = centrer_reduire(df["duree_mois"])

    # Un effet tiré au hasard par pays et par région (liste triée pour rester reproductible).
    pays = sorted(df["pays"].unique())
    effet_pays = dict(zip(pays, alea.normal(0, ECART_TYPE_EFFET_PAYS, len(pays))))
    regions = sorted(df["region"].unique())
    effet_region = dict(zip(regions, alea.normal(0, ECART_TYPE_EFFET_REGION, len(regions))))

    infrastructure = df["secteur"].isin(SECTEURS_INFRASTRUCTURE)
    rentabilite = (
        RENTABILITE_DE_BASE
        + df["secteur"].map(EFFET_SECTEUR)
        + df["pays"].map(effet_pays)
        + df["type_financement"].map(EFFET_FINANCEMENT)
        - 0.8 * z_cout                    # les gros projets rapportent un peu moins...
        + 1.0 * z_cout * infrastructure   # ...sauf en énergie et en transport
        - 1.2 * z_duree                   # les projets longs rapportent moins
        - 0.5 * z_duree ** 2              # les durées extrêmes sont pénalisées
        + alea.normal(0, ECART_TYPE_BRUIT_RENTABILITE, len(df))
    )

    # Score de risque caché : élevé si la rentabilité est faible, le projet long ou cher.
    score_risque = (
        -0.6 * centrer_reduire(rentabilite)
        + 0.5 * z_duree
        + 0.4 * z_cout
        + df["region"].map(effet_region)
        + alea.normal(0, ECART_TYPE_BRUIT_RISQUE, len(df))
    )
    # Seuils choisis sur les projets terminés pour obtenir 40 % / 40 % / 20 %.
    score_termines = score_risque[df["statut"] == "termine"]
    seuil_bas = score_termines.quantile(PART_RISQUE_FAIBLE)
    seuil_haut = score_termines.quantile(PART_RISQUE_FAIBLE + PART_RISQUE_MOYEN)

    df["rentabilite"] = rentabilite.round(1)
    df["risque"] = np.where(score_risque <= seuil_bas, "faible", np.where(score_risque <= seuil_haut, "moyen", "eleve"))
    return df


def verifier_apprenabilite(df: pd.DataFrame) -> None:
    """Contrôle rapide : un modèle linéaire simple doit battre le hasard sans être parfait.

    Entraînement sur 80 % des projets terminés, mesure sur les 20 % restants.
    """
    termines = df[df["statut"] == "termine"].sample(frac=1, random_state=GRAINE)
    X = pd.get_dummies(termines[["secteur", "pays", "region", "type_financement"]]).astype(float)
    X["log_cout"] = np.log10(termines["cout_total"])
    X["duree"] = termines["duree_mois"] / 12
    X["constante"] = 1.0
    X = X.to_numpy()
    coupe = int(0.8 * len(termines))

    def ajuster_et_predire(y: np.ndarray) -> np.ndarray:
        # Moindres carrés avec une petite pénalité (Ridge) pour rester stable.
        A = X[:coupe]
        coefficients = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y[:coupe])
        return X[coupe:] @ coefficients

    y = termines["rentabilite"].to_numpy()
    prediction = ajuster_et_predire(y)
    r2 = 1 - ((y[coupe:] - prediction) ** 2).sum() / ((y[coupe:] - y[:coupe].mean()) ** 2).sum()

    # Risque : une régression par classe, on garde la classe au score le plus haut.
    classes = ["faible", "moyen", "eleve"]
    scores = np.column_stack([ajuster_et_predire((termines["risque"] == c).to_numpy(float)) for c in classes])
    risque_predit = np.array(classes)[scores.argmax(axis=1)]
    vrai = termines["risque"].to_numpy()[coupe:]
    exactitude = (risque_predit == vrai).mean()
    reference = pd.Series(vrai).value_counts(normalize=True).max()
    print(f"Apprenabilité (test, {len(vrai)} projets) : R² rentabilité = {r2:.2f} ; "
          f"exactitude risque = {exactitude:.2f} (classe majoritaire : {reference:.2f})")


def injecter_anomalies(df: pd.DataFrame, alea: np.random.Generator) -> pd.DataFrame:
    """Modifie environ 2 % des projets et renvoie la liste des modifications (vérité terrain)."""
    nb = round(PART_ANOMALIES * len(df))
    types = ["cout_aberrant", "duree_incoherente", "rentabilite_extreme"]
    lignes_choisies = alea.choice(df.index, size=nb, replace=False)
    anomalies = []
    for i, ligne in enumerate(lignes_choisies):
        type_anomalie = types[i % 3]
        # La rentabilité n'est connue que pour les projets terminés.
        if type_anomalie == "rentabilite_extreme" and df.at[ligne, "statut"] == "en_cours":
            type_anomalie = "cout_aberrant"

        if type_anomalie == "cout_aberrant":       # coût multiplié par 10 à 50
            colonne = "cout_total"
            nouvelle = round(df.at[ligne, colonne] * alea.uniform(10, 50), 2)
        elif type_anomalie == "duree_incoherente":  # durée de 1 à 3 mois, trop courte pour le coût
            colonne = "duree_mois"
            nouvelle = int(alea.integers(1, 4))
        else:                                       # rentabilité de +35 à +60 % ou de -60 à -35 %
            colonne = "rentabilite"
            nouvelle = round(float(alea.choice([-1, 1]) * alea.uniform(35, 60)), 1)

        anomalies.append({
            "id_projet": df.at[ligne, "id_projet"],
            "type_anomalie": type_anomalie,
            "colonne": colonne,
            "valeur_initiale": df.at[ligne, colonne],
            "valeur_modifiee": nouvelle,
        })
        df.at[ligne, colonne] = nouvelle
    return pd.DataFrame(anomalies).sort_values("id_projet")


def ecrire_documents(df: pd.DataFrame) -> None:
    """Écrit un fichier texte par projet. Le texte (PDO, résumé) vient tel quel de la Banque mondiale."""
    DOSSIER_DOCS.mkdir(parents=True, exist_ok=True)
    for ancien in DOSSIER_DOCS.glob("P*.txt"):  # évite de garder les documents d'un ancien tirage
        ancien.unlink()
    for p in df.to_dict("records"):
        texte = (
            f"Projet : {p['nom']}\n"
            f"Identifiant : {p['id_projet']}\n"
            f"Pays : {p['pays']} ({p['region']})\n"
            f"Secteur : {p['secteur']}\n"
            f"Coût total : {p['cout_total']} millions de dollars\n"
            f"Durée prévue : {p['duree_mois']} mois\n"
            f"Année d'approbation : {p['annee_approbation']}\n"
            f"Statut : {p['statut']}\n\n"
            f"Objectif de développement (PDO) :\n{p['pdo']}\n\n"
            f"Résumé :\n{p['resume'] or 'Non disponible.'}\n"
        )
        (DOSSIER_DOCS / f"{p['id_projet']}.txt").write_text(texte, encoding="utf-8")


def main() -> None:
    alea = np.random.default_rng(GRAINE)
    df, date_telechargement = lire_et_filtrer()
    print(f"Brut téléchargé le {date_telechargement}")

    # Tirage au hasard : 1 500 terminés et 500 en cours.
    termines = df[df["statut"] == "termine"].sample(NB_TERMINES, random_state=GRAINE)
    en_cours = df[df["statut"] == "en_cours"].sample(NB_EN_COURS, random_state=GRAINE)
    df = pd.concat([termines, en_cours]).sort_values("id_projet").reset_index(drop=True)

    df = simuler_cibles(df, alea)
    verifier_apprenabilite(df)

    # Les projets en cours sont les dossiers à évaluer : leurs cibles sont inconnues.
    df.loc[df["statut"] == "en_cours", ["rentabilite", "risque"]] = np.nan

    anomalies = injecter_anomalies(df, alea)

    colonnes = ["id_projet", "nom", "secteur", "pays", "region", "cout_total", "duree_mois",
                "annee_approbation", "type_financement", "statut", "rentabilite", "risque"]
    FICHIER_PROJETS.parent.mkdir(parents=True, exist_ok=True)
    df[colonnes].to_csv(FICHIER_PROJETS, index=False, encoding="utf-8")
    anomalies.to_csv(FICHIER_ANOMALIES, index=False, encoding="utf-8")
    ecrire_documents(df)

    print(f"{len(df)} projets écrits dans {FICHIER_PROJETS}")
    print(f"{len(anomalies)} anomalies écrites dans {FICHIER_ANOMALIES}")
    print(df["statut"].value_counts().to_string())
    print(df["secteur"].value_counts().to_string())
    print(df["risque"].value_counts(normalize=True).round(3).to_string())
    print(df["rentabilite"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
