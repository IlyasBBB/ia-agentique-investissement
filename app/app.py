"""Application Streamlit : analyse et classement de projets d'investissement.

L'application ne calcule rien elle-même : elle charge les projets, transmet la
demande à l'agent (src/agent.py) et affiche ce qu'il rend. Lancement :
    streamlit run app/app.py

Ce dossier contient aussi :
- style.css      : l'habillage (papier saumon, manchette, cartes, journal) ;
- composants.py  : les blocs écrits en HTML (bandeau, cartes, alerte, journal) ;
- graphiques.py  : les graphiques Plotly, dont le nuage 3D ;
- scene3d/       : le relief 3D du portefeuille (Three.js), composant maison.
"""
from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DOSSIER_APP = Path(__file__).resolve().parent
RACINE = DOSSIER_APP.parent
# Pour trouver src/ et les deux modules voisins quand Streamlit lance ce fichier.
sys.path[:0] = [str(RACINE), str(DOSSIER_APP)]

import composants  # noqa: E402
import graphiques  # noqa: E402
from graphiques import LIBELLES  # noqa: E402
from src import outils  # noqa: E402
from src.agent import MODELE_PAR_DEFAUT, normaliser_poids, pipeline_fixe, run_agent  # noqa: E402
from src.contrat import CRITERES, POIDS_PAR_DEFAUT  # noqa: E402

NOM_PRODUIT = "InvestMind"
ACCROCHE = ("Un agent prédit la rentabilité et le risque, repère les dossiers anormaux, classe les projets "
            "selon plusieurs critères et rédige une recommandation justifiée.")
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
DOSSIER_DOCUMENTS = RACINE / "data" / "docs"
FICHIER_STYLE = DOSSIER_APP / "style.css"
FICHIER_SHAP = RACINE / "reports" / "role1" / "shap_importance.csv"  # fourni par le rôle 1
COLONNES_OBLIGATOIRES = ["id_projet", "nom", "secteur", "pays", "cout_total", "duree_mois"]
MODE_AGENT = "Agent (LLM)"
MODE_FIXE = "Pipeline fixe (sans LLM)"
EXEMPLES = [
    "Je veux des projets peu risqués avec un fort impact environnemental.",
    "Je cherche les projets les plus rentables, même s'ils sont un peu risqués.",
    "Je veux des projets rapides et peu coûteux, faciles à mettre en œuvre.",
]
COLONNES_TABLEAU = {
    "rang": "Rang", "id_projet": "Projet", "nom": "Nom", "secteur": "Secteur", "pays": "Pays", "score": "Score",
    "rentabilite_predite": "Rentabilité prédite (%)", "proba_risque": "Proba. risque élevé",
    "risque_predit": "Risque prédit", "cout_total": "Coût (M$)", "duree_mois": "Durée (mois)",
    "impact_env_social": "Impact /5", "alignement_strategique": "Alignement /5", "faisabilite": "Faisabilité /5",
    "anomalie": "Dossier anormal",
}
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
        "novembre", "décembre"]
# Relief 3D : page HTML de app/scene3d servie par Streamlit, qui garde la même scène d'un clic à l'autre.
relief_3d = components.declare_component("relief_portefeuille", path=str(DOSSIER_APP / "scene3d"))


# --- Chargement des données ------------------------------------------------

@st.cache_data(show_spinner="Chargement des projets et des documents (une seule fois)…")
def charger_jeu_banque_mondiale() -> tuple[pd.DataFrame, dict[str, str]]:
    """Lit projets.csv et un document texte par projet. Mis en cache : la lecture des textes est lente."""
    projets = pd.read_csv(FICHIER_PROJETS)
    documents = {f.stem: f.read_text(encoding="utf-8") for f in DOSSIER_DOCUMENTS.glob("*.txt")}
    return projets, documents


@st.cache_resource(show_spinner="Chargement des modèles de texte (une seule fois)…")
def prechauffer_modeles(_exemple: str) -> None:
    """Premier appel à extract et retrieve dès l'ouverture : la première analyse n'attend pas leur chargement."""
    outils.extract(_exemple)
    outils.retrieve("environmental impact", 1)


def charger_fichiers_importes(csv, textes) -> tuple[pd.DataFrame, dict[str, str]] | None:
    """Lit un CSV et des fichiers texte importés. Les textes sont associés au projet par leur nom (P001.txt)."""
    projets = pd.read_csv(csv)
    manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in projets.columns]
    if manquantes:
        st.error(f"Colonnes manquantes dans le CSV : {', '.join(manquantes)}. Voir docs/contrat.md.")
        return None
    projets["id_projet"] = projets["id_projet"].astype(str)
    documents = {Path(f.name).stem: f.getvalue().decode("utf-8", errors="replace") for f in textes}
    return projets, documents


@st.cache_data(ttl=30, show_spinner=False)
def etat_ollama() -> tuple[bool, list[str]]:
    """Vrai si le serveur Ollama répond, avec la liste des modèles installés."""
    try:
        import ollama

        modeles = ollama.Client(timeout=3).list().models
        return True, [m.model for m in modeles]
    except Exception:
        return False, []


# --- Petites fonctions d'affichage -------------------------------------------

def afficher_html(bloc: str) -> None:
    """Affiche un bloc HTML préparé par composants.py (les textes des données y sont déjà échappés)."""
    st.markdown(bloc, unsafe_allow_html=True)


def appliquer_style() -> None:
    """Injecte la feuille de style de l'application (app/style.css)."""
    afficher_html(f"<style>{FICHIER_STYLE.read_text(encoding='utf-8')}</style>")


def afficher_graphique(figure) -> None:
    """Affiche une figure Plotly avec son propre modèle, sans le thème par défaut de Streamlit."""
    st.plotly_chart(figure, width="stretch", theme=None, config={"displaylogo": False})


def poids_utilises(etapes: list[dict], poids_fixes: dict | None) -> dict[str, float]:
    """Retrouve les poids du classement : ceux de l'utilisateur, sinon ceux du dernier appel à classer."""
    if poids_fixes:
        return normaliser_poids(poids_fixes)
    appels = [e for e in etapes if e["outil"] == "classer"]
    return normaliser_poids(appels[-1]["arguments"].get("poids") if appels else None)


def trois_premiers(classement: pd.DataFrame) -> pd.DataFrame:
    """Les trois premiers projets, dossiers anormaux écartés (même règle que l'agent)."""
    if "anomalie" in classement:
        classement = classement[~classement["anomalie"].astype(bool)]
    return classement.head(3)


def dossiers_anormaux(classement: pd.DataFrame) -> list[str]:
    """Identifiants des dossiers signalés comme anormaux dans le classement."""
    if "anomalie" not in classement:
        return []
    return classement.loc[classement["anomalie"].astype(bool), "id_projet"].astype(str).tolist()


# --- Les onglets ---------------------------------------------------------------

def afficher_recommandation(resultat: dict) -> None:
    """Onglet Recommandation : cartes des trois projets, alerte, texte rédigé."""
    classement = resultat["classement"]
    recommandes = trois_premiers(classement)
    if not recommandes.empty:
        afficher_html(composants.cartes_recommandes(recommandes))
    anormaux = dossiers_anormaux(classement)
    if anormaux:
        afficher_html(composants.alerte_anomalies(anormaux))
    st.subheader("Recommandation rédigée")
    st.text(resultat["recommandation"])  # texte brut : on n'interprète pas le texte du LLM comme du Markdown
    st.caption(f"Durée de l'analyse : {resultat['duree_s']:.1f} s")


def afficher_classement(classement: pd.DataFrame) -> None:
    """Onglet Classement : les dix meilleurs scores en barres, puis le tableau complet."""
    if classement.empty:
        st.info("Aucun classement : voir le journal des étapes.")
        return
    st.subheader("Les 10 meilleurs scores")
    afficher_graphique(graphiques.figure_meilleurs_scores(classement))
    st.subheader("Classement complet")
    tableau = classement[[c for c in COLONNES_TABLEAU if c in classement]]
    if "anomalie" in tableau:  # lignes des dossiers anormaux sur fond rose pâle
        surligner = lambda ligne: ["background-color: #f7dcd9; color: #5c0f24" if ligne["anomalie"] else ""] * len(ligne)  # noqa: E731
        tableau = tableau.style.apply(surligner, axis=1).format(precision=3, decimal=",")
    st.dataframe(tableau, hide_index=True, width="stretch", column_config=COLONNES_TABLEAU)


def afficher_vue_3d(classement: pd.DataFrame) -> None:
    """Onglet Vue 3D : chaque projet placé selon rentabilité, risque et impact."""
    figure = graphiques.figure_nuage_3d(classement, trois_premiers(classement)["id_projet"].tolist())
    if figure is None:
        st.info("Vue 3D indisponible : le classement n'a pas été produit.")
        return
    afficher_html('<p class="lecture">Chaque point est un projet. <b>Position</b> : rentabilité prédite, '
                  'probabilité de risque élevé, impact environnemental et social. <b>Couleur</b> : score '
                  '(plus foncé = meilleur). <b>Taille</b> : coût total. <b>Losanges ocre</b> : les trois projets '
                  'recommandés. <b>Croix bordeaux</b> : dossiers anormaux. Faites glisser pour tourner, '
                  'molette pour zoomer, survolez un point pour sa fiche.</p>')
    afficher_graphique(figure)


def afficher_journal(etapes: list[dict]) -> None:
    """Onglet Journal : chaque outil appelé, dans l'ordre, avec arguments, résultat et durée."""
    st.caption("Chaque outil appelé, avec ses arguments, son résultat et sa durée.")
    if not etapes:
        st.info("Aucune étape.")
        return
    afficher_html(composants.journal_etapes(etapes))


@st.cache_data(show_spinner=False)
def lire_shap(fichier: Path = FICHIER_SHAP) -> pd.DataFrame | None:
    """Importance SHAP par variable (colonnes variable, importance), ou None si le fichier manque ou est invalide."""
    try:
        importances = pd.read_csv(fichier)
    except (OSError, ValueError):
        return None
    if not {"variable", "importance"} <= set(importances.columns) or importances.empty:
        return None
    importances["importance"] = pd.to_numeric(importances["importance"], errors="coerce")
    return importances.dropna(subset=["importance"])


def afficher_explication(resultat: dict, poids_fixes: dict | None) -> None:
    """Onglet Explication : ce qui pèse sur la rentabilité prédite (SHAP), puis les poids du classement."""
    importances = lire_shap()
    if importances is None:
        st.info("Graphique SHAP de la prédiction de rentabilité : fichier reports/role1/shap_importance.csv absent.")
    else:
        st.subheader("Ce qui pèse sur la rentabilité prédite")
        afficher_graphique(graphiques.figure_shap(importances))
        note = ("Valeur SHAP moyenne de chaque variable : de combien, en moyenne, elle écarte la rentabilité "
                "prédite de la moyenne. Modèle Ridge du rôle 1, appris sur les 1 500 projets terminés.")
        if "supervise" not in outils.modules_reels():
            note += " Les prédictions affichées viennent en ce moment de la fausse fonction predict."
        st.caption(note)
    st.subheader("Poids utilisés pour le classement")
    afficher_graphique(graphiques.figure_poids(poids_utilises(resultat["etapes"], poids_fixes)))


def afficher_comparaison(agent: dict, reference: dict, poids_fixes: dict | None) -> None:
    """Onglet Comparaison : l'agent face au pipeline fixe (durée, trois premiers, poids)."""
    premiers = {titre: trois_premiers(r["classement"])["id_projet"].astype(str).tolist()
                for titre, r in ((MODE_AGENT, agent), (MODE_FIXE, reference))}
    panneaux = [composants.panneau_mode(titre, couleur, r["duree_s"], premiers[titre], len(r["etapes"]))
                for titre, couleur, r in ((MODE_AGENT, graphiques.BLEU, agent), (MODE_FIXE, graphiques.OCRE, reference))]
    afficher_html(f'<div class="duel">{"".join(panneaux)}</div>')
    communs = set(premiers[MODE_AGENT]) & set(premiers[MODE_FIXE])
    st.info(f"{len(communs)} projet(s) en commun dans les trois premiers. Le pipeline fixe utilise toujours les "
            "mêmes poids ; l'agent les adapte à la demande.")
    st.subheader("Poids utilisés")
    afficher_graphique(graphiques.figure_comparaison_poids(
        poids_utilises(agent["etapes"], poids_fixes), poids_utilises(reference["etapes"], poids_fixes),
        MODE_AGENT, MODE_FIXE))


# --- Poids en direct et relief 3D ---------------------------------------------------

def reclasser(classement: pd.DataFrame, poids: dict[str, float]) -> pd.DataFrame:
    """Refait le classement avec d'autres poids, sans relancer l'agent : rank sur les sept critères déjà calculés."""
    nouveau = outils.rank(classement[["id_projet", *CRITERES]], poids)
    infos = classement.drop(columns=[c for c in ("score", "rang", *CRITERES) if c in classement])
    return nouveau.merge(infos, on="id_projet", how="left")


def projets_pour_relief(projets: pd.DataFrame, classement: pd.DataFrame | None) -> list[dict]:
    """Ce que reçoit la scène 3D : une ligne courte par projet (identité, rang, score, recommandé, anormal)."""
    if classement is None:  # avant l'analyse : colonnes au repos, dans l'ordre du fichier
        lignes = projets.assign(rang=range(1, len(projets) + 1), score=None, anomalie=False)
        recommandes = []
    else:
        lignes = classement
        recommandes = trois_premiers(classement)["id_projet"].tolist()
    return [{"id": str(p["id_projet"]), "nom": str(p.get("nom", ""))[:80], "secteur": str(p.get("secteur", "")),
             "pays": str(p.get("pays", "")), "rang": int(p["rang"]),
             "score": None if pd.isna(p["score"]) else round(float(p["score"]), 4),
             "anomalie": bool(p["anomalie"]), "reco": p["id_projet"] in recommandes}
            for p in lignes.to_dict("records")]


def date_du_jour() -> str:
    """La date à la française, pour la manchette : « mardi 17 novembre 2026 »."""
    jour = datetime.date.today()
    return f"{JOURS[jour.weekday()]} {jour.day} {MOIS[jour.month - 1]} {jour.year}"


def demander_analyse() -> None:
    """Appelé au clic sur « Analyser » : l'analyse se lance au début de la mise à jour suivante."""
    st.session_state["analyse_demandee"] = True


def lancer_analyse(demande: str, projets: pd.DataFrame, documents: dict, mode: str, comparer: bool,
                   poids_fixes: dict | None) -> None:
    """Lance l'agent (ou le pipeline fixe), garde le résultat en session et cale les curseurs sur ses poids."""
    message = ("L'agent lit la demande, choisit ses outils et rédige la recommandation (environ une minute)…"
               if mode == MODE_AGENT else "Le pipeline fixe appelle les cinq outils dans l'ordre…")
    with st.spinner(message):
        debut = time.perf_counter()
        if mode == MODE_AGENT:
            resultat = run_agent(demande, projets, documents, poids_fixes)
        else:
            resultat = pipeline_fixe(demande, projets, documents, poids_fixes)
        resultat["duree_s"] = time.perf_counter() - debut
        reference = None
        if comparer:
            debut = time.perf_counter()
            reference = pipeline_fixe(demande, projets, documents, poids_fixes)
            reference["duree_s"] = time.perf_counter() - debut
    poids = poids_utilises(resultat["etapes"], poids_fixes)
    # Gardé en mémoire de session : changer d'onglet ou bouger un curseur ne relance pas l'analyse.
    st.session_state.update(resultat=resultat, reference=reference, poids_analyse=poids_fixes, poids_resultat=poids)
    for critere, valeur in poids.items():  # les curseurs partent des poids de l'analyse
        st.session_state[f"poids_{critere}"] = round(valeur, 2)


def afficher_resultats(resultat: dict, classement: pd.DataFrame, recalcule: bool,
                       poids_curseurs: dict | None) -> None:
    """Les onglets de résultats, sous la demande et le relief."""
    reference = st.session_state["reference"]
    poids_analyse = st.session_state["poids_analyse"]
    if recalcule:
        afficher_html('<p class="note vive">Cartes et classement suivent vos poids. La recommandation rédigée, '
                      'le journal et la comparaison restent ceux de l\'analyse.</p>')
    noms = ["Recommandation", "Classement", "Vue 3D", "Journal des étapes", "Explication"]
    onglets = st.tabs(noms + (["Comparaison"] if reference else []))
    with onglets[0]:
        afficher_recommandation({**resultat, "classement": classement})
    with onglets[1]:
        afficher_classement(classement)
    with onglets[2]:
        afficher_vue_3d(classement)
    with onglets[3]:
        afficher_journal(resultat["etapes"])
    with onglets[4]:
        afficher_explication(resultat, poids_curseurs if recalcule else poids_analyse)
    if reference:
        with onglets[5]:
            afficher_comparaison(resultat, reference, poids_analyse)


@st.fragment
def atelier(projets: pd.DataFrame, documents: dict, mode: str, comparer: bool) -> None:
    """Demande, poids, relief 3D et résultats. Fragment : un clic ici ne recharge que ce bloc, pas la page."""
    if st.session_state.pop("analyse_demandee", False) and st.session_state.get("demande", "").strip():
        fixes = st.session_state.get("choix_poids") == "Je les fixe moi-même"
        poids_fixes = normaliser_poids({c: st.session_state.get(f"poids_{c}", POIDS_PAR_DEFAUT[c])
                                        for c in CRITERES}) if fixes else None
        lancer_analyse(st.session_state["demande"], projets, documents, mode, comparer, poids_fixes)
    resultat = st.session_state.get("resultat")

    gauche, droite = st.columns([5, 7], gap="large")
    with gauche, st.container(key="panneau_demande"):
        st.subheader("Votre demande")
        for exemple in EXEMPLES:
            st.button(exemple, on_click=lambda e=exemple: st.session_state.update(demande=e), width="stretch")
        demande = st.text_area("Décrivez vos priorités", key="demande", placeholder=EXEMPLES[0])
        choix_poids = st.radio("Poids des critères", ["L'agent les choisit selon la demande", "Je les fixe moi-même"],
                               horizontal=True, key="choix_poids")
        poids_curseurs = None
        if choix_poids == "Je les fixe moi-même" or resultat:
            if resultat:
                afficher_html('<p class="note vive">Bougez un curseur : le classement et le relief se recalculent '
                              'aussitôt, sans relancer l\'agent.</p>')
            colonnes = st.columns(2)
            for c in CRITERES:  # valeur de départ en session (et pas en argument) : lancer_analyse peut la changer
                st.session_state.setdefault(f"poids_{c}", POIDS_PAR_DEFAUT[c])
            bruts = {c: colonnes[i % 2].slider(LIBELLES[c], 0.0, 1.0, step=0.05, key=f"poids_{c}")
                     for i, c in enumerate(CRITERES)}
            poids_curseurs = normaliser_poids(bruts)  # ramenés à une somme de 1
        elif mode == MODE_FIXE:
            afficher_html('<p class="note">Le pipeline fixe ne lit pas la demande pour les poids : '
                          'il utilise les poids par défaut.</p>')
        st.button("Analyser", type="primary", key="analyser", disabled=not demande.strip(),
                  on_click=demander_analyse, width="stretch")

    # Le classement affiché suit les curseurs dès qu'ils s'écartent des poids de l'analyse.
    classement = resultat["classement"] if resultat else None
    recalcule = False
    if resultat and not classement.empty and poids_curseurs:
        ecart = max(abs(poids_curseurs[c] - st.session_state["poids_resultat"][c]) for c in CRITERES)
        if ecart > 0.005:
            classement, recalcule = reclasser(classement, poids_curseurs), True

    with droite, st.container(key="zone_relief"):
        if not resultat:
            etat = f"{len(projets)} projets au repos · lancez une analyse pour les classer"
        elif recalcule:
            etat = "Classement recalculé avec vos poids"
        else:
            etat = f"Classement de l'analyse · {len(classement)} projets"
        choisi = relief_3d(projets=projets_pour_relief(projets, classement), hauteur=520, etat=etat,
                           key="relief", default=None)
        if choisi and classement is not None and choisi in set(classement["id_projet"].astype(str)):
            ligne = classement[classement["id_projet"].astype(str) == choisi].iloc[0].to_dict()
            afficher_html(composants.fiche_projet(ligne))
        elif resultat:
            afficher_html('<p class="note">Cliquez une colonne du relief pour afficher la fiche du projet.</p>')

    if resultat:
        afficher_resultats(resultat, classement, recalcule, poids_curseurs)
    else:
        afficher_html('<p class="attente">Aucune analyse pour l\'instant. Décrivez vos priorités ou choisissez un '
                      'exemple, puis lancez l\'analyse : les projets du relief se classeront, et la recommandation, '
                      'le classement et le journal des étapes s\'afficheront ici.</p>')


# --- La page ------------------------------------------------------------------

st.set_page_config(page_title=f"{NOM_PRODUIT} · sélection de projets d'investissement", layout="wide")
appliquer_style()
st.title(NOM_PRODUIT)

with st.sidebar:
    st.header("Données")
    source = st.radio("Source", ["Projets Banque mondiale", "Importer mes fichiers"], key="source")
    if source == "Projets Banque mondiale":
        jeu = charger_jeu_banque_mondiale()
        st.caption("Données réelles de la Banque mondiale (CC BY 4.0). Rentabilité et risque sont simulés : "
                   "voir data/README.md.")
    else:
        csv = st.file_uploader("Projets (CSV)", type="csv")
        textes = st.file_uploader("Documents (un .txt par projet, nommé d'après id_projet)", type="txt",
                                  accept_multiple_files=True)
        jeu = charger_fichiers_importes(csv, textes) if csv else None
    if jeu is None:
        st.info("Importez un fichier CSV pour commencer.")
        st.stop()
    projets, documents = jeu
    nb_charges = len(projets)
    if "nlp" in outils.modules_reels() and documents:
        prechauffer_modeles(next(iter(documents.values())))

    st.header("Projets à analyser")
    if "statut" in projets:
        statuts = st.multiselect("Statut", sorted(projets["statut"].dropna().unique()),
                                 default=["en_cours"] if "en_cours" in set(projets["statut"]) else None,
                                 help="« en_cours » : les dossiers à évaluer, dont la rentabilité est inconnue.")
        if statuts:
            projets = projets[projets["statut"].isin(statuts)]
    secteurs = st.multiselect("Secteurs (tous si vide)", sorted(projets["secteur"].dropna().unique()),
                              placeholder="Tous les secteurs")
    if secteurs:
        projets = projets[projets["secteur"].isin(secteurs)]
    if "region" in projets:
        regions = st.multiselect("Régions (toutes si vide)", sorted(projets["region"].dropna().unique()),
                                 placeholder="Toutes les régions")
        if regions:
            projets = projets[projets["region"].isin(regions)]
    if projets.empty:
        st.warning("Aucun projet ne correspond aux filtres.")
        st.stop()
    maximum = st.number_input("Nombre maximum de projets", 1, len(projets), min(500, len(projets)), key="maximum")
    projets = projets.head(int(maximum))
    documents = {i: documents[i] for i in projets["id_projet"] if i in documents}
    st.caption(f"{len(projets)} projets retenus, dont {len(documents)} avec un document.")

    st.header("Moteur")
    mode = st.radio("Mode", [MODE_AGENT, MODE_FIXE], key="mode")
    comparer = mode == MODE_AGENT and st.checkbox("Comparer avec le pipeline fixe", value=True, key="comparer")
    ollama_ok, modeles = etat_ollama()
    modele = os.getenv("OLLAMA_MODEL") or MODELE_PAR_DEFAUT
    if not ollama_ok:
        st.warning("Ollama ne répond pas : lancez `ollama serve`, ou choisissez le pipeline fixe.")
    elif not any(m.startswith(modele) for m in modeles):
        st.warning(f"Modèle {modele} absent : `ollama pull {modele}`.")
    else:
        st.success(f"Ollama prêt, modèle {modele}.")
    st.caption(f"Modules : {outils.description_modules()}")

reels = outils.modules_reels()
modules = ("fonctions factices" if not reels else "modèles de l'équipe" if not outils.utilise_mocks()
           else "modules réels : " + ", ".join(sorted(reels)))
afficher_html(f'<div class="manchette"><span><b>{composants.propre(date_du_jour().capitalize())}</b> · '
              f'Banque mondiale, {composants.nombre(nb_charges)} dossiers</span>'
              f'<span>Rentabilité et risque simulés · {composants.propre(modules)} · LLM local</span></div>')
afficher_html(f'<p class="accroche">{composants.propre(ACCROCHE)}</p>')
afficher_html(composants.bandeau_portefeuille(projets, len(documents), nb_charges))
atelier(projets, documents, mode, comparer)
