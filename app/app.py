"""Application Streamlit : analyse et classement de projets d'investissement.

L'application ne calcule rien elle-même : elle charge les projets, transmet la
demande à l'agent (src/agent.py) et affiche ce qu'il rend. Lancement :
    streamlit run app/app.py

Trois fichiers l'accompagnent dans ce dossier :
- style.css      : l'habillage (thème sombre, cartes, journal) ;
- composants.py  : les blocs écrits en HTML (bandeau, cartes, alerte, journal) ;
- graphiques.py  : les graphiques Plotly, dont le nuage 3D.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

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

NOM_PRODUIT = "Atlas Invest"
ACCROCHE = ("Un agent prédit la rentabilité et le risque, repère les dossiers anormaux, classe les projets "
            "selon plusieurs critères et rédige une recommandation justifiée.")
FICHIER_PROJETS = RACINE / "data" / "processed" / "projets.csv"
DOSSIER_DOCUMENTS = RACINE / "data" / "docs"
FICHIER_STYLE = DOSSIER_APP / "style.css"
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


# --- Chargement des données ------------------------------------------------

@st.cache_data(show_spinner="Chargement des projets et des documents (une seule fois)…")
def charger_jeu_banque_mondiale() -> tuple[pd.DataFrame, dict[str, str]]:
    """Lit projets.csv et un document texte par projet. Mis en cache : la lecture des textes est lente."""
    projets = pd.read_csv(FICHIER_PROJETS)
    documents = {f.stem: f.read_text(encoding="utf-8") for f in DOSSIER_DOCUMENTS.glob("*.txt")}
    return projets, documents


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
    """Affiche une figure Plotly avec son propre modèle sombre, sans le thème par défaut de Streamlit."""
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
    if "anomalie" in tableau:  # lignes des dossiers anormaux sur fond rouge sombre
        surligner = lambda ligne: ["background-color: #4a1626; color: #ffd3dc" if ligne["anomalie"] else ""] * len(ligne)  # noqa: E731
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
                  '(plus clair = meilleur). <b>Taille</b> : coût total. <b>Losanges ambre</b> : les trois projets '
                  'recommandés. <b>Croix rouges</b> : dossiers anormaux. Faites glisser pour tourner, '
                  'molette pour zoomer, survolez un point pour sa fiche.</p>')
    afficher_graphique(figure)


def afficher_journal(etapes: list[dict]) -> None:
    """Onglet Journal : chaque outil appelé, dans l'ordre, avec arguments, résultat et durée."""
    st.caption("Chaque outil appelé, avec ses arguments, son résultat et sa durée.")
    if not etapes:
        st.info("Aucune étape.")
        return
    afficher_html(composants.journal_etapes(etapes))


def afficher_explication(resultat: dict, poids_fixes: dict | None) -> None:
    """Onglet Explication : les poids utilisés pour le classement (et bientôt le graphique SHAP)."""
    st.info("Graphique SHAP de la prédiction de rentabilité : fourni par le rôle 1 (à venir).")
    st.subheader("Poids utilisés pour le classement")
    afficher_graphique(graphiques.figure_poids(poids_utilises(resultat["etapes"], poids_fixes)))


def afficher_comparaison(agent: dict, reference: dict, poids_fixes: dict | None) -> None:
    """Onglet Comparaison : l'agent face au pipeline fixe (durée, trois premiers, poids)."""
    premiers = {titre: trois_premiers(r["classement"])["id_projet"].astype(str).tolist()
                for titre, r in ((MODE_AGENT, agent), (MODE_FIXE, reference))}
    panneaux = [composants.panneau_mode(titre, couleur, r["duree_s"], premiers[titre], len(r["etapes"]))
                for titre, couleur, r in ((MODE_AGENT, graphiques.CYAN, agent), (MODE_FIXE, graphiques.AMBRE, reference))]
    afficher_html(f'<div class="duel">{"".join(panneaux)}</div>')
    communs = set(premiers[MODE_AGENT]) & set(premiers[MODE_FIXE])
    st.info(f"{len(communs)} projet(s) en commun dans les trois premiers. Le pipeline fixe utilise toujours les "
            "mêmes poids ; l'agent les adapte à la demande.")
    st.subheader("Poids utilisés")
    afficher_graphique(graphiques.figure_comparaison_poids(
        poids_utilises(agent["etapes"], poids_fixes), poids_utilises(reference["etapes"], poids_fixes),
        MODE_AGENT, MODE_FIXE))


# --- La page ------------------------------------------------------------------

st.set_page_config(page_title=f"{NOM_PRODUIT} · sélection de projets d'investissement", layout="wide")
appliquer_style()
st.title(NOM_PRODUIT)
afficher_html(f'<p class="accroche">{composants.propre(ACCROCHE)}</p>')

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
    st.caption("Modules : fausses fonctions (USE_MOCKS=1)" if outils.utilise_mocks() else "Modules : vrais modules des rôles 1 à 3")

afficher_html(composants.bandeau_portefeuille(projets, len(documents), nb_charges))

with st.container(key="panneau_demande"):
    st.subheader("Votre demande")
    exemples = st.columns(len(EXEMPLES))
    for colonne, exemple in zip(exemples, EXEMPLES):
        colonne.button(exemple, on_click=lambda e=exemple: st.session_state.update(demande=e), width="stretch")
    demande = st.text_area("Décrivez vos priorités", key="demande", placeholder=EXEMPLES[0])

    choix_poids = st.radio("Poids des critères", ["L'agent les choisit selon la demande", "Je les fixe moi-même"],
                           horizontal=True, key="choix_poids")
    poids_fixes = None
    if choix_poids == "Je les fixe moi-même":
        colonnes = st.columns(4)
        bruts = {c: colonnes[i % 4].slider(LIBELLES[c], 0.0, 1.0, POIDS_PAR_DEFAUT[c], 0.05, key=f"poids_{c}")
                 for i, c in enumerate(CRITERES)}
        poids_fixes = normaliser_poids(bruts)  # ramenés à une somme de 1
        st.caption("Poids ramenés à une somme de 1 : " + ", ".join(f"{LIBELLES[c]} {p:.2f}" for c, p in poids_fixes.items() if p))
    elif mode == MODE_FIXE:
        st.caption("Le pipeline fixe ne lit pas la demande pour les poids : il utilise les poids par défaut.")

    lancer = st.button("Analyser", type="primary", key="analyser", disabled=not demande.strip())

if lancer:
    with st.spinner("Analyse en cours (environ une minute avec l'agent)…"):
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
    # Gardé en mémoire de session : changer d'onglet ne relance pas l'analyse.
    st.session_state.update(resultat=resultat, reference=reference, poids_analyse=poids_fixes)

if "resultat" not in st.session_state:
    afficher_html('<p class="attente">Aucune analyse pour l\'instant. Décrivez vos priorités (ou choisissez un '
                  'exemple), puis lancez l\'analyse : la recommandation, le classement, la vue 3D et le journal '
                  'des étapes s\'afficheront ici.</p>')
else:
    resultat = st.session_state["resultat"]
    reference = st.session_state["reference"]
    poids_analyse = st.session_state["poids_analyse"]
    noms = ["Recommandation", "Classement", "Vue 3D", "Journal des étapes", "Explication"] + (["Comparaison"] if reference else [])
    onglets = st.tabs(noms)
    with onglets[0]:
        afficher_recommandation(resultat)
    with onglets[1]:
        afficher_classement(resultat["classement"])
    with onglets[2]:
        afficher_vue_3d(resultat["classement"])
    with onglets[3]:
        afficher_journal(resultat["etapes"])
    with onglets[4]:
        afficher_explication(resultat, poids_analyse)
    if reference:
        with onglets[5]:
            afficher_comparaison(resultat, reference, poids_analyse)
