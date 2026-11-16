"""Agent d'aide à la décision d'investissement.

Le LLM (Ollama, en local) lit la demande et choisit les outils à appeler, dans
l'ordre qu'il juge utile. Les outils agissent sur tous les projets à la fois :
les données restent dans Python, le LLM ne reçoit qu'un résumé court de chaque
résultat. Il ne calcule rien lui-même, il décide et il rédige.

Deux points d'entrée :
- run_agent : l'agent piloté par le LLM ;
- pipeline_fixe : le modèle de référence, sans LLM, qui appelle toujours les
  mêmes outils dans le même ordre. Il sert de comparaison et de secours.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Le fichier .env est lu avant l'import de src.outils, qui regarde USE_MOCKS.
load_dotenv()

from src import outils  # noqa: E402
from src.contrat import (  # noqa: E402
    CRITERES,
    POIDS_PAR_DEFAUT,
    Extraction,
    Prediction,
    verifier_classement,
    verifier_poids,
)

journal = logging.getLogger(__name__)

MAX_TOURS = 10
MODELE_PAR_DEFAUT = "qwen2.5:3b"
FICHIER_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "systeme.txt"
CRITERES_TEXTE = ("impact_env_social", "alignement_strategique", "faisabilite")
NOTE_NEUTRE = 3  # note donnée à un projet sans document lisible
NB_RECOMMANDES = 3
DELAI_LLM_S = 600  # attente maximale d'une réponse du LLM, en secondes (OLLAMA_TIMEOUT pour changer)
TAILLE_CONTEXTE = 8192  # jetons gardés en mémoire par le LLM : prompt, outils et résumés doivent y tenir

# Description des outils au format attendu par Ollama (appel de fonctions).
OUTILS_LLM = [
    {"type": "function", "function": {
        "name": "extraire_criteres",
        "description": "Lit le document de chaque projet et donne trois notes de 1 à 5 : "
                       "impact environnemental et social, alignement stratégique, faisabilité.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "predire",
        "description": "Prédit pour chaque projet la rentabilité (en %) et la probabilité de risque élevé.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "detecter_anomalies",
        "description": "Repère les dossiers aux chiffres suspects (coût, durée incohérents).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "classer",
        "description": "Classe les projets par analyse multicritère. Les poids traduisent les priorités "
                       "de la demande ; ils sont ramenés à une somme de 1. Critères absents : poids 0. "
                       "Sans poids, les poids par défaut sont utilisés.",
        "parameters": {"type": "object", "properties": {"poids": {
            "type": "object",
            "description": "Poids de chaque critère, entre 0 et 1.",
            "properties": {c: {"type": "number"} for c in CRITERES},
        }}},
    }},
    {"type": "function", "function": {
        "name": "chercher_passages",
        "description": "Cherche dans les documents les passages qui répondent à une question, "
                       "pour justifier la recommandation par une citation.",
        "parameters": {"type": "object", "required": ["question"], "properties": {
            "question": {"type": "string"},
            "k": {"type": "integer", "description": "Nombre de passages (3 par défaut)."},
        }},
    }},
]
NOMS_OUTILS = [o["function"]["name"] for o in OUTILS_LLM]


def normaliser_poids(poids: dict | str | None) -> dict[str, float]:
    """Nettoie des poids (du LLM ou de l'utilisateur) et les ramène à une somme de 1.

    Les critères inconnus et les valeurs illisibles ou négatives sont ignorés.
    S'il ne reste rien, on prend les poids par défaut du contrat.
    """
    if isinstance(poids, str):  # un petit modèle envoie parfois le dictionnaire sous forme de texte
        try:
            poids = json.loads(poids)
        except ValueError:
            poids = None
    if not isinstance(poids, dict):
        poids = {}
    propres = {}
    for critere, valeur in poids.items():
        try:
            if critere in CRITERES and float(valeur) > 0:
                propres[critere] = float(valeur)
        except (TypeError, ValueError):
            pass  # valeur illisible : critère ignoré
    total = sum(propres.values())
    if total == 0:
        return dict(POIDS_PAR_DEFAUT)
    resultat = {c: propres.get(c, 0.0) / total for c in CRITERES}
    verifier_poids(resultat)
    return resultat


def decrire(ligne: dict) -> str:
    """Une ligne du classement en texte, avec les chiffres rendus par les outils."""
    return (f"rang {ligne['rang']}, {ligne['id_projet']} {ligne.get('nom', '')} : score {ligne['score']:.3f}, "
            f"rentabilité prédite {ligne['rentabilite_predite']} %, "
            f"probabilité de risque élevé {ligne['proba_risque']:.2f}, "
            f"impact {ligne['impact_env_social']}/5, alignement {ligne['alignement_strategique']}/5, "
            f"faisabilité {ligne['faisabilite']}/5")


class Portefeuille:
    """Garde les projets et les résultats des outils pendant une analyse."""

    def __init__(self, projets: pd.DataFrame, documents: dict | None = None,
                 poids_imposes: dict | None = None) -> None:
        self.projets = projets.reset_index(drop=True)
        self.documents = documents or {}
        # Poids choisis par l'utilisateur (curseurs de l'application) : ils passent avant ceux du LLM.
        self.poids_imposes = normaliser_poids(poids_imposes) if poids_imposes else None
        self.predictions: pd.DataFrame | None = None
        self.notes: pd.DataFrame | None = None
        self.anomalies: list[str] | None = None
        self.classement: pd.DataFrame | None = None
        self.passages: list[dict] = []
        self.etapes: list[dict] = []

    # --- Les cinq outils : chacun rend un résumé court pour le LLM ---------

    def extraire_criteres(self) -> str:
        """Appelle extract sur le document de chaque projet. Sans document : note neutre."""
        lignes, sans_note = [], 0
        for id_projet in self.projets["id_projet"]:
            notes = {c: NOTE_NEUTRE for c in CRITERES_TEXTE}
            texte = self.documents.get(id_projet)
            try:
                if not texte:
                    raise ValueError("pas de document")
                sortie = Extraction(**outils.extract(texte))  # vérifie le format du contrat
                notes = {c: getattr(sortie, c) for c in CRITERES_TEXTE}
            except Exception as erreur:  # document absent ou sortie invalide : on garde la note neutre
                sans_note += 1
                journal.debug("Pas de notes pour %s : %s", id_projet, erreur)
            lignes.append({"id_projet": id_projet, **notes})
        self.notes = pd.DataFrame(lignes)
        moyennes = ", ".join(f"{c} {self.notes[c].mean():.1f}" for c in CRITERES_TEXTE)
        return (f"Notes obtenues pour {len(lignes) - sans_note} projets "
                f"({sans_note} sans document lisible, note neutre {NOTE_NEUTRE}). Moyennes sur 5 : {moyennes}.")

    def predire(self) -> str:
        """Appelle predict sur chaque projet, sans lui montrer les cibles."""
        lignes = []
        for projet in self.projets.to_dict("records"):
            fiche = {k: v for k, v in projet.items() if k not in ("rentabilite", "risque")}
            sortie = Prediction(**outils.predict(fiche))  # vérifie le format du contrat
            lignes.append({"id_projet": projet["id_projet"],
                           "rentabilite_predite": sortie.rentabilite_predite,
                           "proba_risque": sortie.proba_risque,
                           "risque_predit": sortie.risque})
        self.predictions = pd.DataFrame(lignes)
        repartition = self.predictions["risque_predit"].value_counts().to_dict()
        return (f"Prédictions pour {len(lignes)} projets. Rentabilité prédite moyenne "
                f"{self.predictions['rentabilite_predite'].mean():.1f} %. Risque prédit : {repartition}.")

    def detecter_anomalies(self) -> str:
        """Appelle detect_anomalies et retient les identifiants des dossiers suspects."""
        sortie = outils.detect_anomalies(self.projets)
        self.anomalies = sortie.loc[sortie["anomalie"].astype(bool), "id_projet"].tolist()
        if self.classement is not None:
            self.classement["anomalie"] = self.classement["id_projet"].isin(self.anomalies)
        if not self.anomalies:
            return "Aucun dossier anormal détecté."
        suite = " (dix premiers)" if len(self.anomalies) > 10 else ""
        return (f"{len(self.anomalies)} dossiers anormaux, à écarter de la recommandation{suite} : "
                f"{', '.join(self.anomalies[:10])}.")

    def classer(self, poids: dict | str | None = None) -> str:
        """Construit la matrice des sept critères et appelle rank."""
        # Les étapes manquantes sont lancées d'office : un petit LLM oublie parfois un outil.
        prealables = []
        if self.notes is None:
            prealables.append(self.extraire_criteres())
        if self.predictions is None:
            prealables.append(self.predire())
        if self.anomalies is None:
            prealables.append(self.detecter_anomalies())
        poids = self.poids_imposes or normaliser_poids(poids)

        matrice = self.projets[["id_projet", "cout_total", "duree_mois"]]
        matrice = matrice.merge(self.predictions[["id_projet", "rentabilite_predite", "proba_risque"]], on="id_projet")
        matrice = matrice.merge(self.notes, on="id_projet")
        classement = outils.rank(matrice, poids)
        verifier_classement(classement)

        # Colonnes ajoutées pour l'affichage : nom, secteur, pays, risque prédit, anomalie.
        infos = self.projets[[c for c in ("id_projet", "nom", "secteur", "pays") if c in self.projets]]
        classement = classement.merge(infos, on="id_projet", how="left")
        classement = classement.merge(self.predictions[["id_projet", "risque_predit"]], on="id_projet", how="left")
        classement["anomalie"] = classement["id_projet"].isin(self.anomalies)
        self.classement = classement

        poids_texte = ", ".join(f"{c} {p:.2f}" for c, p in poids.items() if p > 0)
        retenus = " | ".join(decrire(r) for r in self.recommandes().to_dict("records"))
        resume = prealables + [f"Poids utilisés : {poids_texte}.",
                               f"Projets à recommander (dossiers anormaux déjà écartés) : {retenus}."]
        if self.anomalies:
            resume.append(f"Dossiers anormaux à signaler en alerte : {', '.join(self.anomalies[:10])}.")
        return " ".join(resume)

    def chercher_passages(self, question: str, k: int = 3) -> str:
        """Appelle retrieve et rend les passages trouvés, raccourcis."""
        k = min(max(int(k), 1), 10)
        # On ne garde que les passages des projets analysés : pas de citation d'un autre dossier.
        connus = set(self.projets["id_projet"])
        self.passages = [p for p in outils.retrieve(str(question), k) if p["id_projet"] in connus]
        if not self.passages:
            return "Aucun passage trouvé dans les documents des projets analysés."
        return " | ".join(f"{p['id_projet']} : « {p['passage'][:200]} »" for p in self.passages)

    # --- Aides --------------------------------------------------------------

    def recommandes(self) -> pd.DataFrame:
        """Les trois premiers projets du classement, dossiers anormaux écartés."""
        if self.classement is None:
            return pd.DataFrame()
        return self.classement[~self.classement["anomalie"]].head(NB_RECOMMANDES)

    def appeler(self, nom: str, arguments: dict) -> str:
        """Exécute un outil, l'ajoute au journal et rend le résumé (ou l'erreur) en texte."""
        debut = time.perf_counter()
        if nom not in NOMS_OUTILS:
            resultat = f"Erreur : outil inconnu « {nom} ». Outils disponibles : {', '.join(NOMS_OUTILS)}."
        else:
            outil = getattr(self, nom)
            # Un petit LLM invente parfois des arguments : on ne garde que ceux que l'outil accepte.
            acceptes = {k: v for k, v in arguments.items() if k in inspect.signature(outil).parameters}
            try:
                resultat = outil(**acceptes)
            except Exception as erreur:  # une erreur d'outil ne doit pas arrêter l'agent
                resultat = f"Erreur dans {nom} : {erreur}"
        duree = round(time.perf_counter() - debut, 3)
        self.etapes.append({"outil": nom, "arguments": arguments, "resultat": resultat, "duree_s": duree})
        journal.info("Outil %s (%.2f s) : %s", nom, duree, resultat[:150])
        return resultat


def rediger_recommandation(portefeuille: Portefeuille) -> str:
    """Recommandation écrite à partir d'un modèle de texte, sans LLM."""
    if portefeuille.classement is None:
        return "Aucun classement n'a pu être produit : voir les erreurs dans le journal des étapes."
    lignes = ["Projets recommandés :"]
    lignes += [f"- {decrire(r)}." for r in portefeuille.recommandes().to_dict("records")]
    anomalies = portefeuille.anomalies or []
    if anomalies:
        lignes.append(f"Alerte : {len(anomalies)} dossiers anormaux écartés de la recommandation "
                      f"({', '.join(anomalies[:10])}). À vérifier avant toute décision.")
    if portefeuille.passages:
        p = portefeuille.passages[0]
        lignes.append(f"Passage à l'appui ({p['id_projet']}) : « {p['passage'][:300]} »")
    lignes.append("Limites : les prédictions viennent d'un modèle entraîné sur des cibles simulées ; "
                  "cette recommandation aide à la décision, elle ne la remplace pas.")
    return "\n".join(lignes)


def controler_recommandation(portefeuille: Portefeuille, texte: str) -> str:
    """Vérifie le texte du LLM et ajoute ce qu'il a oublié ou n'a pas pu vérifier."""
    oublis = [decrire(r) for r in portefeuille.recommandes().to_dict("records") if r["id_projet"] not in texte]
    anomalies = portefeuille.anomalies or []
    alerte_oubliee = bool(anomalies) and not any(a in texte for a in anomalies)
    # Vu avec qwen2.5:3b : sans avoir cherché de passage, il en invente un et le cite.
    citation_non_verifiee = not portefeuille.passages and "passage cité" in texte.lower() \
        and "passage cité : aucun" not in texte.lower()
    if not oublis and not alerte_oubliee and not citation_non_verifiee:
        return texte
    ajouts = ["", "", "Contrôle automatique :"]
    ajouts += [f"- Projet à recommander oublié dans le texte : {o}." for o in oublis]
    if alerte_oubliee:
        ajouts.append(f"- Alerte oubliée : {len(anomalies)} dossiers anormaux écartés de la recommandation "
                      f"({', '.join(anomalies[:10])}{' …' if len(anomalies) > 10 else ''}).")
    if citation_non_verifiee:
        ajouts.append("- Le passage cité n'a pas été rendu par l'outil chercher_passages : "
                      "il n'est pas vérifié et ne doit pas être pris en compte.")
    return texte + "\n".join(ajouts)


def _sortie(portefeuille: Portefeuille, recommandation: str) -> dict:
    """Met la réponse au format du contrat."""
    classement = portefeuille.classement
    if classement is None:  # même en cas d'erreur, on rend un tableau aux bonnes colonnes
        classement = pd.DataFrame(columns=["id_projet", "score", "rang"])
    return {"classement": classement, "recommandation": recommandation, "etapes": portefeuille.etapes}


def _secours(portefeuille: Portefeuille, raison: str) -> dict:
    """Le LLM n'a pas abouti : on termine sans lui, en le disant clairement."""
    if portefeuille.classement is None:
        portefeuille.appeler("classer", {})
    texte = f"L'agent n'a pas terminé ({raison}). Recommandation produite sans LLM :\n"
    return _sortie(portefeuille, texte + rediger_recommandation(portefeuille))


def pipeline_fixe(demande: str, projets: pd.DataFrame, documents: dict | None = None,
                  poids: dict | None = None) -> dict:
    """Modèle de référence : les cinq outils, toujours dans le même ordre, sans LLM."""
    portefeuille = Portefeuille(projets, documents, poids)
    portefeuille.appeler("extraire_criteres", {})
    portefeuille.appeler("predire", {})
    portefeuille.appeler("detecter_anomalies", {})
    portefeuille.appeler("classer", {})
    portefeuille.appeler("chercher_passages", {"question": demande})
    return _sortie(portefeuille, rediger_recommandation(portefeuille))


def run_agent(demande: str, projets: pd.DataFrame, documents: dict | None = None,
              poids: dict | None = None, client: object | None = None, modele: str | None = None) -> dict:
    """Agent piloté par le LLM. `client` et `modele` servent aux tests (faux LLM)."""
    if client is None:
        import ollama  # importé ici pour que les tests et pipeline_fixe marchent sans Ollama

        delai = float(os.getenv("OLLAMA_TIMEOUT") or DELAI_LLM_S)
        client = ollama.Client(host=os.getenv("OLLAMA_HOST") or None, timeout=delai)
    modele = modele or os.getenv("OLLAMA_MODEL") or MODELE_PAR_DEFAUT
    portefeuille = Portefeuille(projets, documents, poids)

    contexte = f"Demande : {demande}\n\nPortefeuille : {len(projets)} projets, {len(portefeuille.documents)} documents."
    if "secteur" in projets:
        contexte += f" Secteurs les plus fréquents : {projets['secteur'].value_counts().head(8).to_dict()}."
    if portefeuille.poids_imposes:
        contexte += ("\nL'utilisateur a fixé les poids lui-même : appelle classer sans poids, "
                     "les siens seront utilisés.")
    messages = [{"role": "system", "content": FICHIER_PROMPT.read_text(encoding="utf-8")},
                {"role": "user", "content": contexte}]

    for tour in range(1, MAX_TOURS + 1):
        debut = time.perf_counter()
        try:
            reponse = client.chat(model=modele, messages=messages, tools=OUTILS_LLM,
                                  options={"temperature": 0, "num_ctx": TAILLE_CONTEXTE}, think=False)
        except Exception as erreur:  # Ollama arrêté, modèle absent, délai dépassé, etc.
            portefeuille.etapes.append({"outil": "llm", "arguments": {}, "resultat": f"Erreur du LLM : {erreur}",
                                        "duree_s": round(time.perf_counter() - debut, 3)})
            return _secours(portefeuille, "erreur du LLM")
        message = reponse.message
        appels = message.tool_calls or []
        journal.info("Tour %d : réponse du LLM en %.1f s, %d appel(s) d'outil",
                     tour, time.perf_counter() - debut, len(appels))

        if not appels:  # plus d'outil demandé : le texte est la recommandation finale
            texte = re.sub(r"<think>.*?</think>", "", message.content or "", flags=re.S).strip()
            if not texte:
                return _secours(portefeuille, "réponse vide du LLM")
            if portefeuille.classement is None:
                return _secours(portefeuille, "le LLM a répondu sans classer les projets")
            return _sortie(portefeuille, controler_recommandation(portefeuille, texte))

        # On remet dans la conversation la demande d'outils du LLM, puis le résultat de chaque outil.
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
            {"function": {"name": a.function.name, "arguments": dict(a.function.arguments or {})}} for a in appels]})
        for appel in appels:
            resultat = portefeuille.appeler(appel.function.name, dict(appel.function.arguments or {}))
            messages.append({"role": "tool", "tool_name": appel.function.name, "content": resultat})

    return _secours(portefeuille, f"plafond de {MAX_TOURS} tours atteint")
