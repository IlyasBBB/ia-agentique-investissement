"""Tests de l'agent (src/agent.py).

Le LLM est remplacé par un faux client qui rejoue un scénario écrit à l'avance :
les tests n'ont donc pas besoin d'Ollama et donnent toujours le même résultat.
Seuls les tests marqués llm appellent le vrai modèle. Ils sont lents, donc écartés
par défaut (pytest.ini) : les lancer avec `python -m pytest -m llm`.

En fin de fichier : les cinq scénarios de la tâche 5 (définis dans src/scenarios.py)
et les tests des vérifications automatiques utilisées pour les évaluer.
"""
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from src import agent, outils
from src.agent import MAX_TOURS, Portefeuille, normaliser_poids, pipeline_fixe, run_agent
from src.contrat import CRITERES, POIDS_PAR_DEFAUT, Etape, Passage, verifier_classement, verifier_poids
from src.mocks import DOCUMENTS_EXEMPLE, PROJETS_EXEMPLE
from src.scenarios import (SCENARIOS, charger_projets_scenarios, chiffres_non_retrouves, citation_verifiee,
                           dossiers_anormaux, poids_coherents, trois_premiers, verifier_sortie)

DEMANDE = "Je veux des projets peu risqués avec un fort impact environnemental"
ORDRE_FIXE = ["extraire_criteres", "predire", "detecter_anomalies", "classer", "chercher_passages"]
ANOMALIE = "P005"  # coût énorme pour 6 mois : signalé par la fausse détection


def appel(nom: str, **arguments) -> SimpleNamespace:
    """Une demande d'outil, comme la renvoie Ollama."""
    return SimpleNamespace(function=SimpleNamespace(name=nom, arguments=arguments))


def reponse(texte: str = "", appels: list | None = None) -> SimpleNamespace:
    """Une réponse du LLM : du texte, ou des demandes d'outils."""
    return SimpleNamespace(message=SimpleNamespace(content=texte, tool_calls=appels))


class FauxLLM:
    """Faux client Ollama : rend les réponses du scénario une par une."""

    def __init__(self, scenario: list) -> None:
        self.scenario = list(scenario)
        self.nb_appels = 0

    def chat(self, **parametres) -> SimpleNamespace:
        self.nb_appels += 1
        suivante = self.scenario.pop(0) if len(self.scenario) > 1 else self.scenario[0]
        if isinstance(suivante, Exception):
            raise suivante
        return suivante


def lancer(scenario: list, poids: dict | None = None) -> dict:
    """Lance l'agent sur les projets d'exemple avec un faux LLM."""
    return run_agent(DEMANDE, PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, poids, client=FauxLLM(scenario))


def outils_appeles(sortie: dict) -> list[str]:
    return [etape["outil"] for etape in sortie["etapes"]]


def poids_utilises(monkeypatch) -> dict:
    """Espionne rank pour voir les poids que l'agent lui passe."""
    vus = {}
    vrai_rank = outils.rank

    def faux_rank(matrice, poids):
        vus.update(poids)
        return vrai_rank(matrice, poids)

    monkeypatch.setattr(agent.outils, "rank", faux_rank)
    return vus


SCENARIO_NORMAL = [
    reponse(appels=[appel("extraire_criteres")]),
    reponse(appels=[appel("predire")]),
    reponse(appels=[appel("detecter_anomalies")]),
    # Comme le vrai LLM, classer et chercher_passages demandés dans le même tour.
    reponse(appels=[appel("classer", poids={"proba_risque": 3, "impact_env_social": 1}),
                    appel("chercher_passages", question="impact environnemental", k=2)]),
    reponse("Je recommande ces trois projets."),
]


def test_scenario_normal():
    sortie = lancer(SCENARIO_NORMAL)
    assert set(sortie) == {"classement", "recommandation", "etapes"}
    assert outils_appeles(sortie) == ORDRE_FIXE
    for etape in sortie["etapes"]:
        Etape(**etape)
    verifier_classement(sortie["classement"])
    assert len(sortie["classement"]) == len(PROJETS_EXEMPLE)
    assert sortie["recommandation"].startswith("Je recommande ces trois projets.")


def test_poids_du_llm_normalises(monkeypatch):
    vus = poids_utilises(monkeypatch)
    lancer(SCENARIO_NORMAL)
    verifier_poids(vus)
    assert vus["proba_risque"] == pytest.approx(0.75)
    assert vus["impact_env_social"] == pytest.approx(0.25)
    assert vus["cout_total"] == 0


def test_normaliser_poids_cas_limites():
    assert normaliser_poids(None) == POIDS_PAR_DEFAUT
    assert normaliser_poids({"critere_inconnu": 1, "faisabilite": "beaucoup"}) == POIDS_PAR_DEFAUT
    assert normaliser_poids('{"faisabilite": 2}')["faisabilite"] == 1
    assert set(normaliser_poids({"cout_total": 5})) == set(CRITERES)


def test_poids_de_l_utilisateur_prioritaires(monkeypatch):
    vus = poids_utilises(monkeypatch)
    lancer(SCENARIO_NORMAL, poids={"faisabilite": 1.0})
    assert vus["faisabilite"] == 1
    assert vus["proba_risque"] == 0


def test_outil_inconnu_journalise_sans_plantage():
    sortie = lancer([reponse(appels=[appel("calculer_van")]), *SCENARIO_NORMAL])
    assert sortie["etapes"][0]["outil"] == "calculer_van"
    assert "outil inconnu" in sortie["etapes"][0]["resultat"]
    assert sortie["recommandation"].startswith("Je recommande ces trois projets.")


def test_erreur_d_outil_journalisee_sans_plantage(monkeypatch):
    def predict_en_panne(projet):
        raise RuntimeError("modèle introuvable")

    monkeypatch.setattr(agent.outils, "predict", predict_en_panne)
    sortie = lancer(SCENARIO_NORMAL)
    etape = sortie["etapes"][1]
    assert etape["outil"] == "predire"
    assert "Erreur dans predire : modèle introuvable" in etape["resultat"]
    # Sans prédiction, pas de classement : la sortie garde quand même le format du contrat.
    assert list(sortie["classement"].columns) == ["id_projet", "score", "rang"]
    assert "sans LLM" in sortie["recommandation"]


def test_arguments_inventes_ignores():
    # Vu avec qwen2.5:3b : il ajoute un argument « documents » aux outils qui n'en ont pas.
    sortie = lancer([reponse(appels=[appel("predire", documents=["P001"])]), *SCENARIO_NORMAL])
    assert sortie["etapes"][0]["resultat"].startswith("Prédictions")
    assert sortie["etapes"][0]["arguments"] == {"documents": ["P001"]}  # le journal garde ce que le LLM a envoyé
    verifier_classement(sortie["classement"])


def test_plafond_de_tours_donne_la_recommandation_de_secours():
    faux = FauxLLM([reponse(appels=[appel("predire")])])  # le LLM ne s'arrête jamais
    sortie = run_agent(DEMANDE, PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, client=faux)
    assert faux.nb_appels == MAX_TOURS
    assert "plafond" in sortie["recommandation"]
    assert "Projets recommandés" in sortie["recommandation"]
    verifier_classement(sortie["classement"])


def test_llm_en_panne_donne_la_recommandation_de_secours():
    sortie = lancer([ConnectionError("Ollama ne répond pas")])
    assert sortie["etapes"][0]["outil"] == "llm"
    assert "Ollama ne répond pas" in sortie["etapes"][0]["resultat"]
    assert "erreur du LLM" in sortie["recommandation"]
    verifier_classement(sortie["classement"])
    for etape in sortie["etapes"]:
        Etape(**etape)


def test_reponse_sans_outil_donne_la_recommandation_de_secours():
    sortie = lancer([reponse("Je recommande P005, rentabilité 42 %.")])
    assert "sans classer" in sortie["recommandation"]
    assert "42 %" not in sortie["recommandation"]


def test_pipeline_fixe_conforme_et_toujours_dans_le_meme_ordre():
    premiere = pipeline_fixe(DEMANDE, PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE)
    seconde = pipeline_fixe("Autre demande", PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE)
    assert outils_appeles(premiere) == outils_appeles(seconde) == ORDRE_FIXE
    for etape in premiere["etapes"]:
        Etape(**etape)
    verifier_classement(premiere["classement"])
    assert isinstance(premiere["recommandation"], str)
    assert "cibles simulées" in premiere["recommandation"]
    assert premiere["classement"].equals(seconde["classement"])


def test_anomalies_exclues_du_top_3():
    # Tout le poids sur la durée : le dossier anormal (6 mois) arrive premier du classement.
    portefeuille = Portefeuille(PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, {"duree_mois": 1.0})
    resume = portefeuille.appeler("classer", {})
    assert portefeuille.classement.iloc[0]["id_projet"] == ANOMALIE
    assert bool(portefeuille.classement.iloc[0]["anomalie"])
    retenus = portefeuille.recommandes()
    assert len(retenus) == 3
    assert ANOMALIE not in list(retenus["id_projet"])
    a_recommander = resume.split("Projets à recommander")[1].split("Dossiers anormaux à signaler")[0]
    assert ANOMALIE not in a_recommander

    sortie = pipeline_fixe(DEMANDE, PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, {"duree_mois": 1.0})
    recommandes, alerte = sortie["recommandation"].split("Alerte")
    assert ANOMALIE not in recommandes
    assert ANOMALIE in alerte


def test_projet_sans_document_recoit_la_note_neutre():
    sortie = pipeline_fixe(DEMANDE, PROJETS_EXEMPLE, documents=None)
    assert "5 sans document" in sortie["etapes"][0]["resultat"]
    assert (sortie["classement"]["faisabilite"] == agent.NOTE_NEUTRE).all()


@pytest.mark.llm
def test_agent_avec_le_vrai_llm():
    """Lance le vrai agent. Sauté si Ollama ou le modèle n'est pas disponible."""
    ollama = pytest.importorskip("ollama")
    modele = os.getenv("OLLAMA_MODEL") or agent.MODELE_PAR_DEFAUT
    try:
        disponibles = [m.model for m in ollama.Client(host=os.getenv("OLLAMA_HOST") or None).list().models]
    except Exception:
        pytest.skip("Serveur Ollama injoignable : lancer `ollama serve`")
    if modele not in disponibles:
        pytest.skip(f"Modèle {modele} absent : lancer `ollama pull {modele}`")

    sortie = run_agent(DEMANDE, PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE)
    verifier_classement(sortie["classement"])
    assert "classer" in outils_appeles(sortie)
    assert len(sortie["recommandation"]) > 50
    for etape in sortie["etapes"]:
        Etape(**etape)


def test_dossier_anormal_ne_prend_pas_de_numero(monkeypatch):
    # Vu dans l'application : le n° 2 du classement est anormal, les cartes affichaient « N° 1, 3, 4 ».
    portefeuille = Portefeuille(PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, None)
    portefeuille.appeler("classer", {})
    deuxieme = portefeuille.classement.iloc[1]["id_projet"]

    def fausse_detection(df):
        return df.assign(anomalie=df["id_projet"] == deuxieme)

    monkeypatch.setattr(agent.outils, "detect_anomalies", fausse_detection)
    portefeuille = Portefeuille(PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, None)
    resume = portefeuille.classer()
    classement = portefeuille.classement
    assert classement.loc[classement["id_projet"] == deuxieme, "rang_retenu"].isna().all()
    assert list(portefeuille.recommandes()["rang_retenu"]) == [1, 2, 3]
    assert "rang 4" not in resume and "rang 3" in resume  # texte envoyé au LLM
    assert sorted(classement["rang"]) == list(range(1, len(classement) + 1))  # rang de rank inchangé
    texte = agent.rediger_recommandation(portefeuille)
    assert "rang 1" in texte and "rang 2" in texte and "rang 3" in texte and "rang 4" not in texte


def test_numeroter_sans_colonne_anomalie():
    classement = pd.DataFrame({"id_projet": ["A", "B"], "score": [0.9, 0.1], "rang": [1, 2]})
    assert list(agent.numeroter(classement)["rang_retenu"]) == [1, 2]


def test_chercher_passages_garde_k_passages_des_projets_analyses(monkeypatch):
    # retrieve cherche dans tout le corpus : les premiers passages peuvent venir d'autres projets.
    connu = PROJETS_EXEMPLE["id_projet"].iloc[0]
    corpus = [{"id_projet": f"X{i}", "passage": "autre", "score": 1.0} for i in range(10)]
    corpus += [{"id_projet": connu, "passage": f"passage {i}", "score": 0.5} for i in range(5)]
    demandes = []

    def faux_retrieve(question, k=3):
        demandes.append(k)
        return corpus[:k]

    monkeypatch.setattr(agent.outils, "retrieve", faux_retrieve)
    portefeuille = Portefeuille(PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, None)
    portefeuille.chercher_passages("impact", k=2)
    assert demandes == [agent.NB_PASSAGES_CHERCHES]
    assert [p["passage"] for p in portefeuille.passages] == ["passage 0", "passage 1"]


def test_controle_ajoute_ce_que_le_llm_a_oublie():
    # Vu avec qwen2.5:3b : deux projets sur trois cités et « aucun dossier anormal » alors qu'il y en a un.
    portefeuille = Portefeuille(PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, {"duree_mois": 1.0})
    portefeuille.appeler("classer", {})
    premier, deuxieme, troisieme = portefeuille.recommandes()["id_projet"]
    texte = agent.controler_recommandation(portefeuille, f"Je recommande {premier} et {deuxieme}.")
    controle = texte.split("Contrôle automatique")[1]
    assert troisieme in controle and ANOMALIE in controle
    complet = f"{premier}, {deuxieme}, {troisieme}. Alerte : {ANOMALIE}."
    assert agent.controler_recommandation(portefeuille, complet) == complet


def test_controle_signale_une_citation_inventee():
    # Vu avec qwen2.5:3b sur 500 projets : chercher_passages jamais appelé, mais un « passage cité » inventé.
    portefeuille = Portefeuille(PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, None)
    portefeuille.appeler("classer", {})
    ids = ", ".join(portefeuille.recommandes()["id_projet"]) + ", " + ", ".join(portefeuille.anomalies)
    invente = agent.controler_recommandation(portefeuille, f"{ids}. Passage cité : « texte inventé »")
    assert "n'est pas vérifié" in invente
    honnete = f"{ids}. Passage cité : aucun."
    assert agent.controler_recommandation(portefeuille, honnete) == honnete


def test_passages_cherches_d_office_apres_classer():
    # Vu avec qwen2.5:3b sur 500 projets : il s'arrête après classer, sans chercher de passage.
    sortie = lancer([reponse(appels=[appel("classer", poids={"proba_risque": 1})]),
                     reponse("Je recommande ces trois projets.")])
    auto = [e for e in sortie["etapes"] if e["outil"] == "chercher_passages"]
    assert len(auto) == 1 and auto[0]["arguments"]["lance_par"] == "agent"
    premier = sortie["classement"].query("not anomalie").iloc[0]["nom"]
    assert premier in auto[0]["arguments"]["question"]  # question tirée des projets recommandés


def test_agent_transmet_ses_documents_a_retrieve(monkeypatch):
    # Contrat du 18 novembre : retrieve(question, k, documents) ne cherche que dans ces documents.
    recus = []

    def faux_retrieve(question, k=3, documents=None):
        recus.append(documents)
        return [{"id_projet": i, "passage": texte, "score": 1.0} for i, texte in (documents or {}).items()][:k]

    monkeypatch.setattr(agent.outils, "retrieve", faux_retrieve)
    importes = {"P001": "Un document importé qui parle d'énergie solaire."}
    portefeuille = Portefeuille(PROJETS_EXEMPLE, importes, None)
    portefeuille.chercher_passages("solar energy", k=1)
    assert recus == [importes]
    assert portefeuille.passages[0]["passage"] == importes["P001"]


def test_fausse_retrieve_cherche_dans_les_documents_recus():
    from src import mocks

    documents = {"X1": "wind farm in the north", "X2": "school meals"}
    resultats = mocks.retrieve("wind farm", k=2, documents=documents)
    assert {r["id_projet"] for r in resultats} == {"X1", "X2"}
    assert resultats[0]["id_projet"] == "X1"
    for r in resultats:
        Passage(**r)


def test_passages_pas_cherches_deux_fois():
    sortie = lancer(SCENARIO_NORMAL)
    assert [e["outil"] for e in sortie["etapes"]].count("chercher_passages") == 1


@pytest.fixture
def fausse_recherche(monkeypatch):
    """Fausse retrieve, même avec MODULES_REELS=nlp : ces tests vérifient la logique de l'agent, pas le moteur."""
    from src import mocks

    monkeypatch.setattr(agent.outils, "retrieve", mocks.retrieve)


def test_citation_inventee_signalee_meme_avec_des_passages(fausse_recherche):
    portefeuille = Portefeuille(PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, None)
    portefeuille.appeler("classer", {})
    portefeuille.appeler("chercher_passages", {"question": "réduction des émissions"})
    vrai = portefeuille.passages[0]
    ids = ", ".join(portefeuille.recommandes()["id_projet"]) + ", " + ", ".join(portefeuille.anomalies)
    invente = agent.controler_recommandation(portefeuille, f"{ids}. Passage cité : « texte inventé »")
    assert "n'est pas vérifié" in invente and "Passage vérifié" in invente
    fidele = f"{ids}. Passage cité : {vrai['id_projet']} « {vrai['passage']} »"
    assert agent.controler_recommandation(portefeuille, fidele) == fidele


def test_reponse_vide_relancee_une_fois():
    # Vu avec qwen2.5:3b (scénario 1, passage 2) : réponse sans texte ni outil dès le premier tour.
    sortie = lancer([reponse(""), *SCENARIO_NORMAL])
    assert outils_appeles(sortie) == ["llm", *ORDRE_FIXE]
    assert "relance" in sortie["etapes"][0]["resultat"]
    assert sortie["recommandation"].startswith("Je recommande ces trois projets.")
    for etape in sortie["etapes"]:
        Etape(**etape)

    toujours_vide = FauxLLM([reponse("")])
    sortie = run_agent(DEMANDE, PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE, client=toujours_vide)
    assert toujours_vide.nb_appels == 2  # une seule relance, puis le secours
    assert "réponse vide du LLM" in sortie["recommandation"]
    verifier_classement(sortie["classement"])


def test_reponse_sans_outil_gardee_dans_le_journal():
    # Vu avec qwen2.5:3b (scénario 5, demande hors sujet) : le LLM explique que ses outils ne
    # répondent pas à la demande. Ce texte était perdu ; le journal le garde maintenant.
    refus = "Cette demande n'est pas gérée par les outils disponibles."
    sortie = lancer([reponse(refus)])
    assert outils_appeles(sortie) == ["llm", "classer"]
    assert refus in sortie["etapes"][0]["resultat"]
    assert refus not in sortie["recommandation"]  # la recommandation ne contient que du vérifié
    assert "sans classer" in sortie["recommandation"]
    for etape in sortie["etapes"]:
        Etape(**etape)


# --- Tâche 5 : les cinq scénarios ---------------------------------------------------


def exiger_ollama() -> None:
    """Saute le test si Ollama ou le modèle n'est pas disponible."""
    ollama = pytest.importorskip("ollama")
    modele = os.getenv("OLLAMA_MODEL") or agent.MODELE_PAR_DEFAUT
    try:
        disponibles = [m.model for m in ollama.Client(host=os.getenv("OLLAMA_HOST") or None).list().models]
    except Exception:
        pytest.skip("Serveur Ollama injoignable : lancer `ollama serve`")
    if modele not in disponibles:
        pytest.skip(f"Modèle {modele} absent : lancer `ollama pull {modele}`")


def verifier_ce_qui_doit_tenir(sortie: dict, scenario: dict) -> dict:
    """Assertions communes aux scénarios : ce que le code garantit, quel que soit le texte du LLM.

    Les comportements qui dépendent du modèle (poids choisis, ordre des outils, qualité
    du texte) ne sont pas affirmés ici : ils sont mesurés par scripts/evaluer_agent.py
    et commentés dans reports/role4/tests_agent.md.
    """
    resultat = verifier_sortie(sortie, scenario)
    assert resultat["contrat_ok"]
    assert "classer" in outils_appeles(sortie)  # demandé par le LLM, ou lancé par le secours
    assert len(trois_premiers(sortie["classement"])) == 3
    assert resultat["top3_dans_sortie"]  # cités par le LLM, sinon ajoutés par le contrôle automatique
    assert resultat["anomalies_dans_sortie"]
    assert not set(dossiers_anormaux(sortie["classement"])) & set(trois_premiers(sortie["classement"]))
    if scenario["poids_utilisateur"]:
        assert resultat["poids_ok"]  # les poids de l'utilisateur passent toujours avant ceux du LLM
    return resultat


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[f"scenario_{s['numero']}" for s in SCENARIOS])
def test_scenario_avec_le_pipeline_fixe(scenario):
    """Les mêmes vérifications sur le modèle de référence : rapide, sans LLM."""
    projets, documents = charger_projets_scenarios()
    assert len(projets) == 30 and (projets["statut"] == "en_cours").all()
    sortie = pipeline_fixe(scenario["demande"], projets, documents, scenario["poids_utilisateur"])
    resultat = verifier_ce_qui_doit_tenir(sortie, scenario)
    assert resultat["ordre_ok"] and resultat["citation_ok"]
    assert resultat["anomalies"] == "P121507"  # le dossier signalé par la fausse détection
    assert resultat["chiffres_non_retrouves"] == "" and resultat["identifiants_inconnus"] == ""


@pytest.mark.llm
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[f"scenario_{s['numero']}" for s in SCENARIOS])
def test_scenario_avec_le_vrai_llm(scenario):
    """Un scénario de bout en bout avec le vrai modèle, sur les 30 projets de test."""
    exiger_ollama()
    projets, documents = charger_projets_scenarios()
    sortie = run_agent(scenario["demande"], projets, documents, scenario["poids_utilisateur"])
    verifier_ce_qui_doit_tenir(sortie, scenario)


def test_poids_coherents_avec_la_demande():
    peu_risque, plus_rentable, _, poids_fixes, hors_sujet = SCENARIOS
    assert poids_coherents(normaliser_poids({"proba_risque": 0.5, "impact_env_social": 0.5}), peu_risque)
    assert not poids_coherents(dict(POIDS_PAR_DEFAUT), peu_risque)  # la rentabilité pèse plus que l'impact
    assert poids_coherents(dict(POIDS_PAR_DEFAUT), plus_rentable)
    assert poids_coherents(normaliser_poids(poids_fixes["poids_utilisateur"]), poids_fixes)
    assert poids_coherents(dict(POIDS_PAR_DEFAUT), hors_sujet)  # aucune priorité : poids par défaut attendus


def test_chiffre_invente_repere():
    sortie = pipeline_fixe(DEMANDE, PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE)
    classement = sortie["classement"]
    premier = classement.iloc[0]
    juste = f"{premier['id_projet']} : score {premier['score']:.2f}, rentabilité {premier['rentabilite_predite']} %"
    assert chiffres_non_retrouves(juste, classement, POIDS_PAR_DEFAUT) == []
    assert chiffres_non_retrouves(juste + ", VAN 99.7", classement, POIDS_PAR_DEFAUT) == ["99.7"]


def test_citation_verifiee_seulement_si_rendue_par_l_outil(fausse_recherche):
    sortie = pipeline_fixe("émissions de CO2", PROJETS_EXEMPLE, DOCUMENTS_EXEMPLE)
    assert citation_verifiee("Passage cité : aucun.", sortie)
    assert citation_verifiee("Passage cité : P001 « Réduction attendue des émissions de CO2. »", sortie)
    assert not citation_verifiee("Passage cité : P001 « Le projet crée 5 000 emplois. »", sortie)
