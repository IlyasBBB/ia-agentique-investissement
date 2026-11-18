"""Rôle 3 : NLP et RAG.

Deux fonctions du contrat (docs/contrat.md) :

- extract(document) : trois notes qualitatives de 1 à 5 et une justification par critère ;
- retrieve(question, k) : les k passages des documents de projets les plus proches de la question.

Les documents sont les fichiers data/docs/<id_projet>.txt. Seuls l'objectif de développement
(PDO) et le résumé sont analysés : l'en-tête (pays, coût, durée...) est déjà dans les chiffres.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

journal = logging.getLogger(__name__)

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DOCS = RACINE / "data" / "docs"
DOSSIER_MODELES = RACINE / "models" / "nlp"  # ignoré par Git

CRITERES_TEXTE = ("impact_env_social", "alignement_strategique", "faisabilite")
NOTE_NEUTRE = 3
MODELE_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
PHRASES_PAR_PASSAGE = 3
PAS_PASSAGE = 2  # fenêtre de 3 phrases avançant de 2 : une phrase de recouvrement
MOTS_MAX_PHRASE = 60  # au-delà, une phrase est coupée : un passage fait au plus 180 mots

MARQUEUR_PDO = "Objectif de développement (PDO) :"
MARQUEUR_RESUME = "Résumé :"
ABSENT = "Non disponible."


# --- Lecture et nettoyage des documents -------------------------------------


def nettoyer(texte: str) -> str:
    """Répare les défauts fréquents des textes de la Banque mondiale.

    Mots coupés par un tiret de fin de ligne (« UTILIZA- TION »), espaces multiples,
    phrases entièrement en majuscules (anciens PDO) remises en minuscules.
    """
    texte = re.sub(r"(\w)- (\w)", r"\1\2", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    lettres = [c for c in texte if c.isalpha()]
    if lettres and sum(c.isupper() for c in lettres) / len(lettres) > 0.8:
        texte = ". ".join(p.strip().capitalize() for p in texte.lower().split(". "))
    return texte


def corps(document: str) -> str:
    """Rend le texte utile d'un document : PDO puis résumé, nettoyés.

    Si le document ne suit pas le format de data/docs/, tout le texte est gardé.
    """
    if not document:
        return ""
    if MARQUEUR_PDO not in document:
        return nettoyer(document)
    apres_pdo = document.split(MARQUEUR_PDO, 1)[1]
    pdo, _, resume = apres_pdo.partition(MARQUEUR_RESUME)
    morceaux = [nettoyer(m) for m in (pdo, resume) if m.strip() and m.strip() != ABSENT]
    return " ".join(m for m in morceaux if m)


def titre(document: str) -> str:
    """Nom du projet lu dans l'en-tête (chaîne vide s'il manque)."""
    trouve = re.search(r"^Projet : (.+)$", document or "", flags=re.MULTILINE)
    return trouve.group(1).strip() if trouve else ""


def phrases(texte: str) -> list[str]:
    """Découpe un texte en phrases (règle simple : ponctuation forte suivie d'un espace).

    Une « phrase » de plus de MOTS_MAX_PHRASE mots (énumération sans point) est coupée en morceaux.
    """
    resultat = []
    for phrase in re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9(])", texte):
        mots = phrase.split()
        for debut in range(0, len(mots), MOTS_MAX_PHRASE):
            morceau = " ".join(mots[debut:debut + MOTS_MAX_PHRASE])
            if len(morceau) > 2:
                resultat.append(morceau)
    return resultat


def charger_documents(dossier: Path = DOSSIER_DOCS) -> dict[str, str]:
    """Lit tous les documents : {id_projet: texte brut}."""
    return {f.stem: f.read_text(encoding="utf-8") for f in sorted(dossier.glob("*.txt"))}


def decouper_passages(documents: dict[str, str]) -> list[dict]:
    """Découpe chaque document en passages de quelques phrases qui se recouvrent.

    Chaque passage garde le nom du projet à part : il est ajouté au texte indexé
    (il aide à situer un passage isolé) mais pas au passage rendu à l'agent.
    """
    passages = []
    for id_projet, document in documents.items():
        liste = phrases(corps(document))
        nom = titre(document)
        if not liste:
            continue
        dernier = max(len(liste) - PHRASES_PAR_PASSAGE, 0)
        debuts = list(range(0, dernier + 1, PAS_PASSAGE))
        if debuts[-1] != dernier:
            debuts.append(dernier)  # dernière fenêtre, pour ne pas perdre la fin du texte
        for debut in debuts:
            texte = " ".join(liste[debut:debut + PHRASES_PAR_PASSAGE])
            passages.append({"id_projet": id_projet, "passage": texte, "indexe": f"{nom}. {texte}"})
    return passages


# --- Recherche de passages (retrieve) ---------------------------------------


class IndexTfidf:
    """Moteur de référence : TF-IDF sur les passages, similarité cosinus."""

    nom = "tfidf"

    def __init__(self, passages: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.passages = passages
        self.vectoriseur = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        self.matrice = self.vectoriseur.fit_transform([p["indexe"] for p in passages])

    def scores(self, question: str) -> np.ndarray:
        """Similarité cosinus entre la question et chaque passage (vecteurs déjà normés)."""
        return (self.matrice @ self.vectoriseur.transform([question]).T).toarray().ravel()


class IndexEmbeddings:
    """Moteur avancé : embeddings de phrases (sentence-transformers), similarité cosinus."""

    nom = "embeddings"

    def __init__(self, passages: list[dict], signature: str):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # sorties propres pour l'agent
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer

        self.passages = passages
        try:  # modèle déjà téléchargé : aucun accès réseau (démo hors ligne)
            self.modele = SentenceTransformer(MODELE_EMBEDDINGS, device="cpu", local_files_only=True)
        except Exception:
            journal.info("Téléchargement du modèle %s (une seule fois)...", MODELE_EMBEDDINGS)
            self.modele = SentenceTransformer(MODELE_EMBEDDINGS, device="cpu")
        fichier = DOSSIER_MODELES / f"embeddings_{signature}.npy"
        if fichier.exists():
            self.matrice = np.load(fichier)
        else:
            journal.info("Calcul des embeddings de %d passages (une seule fois)...", len(passages))
            self.matrice = self.modele.encode(
                [p["indexe"] for p in passages], batch_size=64, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True,
            ).astype(np.float32)
            DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
            np.save(fichier, self.matrice)

    def scores(self, question: str) -> np.ndarray:
        """Produit scalaire entre vecteurs normés = similarité cosinus."""
        q = self.modele.encode([question], normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)[0]
        return self.matrice @ q


class IndexHybride:
    """Mélange des deux moteurs : alpha * embeddings + (1 - alpha) * TF-IDF.

    Les deux scores sont d'abord ramenés entre 0 et 1 pour chaque question (min-max),
    car la similarité TF-IDF est en général bien plus petite que celle des embeddings.
    """

    nom = "hybride"

    def __init__(self, tfidf: IndexTfidf, embeddings: IndexEmbeddings, alpha: float = 0.5):
        self.tfidf, self.embeddings, self.alpha = tfidf, embeddings, alpha
        self.passages = tfidf.passages

    @staticmethod
    def _normer(s: np.ndarray) -> np.ndarray:
        etendue = s.max() - s.min()
        return (s - s.min()) / etendue if etendue > 0 else np.zeros_like(s)

    def scores(self, question: str) -> np.ndarray:
        """Score combiné de chaque passage."""
        return (self.alpha * self._normer(self.embeddings.scores(question))
                + (1 - self.alpha) * self._normer(self.tfidf.scores(question)))


def signature_documents(documents: dict[str, str]) -> str:
    """Empreinte du corpus et du découpage : l'index est recalculé si l'un des deux change."""
    empreinte = hashlib.sha256(
        f"{MODELE_EMBEDDINGS}|{PHRASES_PAR_PASSAGE}|{PAS_PASSAGE}|{MOTS_MAX_PHRASE}".encode()
    )
    for id_projet, texte in sorted(documents.items()):
        empreinte.update(id_projet.encode())
        empreinte.update(texte.encode("utf-8"))
    return empreinte.hexdigest()[:16]


def construire_index(methode: str | None = None, documents: dict[str, str] | None = None):
    """Construit l'index de recherche. methode : "embeddings" (défaut) ou "tfidf".

    Si les embeddings ne sont pas disponibles (bibliothèque ou modèle absent), repli sur TF-IDF.
    """
    methode = methode or os.getenv("NLP_RETRIEVE", "embeddings")
    documents = charger_documents() if documents is None else documents
    passages = decouper_passages(documents)
    if methode == "embeddings":
        try:
            return IndexEmbeddings(passages, signature_documents(documents))
        except Exception as erreur:  # pas de réseau au premier lancement, bibliothèque absente...
            journal.warning("Embeddings indisponibles (%s) : repli sur TF-IDF.", erreur)
    return IndexTfidf(passages)


@lru_cache(maxsize=2)
def index_par_defaut(methode: str | None = None):
    """Index sur data/docs/, construit au premier appel puis gardé en mémoire."""
    return construire_index(methode)


def chercher(index, question: str, k: int = 3) -> list[dict]:
    """Les k meilleurs passages d'un index, un seul par projet, du plus au moins pertinent."""
    if not question or not question.strip() or k <= 0:
        return []
    scores = index.scores(question)
    resultats, vus = [], set()
    for i in np.argsort(-scores, kind="stable"):
        id_projet = index.passages[i]["id_projet"]
        if id_projet in vus:
            continue
        vus.add(id_projet)
        resultats.append({"id_projet": id_projet, "passage": index.passages[i]["passage"],
                          "score": round(float(scores[i]), 4)})
        if len(resultats) == k:
            break
    return resultats


def retrieve(question: str, k: int = 3) -> list[dict]:
    """Les k passages les plus pertinents pour la question, du plus au moins pertinent.

    Un seul passage par projet, pour que l'agent voie k projets différents.
    Format de chaque élément : {"id_projet", "passage", "score"} (modèle Passage du contrat).
    """
    if not question or not question.strip() or k <= 0:
        return []  # rien à chercher : inutile de charger l'index
    return chercher(index_par_defaut(), question, k)


# --- Extraction des notes (extract) -----------------------------------------
# Modèle livré : combinaison de deux signaux, choisie sur une validation notée à la main
# (scripts/evaluer_extract_nlp.py, reports/role3/choix_extract.json) :
#   - le solde de mots-clés de la grille (mots positifs moins mots négatifs) ;
#   - la note attendue d'une régression logistique sur TF-IDF, apprise sur 640 documents notés
#     par le LLM local selon la grille.
# Chaque signal est centré-réduit sur la validation, on fait leur moyenne, puis des seuils
# convertissent ce score en note de 1 à 5. Les seuils redonnent, sur la validation, la répartition
# des notes de référence : le modèle parle la même « échelle » que l'annotateur, pas celle du LLM.

DOSSIER_ANNOTATIONS = RACINE / "reports" / "role3" / "annotations"
FICHIER_NOTES_LLM = DOSSIER_ANNOTATIONS / "notes_llm.csv"
FICHIER_JEUX = DOSSIER_ANNOTATIONS / "jeux_extract.csv"
FICHIER_VALIDATION = DOSSIER_ANNOTATIONS / "validation_reference.csv"
MOTS_MINIMUM = 15  # en dessous, le texte est jugé trop court : note neutre
NOTES_POSSIBLES = np.array([1, 2, 3, 4, 5])

MOTS_CLES: dict[str, dict[str, tuple[str, ...]]] = {
    "impact_env_social": {
        "plus": ("poverty", "poor", "vulnerable", "women", "gender", "health", "nutrition", "climate",
                 "environment", "biodiversity", "resilience", "access to", "inclusion", "emission",
                 "clean", "water supply", "sanitation", "livelihood", "social"),
        "moins": ("resettlement", "displacement", "pollution", "deforestation"),
    },
    "alignement_strategique": {
        "plus": ("strategy", "national", "government's", "program", "reform", "priority", "policy",
                 "sustainable development", "climate", "human capital", "digital", "employment",
                 "competitiveness"),
        "moins": (),
    },
    "faisabilite": {
        "plus": ("component", "implementation", "monitoring", "evaluation", "capacity", "project management",
                 "unit", "pilot", "second phase", "institutional", "technical assistance", "training"),
        "moins": ("conflict", "crisis", "emergency", "fragile", "post-war", "postwar", "weak capacity"),
    },
}
# Solde minimal pour obtenir 2, 3, 4, 5, réglé sur la validation pour chaque critère.
# Avec ces seuils, un texte sans aucun mot-clé reçoit la note neutre 3.
SEUILS_MOTS_CLES: dict[str, tuple[int, ...]] = {
    "impact_env_social": (-3, -1, 1, 3),
    "alignement_strategique": (-2, 0, 2, 4),
    "faisabilite": (-2, 0, 2, 4),
}
# Régularisation de la régression logistique, choisie sur la validation pour chaque critère.
C_PAR_CRITERE = {"impact_env_social": 0.3, "alignement_strategique": 0.3, "faisabilite": 0.03}


def trop_court(texte: str) -> bool:
    """Vrai si le texte utile est trop court pour juger (note neutre)."""
    return len(texte.split()) < MOTS_MINIMUM


def neutre(raison: str = "Texte trop court pour juger : note neutre.") -> dict:
    """Sortie neutre (note 3 partout), au format du contrat."""
    return {**{c: NOTE_NEUTRE for c in CRITERES_TEXTE}, "justifications": {c: raison for c in CRITERES_TEXTE}}


def solde_mots_cles(texte: str, critere: str) -> int:
    """Nombre d'occurrences des mots-clés positifs du critère, moins celles des négatifs."""
    minuscule = texte.lower()
    mots = MOTS_CLES[critere]
    return sum(minuscule.count(m) for m in mots["plus"]) - sum(minuscule.count(m) for m in mots["moins"])


def note_par_seuils(solde: int, seuils: tuple[int, ...]) -> int:
    """Convertit un solde (mots positifs moins négatifs) en note de 1 à 5."""
    return 1 + sum(solde >= s for s in seuils)


def extract_mots_cles(document: str) -> dict:
    """Modèle de référence : notes tirées du seul solde de mots-clés."""
    texte = corps(document)
    if trop_court(texte):
        return neutre()
    liste = phrases(texte)
    notes, justifications = {}, {}
    for critere, mots in MOTS_CLES.items():
        notes[critere] = note_par_seuils(solde_mots_cles(texte, critere), SEUILS_MOTS_CLES[critere])
        trouvees = [p for p in liste if any(m in p.lower() for m in mots["plus"] + mots["moins"])]
        justifications[critere] = trouvees[0] if trouvees else "Aucun mot-clé du critère dans le texte."
    return {**notes, "justifications": justifications}


class ModeleExtract:
    """Combinaison mots-clés + TF-IDF, recalée sur l'échelle des notes de référence."""

    def __init__(self, C: dict[str, float] | None = None, min_df: int = 2):
        self.C = C or dict(C_PAR_CRITERE)
        self.min_df = min_df
        self.vectoriseur = None
        self.modeles: dict = {}
        self.normes: dict[str, tuple[float, float, float, float]] = {}
        self.seuils: dict[str, np.ndarray] = {}

    # Apprentissage --------------------------------------------------------------

    def fit(self, textes: list[str], notes, textes_validation: list[str] | None = None,
            notes_validation=None) -> "ModeleExtract":
        """Apprend vocabulaire et régressions sur l'entraînement, puis le recalage sur la validation.

        Sans validation, seule la partie TF-IDF est apprise (utilisée par l'évaluation).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectoriseur = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=self.min_df,
                                           sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
        X = self.vectoriseur.fit_transform(textes)
        for critere in CRITERES_TEXTE:
            self.modeles[critere] = LogisticRegression(
                C=self.C[critere], class_weight="balanced", max_iter=2000, random_state=42
            ).fit(X, notes[critere])
        if textes_validation is not None:
            self.recaler(textes_validation, notes_validation)
        return self

    def recaler(self, textes: list[str], notes) -> None:
        """Normes des deux signaux et seuils de notes, calculés sur la validation notée à la main."""
        attendu = self.note_attendue(textes)
        for critere in CRITERES_TEXTE:
            soldes = np.array([solde_mots_cles(t, critere) for t in textes], dtype=float)
            self.normes[critere] = (attendu[critere].mean(), attendu[critere].std(), soldes.mean(), soldes.std())
            score = self._combiner(critere, attendu[critere], soldes)
            cumul = np.cumsum([np.mean(np.asarray(notes[critere]) == k) for k in NOTES_POSSIBLES])[:-1]
            self.seuils[critere] = np.quantile(score, cumul)

    # Prédiction -----------------------------------------------------------------

    def note_attendue(self, textes: list[str]) -> dict[str, np.ndarray]:
        """Espérance de la note selon les probabilités de chaque régression logistique."""
        X = self.vectoriseur.transform(textes)
        return {c: m.predict_proba(X) @ m.classes_ for c, m in self.modeles.items()}

    def _combiner(self, critere: str, attendu: np.ndarray, soldes: np.ndarray) -> np.ndarray:
        m_a, e_a, m_s, e_s = self.normes[critere]
        return ((attendu - m_a) / e_a + (soldes - m_s) / e_s) / 2

    def scores(self, textes: list[str]) -> dict[str, np.ndarray]:
        """Score combiné de chaque texte pour chaque critère."""
        attendu = self.note_attendue(textes)
        return {c: self._combiner(c, attendu[c], np.array([solde_mots_cles(t, c) for t in textes], dtype=float))
                for c in CRITERES_TEXTE}

    def predire(self, textes: list[str]) -> dict[str, np.ndarray]:
        """Notes de 1 à 5 pour une liste de textes, sans la règle du texte court."""
        return {c: 1 + np.searchsorted(self.seuils[c], s, side="right") for c, s in self.scores(textes).items()}

    # Interprétabilité ---------------------------------------------------------------

    def direction(self, critere: str) -> np.ndarray:
        """Poids de chaque mot sur la note attendue : positif, il tire la note vers le haut."""
        modele = self.modeles[critere]
        return modele.coef_.T @ (modele.classes_ - modele.classes_.mean()) / len(modele.classes_)

    def mots_influents(self, critere: str, n: int = 15) -> dict[str, list[tuple[str, float]]]:
        """Les n mots ou paires de mots qui tirent le plus la note vers le haut, et vers le bas."""
        d = self.direction(critere)
        mots = self.vectoriseur.get_feature_names_out()
        ordre = np.argsort(d)
        return {"vers_le_haut": [(mots[i], float(d[i])) for i in ordre[::-1][:n]],
                "vers_le_bas": [(mots[i], float(d[i])) for i in ordre[:n]]}

    def justifier(self, texte: str, critere: str, note: int) -> str:
        """Phrase du texte qui pèse le plus dans le sens de la note (vers le haut si note >= 3)."""
        liste = phrases(texte) or [texte]
        _, e_a, _, e_s = self.normes[critere]
        poids = (self.vectoriseur.transform(liste) @ self.direction(critere)) / e_a + np.array(
            [solde_mots_cles(p, critere) for p in liste]) / e_s
        return liste[int(np.argmax(poids) if note >= NOTE_NEUTRE else np.argmin(poids))]

    def extraire(self, document: str) -> dict:
        """Sortie au format du contrat pour un document."""
        texte = corps(document)
        if trop_court(texte):
            return neutre()
        notes = {c: int(v[0]) for c, v in self.predire([texte]).items()}
        return {**notes, "justifications": {c: self.justifier(texte, c, n) for c, n in notes.items()}}


def textes_annotes(ids) -> list[str]:
    """Texte utile des projets demandés."""
    return [corps((DOSSIER_DOCS / f"{i}.txt").read_text(encoding="utf-8")) for i in ids]


@lru_cache(maxsize=1)
def modele_extract() -> ModeleExtract | None:
    """Modèle livré, appris au premier appel (quelques secondes) puis enregistré dans models/nlp/.

    Entraînement : 640 documents notés par le LLM. Recalage : 160 documents notés à la main.
    Le jeu de test n'est jamais utilisé. Réappris si les annotations ou les réglages changent.
    Rend None si les annotations sont absentes : extract repasse alors aux mots-clés.
    """
    import joblib
    import pandas as pd

    fichiers = (FICHIER_NOTES_LLM, FICHIER_JEUX, FICHIER_VALIDATION)
    if not all(f.exists() for f in fichiers):
        journal.warning("Annotations absentes : extract utilise les mots-clés.")
        return None
    empreinte = hashlib.sha256(b"".join(f.read_bytes() for f in fichiers) + json.dumps(
        [C_PAR_CRITERE, SEUILS_MOTS_CLES, MOTS_CLES], sort_keys=True).encode()).hexdigest()[:16]
    fichier = DOSSIER_MODELES / f"extract_{empreinte}.joblib"
    if fichier.exists():
        return joblib.load(fichier)
    jeux = pd.read_csv(FICHIER_JEUX)
    ent = pd.read_csv(FICHIER_NOTES_LLM).merge(jeux[jeux.jeu == "entrainement"], on="id_projet")
    val = pd.read_csv(FICHIER_VALIDATION)
    try:
        modele = ModeleExtract(C_PAR_CRITERE).fit(textes_annotes(ent.id_projet), ent,
                                                  textes_annotes(val.id_projet), val)
    except ValueError as erreur:  # annotations incomplètes
        journal.warning("Apprentissage impossible (%s) : extract utilise les mots-clés.", erreur)
        return None
    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    for ancien in DOSSIER_MODELES.glob("extract_*.joblib"):  # versions périmées du modèle
        ancien.unlink()
    joblib.dump(modele, fichier)
    journal.info("Modèle extract appris sur %d documents, recalé sur %d, enregistré dans %s",
                 len(ent), len(val), fichier)
    return modele


def extract(document: str) -> dict:
    """Trois notes de 1 à 5 tirées du texte d'un projet, avec une phrase de justification par critère.

    Texte vide ou trop court (moins de 15 mots utiles) : note neutre 3 partout.
    Format : modèle Extraction du contrat.
    """
    if trop_court(corps(document or "")):
        return neutre()
    modele = modele_extract()
    return modele.extraire(document) if modele else extract_mots_cles(document)
