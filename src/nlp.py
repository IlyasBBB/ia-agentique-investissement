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
