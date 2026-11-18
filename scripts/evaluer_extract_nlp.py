"""Évalue extract et choisit le modèle livré, sans fuite de données.

Données :
- entrainement (640 documents) : notes du LLM local (étiquettes automatiques) ;
- validation (160 documents) : notes de référence faites à la main selon la grille ;
- test (100 documents) : notes de référence faites à la main, ouvert une seule fois à la fin.

Pourquoi une validation notée à la main : le LLM n'applique pas la grille comme l'annotateur
(il met par exemple 5 en alignement à plus de la moitié des documents). Une validation notée par
le LLM choisirait le modèle qui imite le mieux le LLM, pas celui qui note le mieux selon la grille.

Candidats, tous réglés sur la validation :
1. mots_cles : solde de mots-clés et seuils, décalage des seuils réglé par critère ;
2. tfidf_llm : TF-IDF + régression logistique apprise sur les notes du LLM, C réglé par critère ;
3. tfidf_llm_recale : le même modèle, dont la note attendue est recalée sur l'échelle de référence
   (seuils placés pour retrouver la répartition des notes de la validation) ;
4. combinaison : moyenne du solde de mots-clés et de la note attendue TF-IDF (centrés-réduits),
   recalée de la même façon ;
5. tfidf_humain : TF-IDF + régression logistique apprise sur les 160 notes de validation,
   C choisi et score estimé par validation croisée à 5 plis (vocabulaire appris dans chaque pli).

Règle de choix, fixée avant de regarder le test : meilleur kappa moyen sur la validation ;
si l'écart avec un modèle plus simple est inférieur à 0,02, on garde le plus simple
(ordre de simplicité : mots_cles, tfidf_llm, tfidf_llm_recale, combinaison, tfidf_humain).

Pour tous les modèles, un texte de moins de 15 mots reçoit la note neutre 3 (même règle partout).
Mesure principale : kappa pondéré quadratique (0 = hasard, 1 = accord parfait), plus l'exactitude
et l'erreur absolue moyenne.

Usage : python scripts/evaluer_extract_nlp.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import nlp  # noqa: E402

DOSSIER = nlp.RACINE / "reports" / "role3"
ANNOT = DOSSIER / "annotations"
CRITERES = list(nlp.CRITERES_TEXTE)
NOTES = np.array([1, 2, 3, 4, 5])
VALEURS_C = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SEUILS_BASE = (0, 2, 4, 6)  # solde minimal pour 2, 3, 4, 5 avant décalage
DECALAGES = (-4, -3, -2, -1, 0, 1, 2, 3)
ORDRE_SIMPLICITE = ("mots_cles", "tfidf_llm", "tfidf_llm_recale", "combinaison", "tfidf_humain")
MARGE_EGALITE = 0.02
GRAINE = 42

journal = logging.getLogger("evaluer_extract")


# --- Mesures ------------------------------------------------------------------


def kappa(vrai, predit) -> float:
    """Kappa pondéré quadratique sur l'échelle 1 à 5."""
    return float(cohen_kappa_score(vrai, predit, weights="quadratic", labels=list(NOTES)))


def mesures(vrai: pd.DataFrame, predit: pd.DataFrame) -> pd.DataFrame:
    """Exactitude, MAE et kappa pour chaque critère, plus la moyenne."""
    t = pd.DataFrame({c: {"exactitude": accuracy_score(vrai[c], predit[c]),
                          "mae": mean_absolute_error(vrai[c], predit[c]),
                          "kappa": kappa(vrai[c], predit[c])} for c in CRITERES}).T
    t.loc["moyenne"] = t.mean()
    return t


def intervalle(vrai: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, n: int = 2000) -> tuple[float, float]:
    """IC à 95 % (bootstrap sur les documents) de la différence de kappa moyen a - b."""
    rng = np.random.default_rng(GRAINE)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(vrai), len(vrai))
        diffs.append(np.mean([kappa(vrai[c].iloc[i], a[c].iloc[i]) - kappa(vrai[c].iloc[i], b[c].iloc[i])
                              for c in CRITERES]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --- Données ------------------------------------------------------------------


def textes_de(ids) -> list[str]:
    """Texte utile (objectif et résumé) de chaque projet."""
    return [nlp.corps((nlp.DOSSIER_DOCS / f"{i}.txt").read_text(encoding="utf-8")) for i in ids]


def court(textes: list[str]) -> np.ndarray:
    """Masque des textes trop courts (note neutre)."""
    return np.array([nlp.trop_court(t) for t in textes])


def regle_court(notes: pd.DataFrame, textes: list[str]) -> pd.DataFrame:
    """Applique la note neutre 3 aux textes trop courts."""
    notes = notes.copy()
    notes.loc[court(textes), CRITERES] = nlp.NOTE_NEUTRE
    return notes.astype(int)


# --- Briques des modèles --------------------------------------------------------


def soldes(textes: list[str]) -> pd.DataFrame:
    """Solde de mots-clés (positifs moins négatifs) de chaque critère."""
    return pd.DataFrame([{c: sum(t.lower().count(m) for m in nlp.MOTS_CLES[c]["plus"])
                          - sum(t.lower().count(m) for m in nlp.MOTS_CLES[c]["moins"]) for c in CRITERES}
                         for t in textes])


def notes_mots_cles(s: pd.DataFrame, decalages: dict[str, int]) -> pd.DataFrame:
    """Notes des mots-clés avec un décalage des seuils par critère."""
    return pd.DataFrame({c: [nlp.note_par_seuils(int(v), seuils_decales(decalages[c])) for v in s[c]]
                         for c in CRITERES})


def seuils_decales(decalage: int) -> tuple[int, ...]:
    """Seuils de base décalés en bloc."""
    return tuple(x + decalage for x in SEUILS_BASE)


def note_attendue(modele, X) -> np.ndarray:
    """Espérance de la note selon les probabilités de la régression logistique."""
    return modele.predict_proba(X) @ modele.classes_


def seuils_repartition(score_val: np.ndarray, notes_val: pd.Series) -> np.ndarray:
    """Seuils qui redonnent, sur la validation, la répartition des notes de référence."""
    cumul = np.cumsum([np.mean(notes_val == k) for k in NOTES])[:-1]
    return np.quantile(score_val, cumul)


def appliquer_seuils(score: np.ndarray, seuils: np.ndarray) -> np.ndarray:
    """Note de 1 à 5 selon la position du score par rapport aux seuils."""
    return 1 + np.searchsorted(seuils, score, side="right")


# --- Programme principal ----------------------------------------------------------


def main() -> None:
    """Réglages et choix sur la validation, puis mesure unique sur le test."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    jeux = pd.read_csv(ANNOT / "jeux_extract.csv")
    llm = pd.read_csv(nlp.FICHIER_NOTES_LLM).set_index("id_projet")
    ent = llm.loc[jeux[jeux.jeu == "entrainement"].id_projet].reset_index()
    val = pd.read_csv(ANNOT / "validation_reference.csv")
    test = pd.read_csv(ANNOT / "test_reference.csv")
    t_ent, t_val, t_test = textes_de(ent.id_projet), textes_de(val.id_projet), textes_de(test.id_projet)
    journal.info("Entraînement %d (notes LLM), validation %d et test %d (notes de référence)",
                 len(t_ent), len(t_val), len(t_test))

    vectoriseur = nlp.ModeleExtract().fit(t_ent, ent).vectoriseur  # vocabulaire appris sur l'entraînement
    X_ent, X_val, X_test = (vectoriseur.transform(t) for t in (t_ent, t_val, t_test))
    s_val, s_test = soldes(t_val), soldes(t_test)

    def lr(C: float, y) -> LogisticRegression:
        return LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=GRAINE).fit(X_ent, y)

    reglages, score_val, param = [], {}, {}

    # 1. mots-clés : décalage des seuils par critère
    dec = {}
    for c in CRITERES:
        res = {d: kappa(val[c], regle_court(notes_mots_cles(s_val, {k: d for k in CRITERES}), t_val)[c]) for d in DECALAGES}
        dec[c] = max(res, key=res.get)
        reglages += [{"modele": "mots_cles", "critere": c, "parametre": d, "kappa_validation": k} for d, k in res.items()]
    param["mots_cles"] = {"decalage_seuils": dec}

    # 2. TF-IDF appris sur les notes du LLM : C par critère
    meilleur_C, modeles_llm = {}, {}
    for c in CRITERES:
        res = {}
        for C in VALEURS_C:
            p = pd.DataFrame({c: lr(C, ent[c]).predict(X_val)})
            res[C] = kappa(val[c], regle_court(p.reindex(columns=CRITERES, fill_value=3), t_val)[c])
            reglages.append({"modele": "tfidf_llm", "critere": c, "parametre": C, "kappa_validation": res[C]})
        meilleur_C[c] = max(res, key=res.get)
        modeles_llm[c] = lr(meilleur_C[c], ent[c])
    param["tfidf_llm"] = {"C": meilleur_C}

    # 3. recalage sur l'échelle de référence (aucun paramètre libre : répartition de la validation)
    attendu_val = {c: note_attendue(modeles_llm[c], X_val) for c in CRITERES}
    attendu_test = {c: note_attendue(modeles_llm[c], X_test) for c in CRITERES}
    seuils_recale = {c: seuils_repartition(attendu_val[c], val[c]) for c in CRITERES}

    # 4. combinaison : c'est la classe du module livré (src/nlp.py) qui est évaluée
    combinaison = nlp.ModeleExtract(meilleur_C).fit(t_ent, ent, t_val, val)

    # 5. TF-IDF appris sur les notes de référence de la validation (validation croisée)
    plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
    C_humain, cv_humain = {}, {}
    for c in CRITERES:
        res = {}
        for C in VALEURS_C:
            pred = np.zeros(len(t_val), dtype=int)
            for i_app, i_eval in plis.split(t_val, val[c]):
                pipe = make_pipeline(TfidfVectorizer(**vectoriseur.get_params()),
                                     LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=GRAINE))
                pipe.fit([t_val[i] for i in i_app], val[c].iloc[i_app])
                pred[i_eval] = pipe.predict([t_val[i] for i in i_eval])
            res[C] = pred
            reglages.append({"modele": "tfidf_humain", "critere": c, "parametre": C,
                             "kappa_validation": kappa(val[c], regle_court(pd.DataFrame({k: pred if k == c else 3 for k in CRITERES}), t_val)[c])})
        meilleur = max(res, key=lambda C: kappa(val[c], res[C]))
        C_humain[c], cv_humain[c] = meilleur, res[meilleur]
    param["tfidf_humain"] = {"C": C_humain}

    # Prédictions de chaque candidat sur la validation (validation croisée pour tfidf_humain)
    pred_val = {
        "mots_cles": notes_mots_cles(s_val, dec),
        "tfidf_llm": pd.DataFrame({c: modeles_llm[c].predict(X_val) for c in CRITERES}),
        "tfidf_llm_recale": pd.DataFrame({c: appliquer_seuils(attendu_val[c], seuils_recale[c]) for c in CRITERES}),
        "combinaison": pd.DataFrame(combinaison.predire(t_val)),
        "tfidf_humain": pd.DataFrame(cv_humain),
    }
    pred_val = {k: regle_court(v, t_val) for k, v in pred_val.items()}
    res_val = {k: mesures(val, v) for k, v in pred_val.items()}
    kappa_val = {k: r.loc["moyenne", "kappa"] for k, r in res_val.items()}
    journal.info("Kappa moyen sur la validation : %s", {k: round(v, 3) for k, v in kappa_val.items()})

    # Règle de choix
    meilleur = max(kappa_val, key=kappa_val.get)
    choisi = next(k for k in ORDRE_SIMPLICITE if kappa_val[k] >= kappa_val[meilleur] - MARGE_EGALITE)
    journal.info("Meilleur sur la validation : %s ; modèle retenu (règle de simplicité) : %s", meilleur, choisi)

    # Test, ouvert une seule fois
    humain_final = {c: make_pipeline(TfidfVectorizer(**vectoriseur.get_params()),
                                     LogisticRegression(C=C_humain[c], class_weight="balanced", max_iter=2000,
                                                        random_state=GRAINE)).fit(t_val, val[c]) for c in CRITERES}
    majo = val[CRITERES].mode().iloc[0]
    pred_test = {
        "majoritaire": pd.DataFrame([majo.to_dict()] * len(t_test)),
        "mots_cles": notes_mots_cles(s_test, dec),
        "tfidf_llm": pd.DataFrame({c: modeles_llm[c].predict(X_test) for c in CRITERES}),
        "tfidf_llm_recale": pd.DataFrame({c: appliquer_seuils(attendu_test[c], seuils_recale[c]) for c in CRITERES}),
        "combinaison": pd.DataFrame(combinaison.predire(t_test)),
        "tfidf_humain": pd.DataFrame({c: humain_final[c].predict(t_test) for c in CRITERES}),
        "llm": llm.loc[test.id_projet, CRITERES].reset_index(drop=True),
    }
    pred_test = {k: regle_court(v, t_test) for k, v in pred_test.items()}
    res_test = pd.concat({k: mesures(test, v) for k, v in pred_test.items()}, names=["modele", "critere"])
    journal.info("Test (%d documents) :\n%s", len(t_test), res_test.round(3).to_string())
    for autre in ("mots_cles", "tfidf_llm", "llm", "majoritaire"):
        if autre != choisi:
            bas, haut = intervalle(test, pred_test[choisi], pred_test[autre])
            diff = res_test.loc[(choisi, "moyenne"), "kappa"] - res_test.loc[(autre, "moyenne"), "kappa"]
            journal.info("kappa moyen %s - %s : %+.3f [%+.3f ; %+.3f]", choisi, autre, diff, bas, haut)

    # Le module livré doit utiliser exactement les réglages choisis ici
    seuils_choisis = {c: seuils_decales(dec[c]) for c in CRITERES}
    if nlp.C_PAR_CRITERE != meilleur_C or nlp.SEUILS_MOTS_CLES != seuils_choisis:
        journal.warning("Réglages à reporter dans src/nlp.py : C_PAR_CRITERE = %s ; SEUILS_MOTS_CLES = %s",
                        meilleur_C, seuils_choisis)
    else:
        journal.info("src/nlp.py utilise bien les réglages choisis sur la validation.")

    # Sorties
    pd.DataFrame(reglages).to_csv(DOSSIER / "reglages_extract.csv", index=False)
    pd.concat({k: r for k, r in res_val.items()}, names=["modele", "critere"]).to_csv(DOSSIER / "resultats_extract_validation.csv")
    res_test.to_csv(DOSSIER / "resultats_extract_test.csv")
    detail = test[["id_projet", "secteur", *CRITERES]].rename(columns={c: f"{c}_ref" for c in CRITERES})
    for nom, p in pred_test.items():
        for c in CRITERES:
            detail[f"{c}_{nom}"] = p[c].to_numpy()
    detail["mots"] = [len(t.split()) for t in t_test]
    detail.to_csv(DOSSIER / "predictions_extract_test.csv", index=False)
    (DOSSIER / "choix_extract.json").write_text(json.dumps(
        {"modele_retenu": choisi, "meilleur_validation": meilleur, "kappa_validation": kappa_val,
         "parametres": param}, indent=2, ensure_ascii=False), encoding="utf-8")

    # Vitesse des mots-clés sur les 2 000 documents
    documents = nlp.charger_documents()
    debut = time.perf_counter()
    for d in documents.values():
        nlp.extract_mots_cles(d)
    journal.info("Mots-clés sur %d documents : %.2f s", len(documents), time.perf_counter() - debut)


if __name__ == "__main__":
    main()
