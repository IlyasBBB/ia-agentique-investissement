"""Blocs HTML de l'application : bandeau d'indicateurs, cartes, alerte, journal.

Chaque fonction rend une chaîne HTML à afficher avec st.markdown. Tout texte qui
vient des données ou du LLM (nom de projet, résultat d'outil, CSV importé) passe
par `propre` : il est affiché tel quel et ne peut pas être lu comme du HTML.
Les classes CSS utilisées ici sont définies dans app/style.css.
"""
from __future__ import annotations

import html
import re

import pandas as pd

ESPACE_FINE = " "  # espace insécable fine : séparateur des milliers en français
ICONE_ALERTE = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2 20h20L12 3z"/>'
                '<path d="M12 10v4"/><path d="M12 17h.01"/></svg>')


def propre(valeur: object) -> str:
    """Rend un texte sûr à placer dans du HTML (les signes < > & et les guillemets sont neutralisés)."""
    return html.escape("" if valeur is None else str(valeur))


def nombre(valeur: float, decimales: int = 0) -> str:
    """Écrit un nombre à la française : 12 345,6."""
    return f"{valeur:,.{decimales}f}".replace(",", ESPACE_FINE).replace(".", ",")


def _cote(libelle: str, valeur: str, detail: str, unite: str = "", voyant: bool = False) -> str:
    """Une case du bandeau : libellé, grande valeur, ligne de détail."""
    point = '<i class="voyant"></i>' if voyant else ""
    petit = f"<small>{propre(unite)}</small>" if unite else ""
    return (f'<div class="cote" role="listitem"><span class="cote-libelle">{point}{propre(libelle)}</span>'
            f'<span class="cote-valeur">{propre(valeur)}{petit}</span>'
            f'<span class="cote-detail">{propre(detail)}</span></div>')


def bandeau_portefeuille(projets: pd.DataFrame, nb_documents: int, nb_charges: int) -> str:
    """Bandeau des indicateurs du portefeuille retenu : projets, coût total, secteurs, pays, documents."""
    cout_md = projets["cout_total"].sum() / 1000  # cout_total est en millions de dollars
    cases = [
        _cote("Projets retenus", nombre(len(projets)), f"sur {nombre(nb_charges)} chargés", voyant=True),
        _cote("Coût total", nombre(cout_md, 1), "somme des coûts des projets", unite="Md$"),
        _cote("Secteurs", nombre(projets["secteur"].nunique()), "secteurs d'activité"),
        _cote("Pays", nombre(projets["pays"].nunique()), "pays ou zones"),
        _cote("Documents", nombre(nb_documents), f"{nb_documents / len(projets):.0%} des projets couverts"),
    ]
    return f'<div class="bandeau" role="list" aria-label="Indicateurs du portefeuille">{"".join(cases)}</div>'


NOTES_CARTE = (("impact_env_social", "Impact"), ("alignement_strategique", "Alignement"), ("faisabilite", "Faisabilité"))


def carte_projet(ligne: dict) -> str:
    """Carte d'un projet recommandé : rang, nom, score global, rentabilité, risque et les trois notes du texte.

    Le nom peut faire une ligne ou quatre : les chiffres sont poussés en bas de la carte pour
    rester alignés d'une carte à l'autre.
    """
    score = float(ligne["score"])
    rentabilite = float(ligne["rentabilite_predite"])
    sens = "hausse" if rentabilite >= 0 else "baisse"
    rentabilite_texte = f"{rentabilite:+.1f}".replace(".", ",")  # le signe + double la couleur
    lieu = " · ".join(str(ligne[c]) for c in ("secteur", "pays") if ligne.get(c))
    numero = ligne.get("rang_retenu", ligne["rang"])  # rang parmi les projets retenus (dossiers anormaux écartés)
    premiere = " premiere" if numero == 1 else ""
    nom = propre(ligne.get("nom", ""))
    notes = "".join(
        f'<div><dt>{titre}</dt><dd>{float(ligne[c]):.0f}<small>/5</small></dd></div>'
        for c, titre in NOTES_CARTE if ligne.get(c) is not None and not pd.isna(ligne[c]))
    return (
        f'<article class="carte{premiere}" aria-label="Projet recommandé n° {propre(numero)}">'
        f'<div class="carte-tete"><span class="carte-rang">N° {propre(numero)}</span>'
        f'<span class="carte-id">{propre(ligne["id_projet"])}</span></div>'
        f'<p class="carte-nom" title="{nom}">{nom}</p>'
        f'<p class="carte-lieu">{propre(lieu)}</p>'
        f'<div class="carte-chiffres">'
        f'<div class="jauge-ligne"><span>Score global <small>(sur 1)</small></span><b>{nombre(score, 3)}</b></div>'
        f'<div class="jauge" role="meter" aria-label="Score global sur 1" aria-valuemin="0" aria-valuemax="1" '
        f'aria-valuenow="{score:.3f}"><span style="width:{score:.1%}"></span></div>'
        f'<dl class="mesures">'
        f'<div><dt>Rentabilité prédite</dt><dd class="{sens}">{rentabilite_texte}{ESPACE_FINE}%</dd></div>'
        f'<div><dt>Probabilité de risque élevé</dt><dd>{float(ligne["proba_risque"]):.0%}</dd></div>'
        f'</dl>'
        + (f'<dl class="notes" aria-label="Notes tirées du document, sur 5">{notes}</dl>' if notes else "")
        + '</div></article>'
    )


def rang_affiche(ligne: dict) -> str:
    """Rang parmi les projets retenus, ou « écarté » pour un dossier anormal."""
    numero = ligne.get("rang_retenu", ligne.get("rang"))
    if ligne.get("anomalie") or numero is None or pd.isna(numero):
        return "écarté"
    return str(int(numero))


def fiche_projet(ligne: dict) -> str:
    """Fiche du projet cliqué dans le relief 3D : identité, rang, score et les sept critères."""
    def note(colonne: str) -> str:
        return f"{float(ligne[colonne]):.0f}/5" if ligne.get(colonne) is not None else "—"

    lieu = " · ".join(str(ligne[c]) for c in ("secteur", "pays") if ligne.get(c))
    mesures = [
        ("Rang", rang_affiche(ligne)), ("Score", nombre(float(ligne["score"]), 3)),
        ("Rentabilité prédite", f"{nombre(float(ligne['rentabilite_predite']), 1)}{ESPACE_FINE}%"),
        ("Risque élevé", f"{float(ligne['proba_risque']):.0%}"),
        ("Coût", f"{nombre(float(ligne['cout_total']))}{ESPACE_FINE}M$"),
        ("Durée", f"{nombre(float(ligne['duree_mois']))} mois"),
        ("Impact", note("impact_env_social")), ("Alignement", note("alignement_strategique")),
        ("Faisabilité", note("faisabilite")),
    ]
    anormale = bool(ligne.get("anomalie"))
    drapeau = '<span class="drapeau">Dossier anormal : écarté de la recommandation</span>' if anormale else ""
    cases = "".join(f"<div><dt>{propre(titre)}</dt><dd>{propre(valeur)}</dd></div>" for titre, valeur in mesures)
    return (f'<section class="fiche{" anormale" if anormale else ""}" aria-label="Fiche du projet">'
            f'<div class="fiche-tete"><span>{propre(ligne["id_projet"])} · {propre(lieu)}</span>{drapeau}</div>'
            f'<p class="fiche-nom">{propre(ligne.get("nom", ""))}</p><dl>{cases}</dl></section>')


def cartes_recommandes(recommandes: pd.DataFrame) -> str:
    """Les cartes des projets recommandés, côte à côte."""
    cartes = "".join(carte_projet(p) for p in recommandes.to_dict("records"))
    return f'<div class="cartes"><div class="cartes-grille">{cartes}</div></div>'


def alerte_anomalies(anormaux: list[str], limite: int = 15) -> str:
    """Encadré rouge : nombre de dossiers anormaux écartés et leurs identifiants."""
    suite = " …" if len(anormaux) > limite else ""
    if len(anormaux) == 1:
        titre = "1 dossier anormal écarté de la recommandation"
    else:
        titre = f"{len(anormaux)} dossiers anormaux écartés de la recommandation"
    return (f'<div class="alerte" role="alert">{ICONE_ALERTE}<div><strong>{propre(titre)}</strong>'
            f'<span>{propre(", ".join(anormaux[:limite]) + suite)}</span></div></div>')


# --- Recommandation rédigée ---------------------------------------------------------
# Le texte vient du LLM (ou du modèle de texte du pipeline fixe). Il suit à peu près le format
# demandé par le prompt, mais un petit modèle varie : « 1. P… » ou « - rang 1, P… », chiffres
# nommés ou non, un ou plusieurs passages. On le range en sections sans rien réécrire ; tout ce
# qui n'est pas reconnu reste affiché tel quel, en paragraphe. Chaque morceau passe par `propre`.

SECTIONS_REDACTION = (
    (r"projets? recommandés?", "Projets recommandés", "projets"),
    (r"alertes?|projets? à éviter", "Alertes", "alertes"),
    (r"passages? (?:cités?|à l'appui)", "Passage cité", "passage"),
    (r"limites?", "Limites", "limites"),
    (r"contrôle automatique", "Contrôle automatique", "controle"),
)
_TETE = re.compile(r"^\s*(" + "|".join(motif for motif, _, _ in SECTIONS_REDACTION) + r")\b\s*(\([^)]*\))?\s*:?\s*(.*)$",
                   re.IGNORECASE)
_PROJET = re.compile(r"^\s*(?:[-•]\s*)?(?:(\d+)[.)]\s*|rang (\d+),\s*)?(P\d{3,}\b.*?)\s:\s(.+)$")
_CITATION = re.compile(r"^\s*(?:[-•]\s*)?(?:\(?(P\d{3,})\)?\s*:\s*)?[«\"“]\s*(.+?)\s*[»\"”]?\s*$")
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")


def _section_de(ligne: str) -> tuple[str, str, str, str] | None:
    """Reconnaît une ligne de titre : (classe, titre affiché, reste de la ligne, précision entre parenthèses)."""
    trouve = _TETE.match(ligne)
    if not trouve:
        return None
    mot = trouve.group(1).lower()
    for motif, titre, classe in SECTIONS_REDACTION:
        if re.fullmatch(motif, mot, re.IGNORECASE):
            return classe, titre, trouve.group(3), (trouve.group(2) or "").strip("() ")
    return None


def _phrase(texte: str) -> str:
    """Une phrase du texte, avec une majuscule initiale et les décimales à la française."""
    texte = _DECIMAL.sub(",", texte.lstrip("-• ").strip())
    return texte[:1].upper() + texte[1:]


def _mesures(brut: str) -> tuple[list[str], str]:
    """Sépare les chiffres d'un projet et un éventuel extrait cité recopié dans la même ligne.

    Le prompt demande « score, rentabilité prédite, probabilité de risque élevé, notes ». Quand
    le LLM écrit les nombres sans leur nom (« 0.959, 10.63 %, 0.12, … »), le nom est déduit de la
    forme du nombre : pourcentage = rentabilité, nombre sur 5 = note, nombre seul = score puis risque.
    """
    coupure = min((i for i in (brut.find("«"), brut.find('"'), brut.find("“")) if i >= 0), default=-1)
    extrait = brut[coupure:].strip(" «»\"“”.") if coupure >= 0 else ""
    morceaux = [m.strip() for m in brut[:coupure if coupure >= 0 else None].rstrip(" .,").split(", ") if m.strip()]
    notes = iter(("impact", "alignement", "faisabilité"))
    seuls = iter(("score", "probabilité de risque élevé"))
    nommes = []
    for morceau in morceaux:
        if re.search(r"[A-Za-zÀ-ÿ]", morceau):
            nommes.append(morceau)
        elif morceau.endswith("%"):
            nommes.append(f"rentabilité prédite {morceau}")
        elif morceau.endswith("/5"):
            nommes.append(f"{next(notes, 'note')} {morceau}")
        else:
            nommes.append(f"{next(seuls, '')} {morceau}".strip())
    return [_DECIMAL.sub(",", m) for m in nommes], extrait


def _ligne_html(classe: str, ligne: str, precision: str = "") -> str:
    """Une ligne d'une section, mise en forme selon la section."""
    if classe in ("projets", "controle"):
        projet = _PROJET.match(ligne)
        if projet and classe == "projets":
            numero = projet.group(1) or projet.group(2) or ""
            mesures, extrait = _mesures(projet.group(4))
            tete = f'<span class="redac-num">{propre(numero)}</span>' if numero else ""
            citation = f'<q>{propre(extrait)}</q>' if extrait else ""
            return (f'<li class="redac-projet">{tete}<div><b>{propre(projet.group(3).strip())}</b>'
                    f'<span>{propre(" · ".join(mesures))}</span>{citation}</div></li>')
        return f"<li>{propre(_phrase(ligne))}</li>"
    if classe == "passage":
        citation = _CITATION.match(ligne)
        if citation and (citation.group(1) or precision or "«" in ligne):
            source = citation.group(1) or precision
            pied = f"<figcaption>Document du projet {propre(source)}</figcaption>" if source else ""
            return f"<figure><blockquote>{propre(citation.group(2).rstrip(' »”'))}</blockquote>{pied}</figure>"
    return f"<p>{propre(_phrase(ligne))}</p>"


def recommandation_redigee(texte: str) -> str:
    """Le texte de la recommandation rangé en sections lisibles (titres, liste des projets, citation)."""
    preambule, sections = [], []  # sections : [classe, titre, précision, lignes]
    for brute in (texte or "").splitlines():
        ligne = brute.strip()
        if not ligne:
            continue
        tete = _section_de(ligne)
        if tete:
            classe, titre, reste, precision = tete
            if sections and sections[-1][0] == classe:  # « Projets à éviter » après « Alertes » : même section
                section = sections[-1]
            else:
                section = [classe, titre, precision, []]
                sections.append(section)
            if classe == "alertes" and reste and ligne.lower().startswith("projet"):
                reste = ligne  # garder « Projets à éviter … » en entier
            if reste:
                section[3].append(reste)
        elif sections:
            sections[-1][3].append(ligne)
        else:
            preambule.append(ligne)
    blocs = [f'<p class="redac-preambule">{propre(p)}</p>' for p in preambule]
    for classe, titre, precision, lignes in sections:
        contenu = "".join(_ligne_html(classe, ligne, precision) for ligne in lignes)
        if classe in ("projets", "controle") and contenu.startswith("<li"):
            contenu = f'<ol class="redac-liste">{contenu}</ol>' if classe == "projets" else f"<ul>{contenu}</ul>"
        blocs.append(f'<section class="redac-{classe}"><h4>{propre(titre)}</h4>{contenu}</section>')
    return f'<article class="redaction">{"".join(blocs)}</article>'


def journal_etapes(etapes: list[dict]) -> str:
    """Le journal de l'agent en ligne du temps : numéro, outil, durée, arguments, résultat."""
    lignes = []
    for numero, etape in enumerate(etapes, 1):
        erreur = " erreur" if str(etape["resultat"]).startswith("Erreur") else ""
        arguments = f'arguments : {etape["arguments"]}' if etape["arguments"] else "sans argument"
        lignes.append(
            f'<li class="etape{erreur}"><span class="etape-num">{numero:02d}</span><div class="etape-corps">'
            f'<div class="etape-tete"><span class="etape-outil">{propre(etape["outil"])}</span>'
            f'<span class="etape-duree">{nombre(etape["duree_s"], 3)} s</span></div>'
            f'<p class="etape-args">{propre(arguments)}</p>'
            f'<p class="etape-resultat">{propre(etape["resultat"])}</p></div></li>')
    total = sum(e["duree_s"] for e in etapes)
    return (f'<ol class="journal">{"".join(lignes)}</ol>'
            f'<p class="journal-total">{len(etapes)} étape(s) · {nombre(total, 3)} s dans les outils</p>')


def panneau_mode(titre: str, couleur: str, duree_s: float, premiers: list[str], nb_outils: int) -> str:
    """Résumé d'un mode pour la comparaison : durée, trois premiers projets, nombre d'outils appelés."""
    return (f'<div class="duel-mode"><p class="duel-titre"><i style="background:{propre(couleur)}"></i>'
            f'{propre(titre)}</p><dl>'
            f'<dt>Durée</dt><dd>{nombre(duree_s, 1)} s</dd>'
            f'<dt>Trois premiers</dt><dd>{propre(", ".join(premiers))}</dd>'
            f'<dt>Outils appelés</dt><dd>{nb_outils}</dd></dl></div>')
