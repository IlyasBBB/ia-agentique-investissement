"""Blocs HTML de l'application : bandeau d'indicateurs, cartes, alerte, journal.

Chaque fonction rend une chaîne HTML à afficher avec st.markdown. Tout texte qui
vient des données ou du LLM (nom de projet, résultat d'outil, CSV importé) passe
par `propre` : il est affiché tel quel et ne peut pas être lu comme du HTML.
Les classes CSS utilisées ici sont définies dans app/style.css.
"""
from __future__ import annotations

import html

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


def carte_projet(ligne: dict) -> str:
    """Carte d'un projet recommandé : rang, score en jauge, rentabilité, risque, secteur et pays."""
    score = float(ligne["score"])
    rentabilite = float(ligne["rentabilite_predite"])
    sens = "hausse" if rentabilite >= 0 else "baisse"
    rentabilite_texte = f"{rentabilite:+.1f}".replace(".", ",")  # le signe + double la couleur
    lieu = " · ".join(str(ligne[c]) for c in ("secteur", "pays") if ligne.get(c))
    premiere = " premiere" if ligne["rang"] == 1 else ""
    return (
        f'<article class="carte{premiere}">'
        f'<div class="carte-tete"><span class="carte-rang">N° {propre(ligne["rang"])}</span>'
        f'<span class="carte-id">{propre(ligne["id_projet"])}</span></div>'
        f'<p class="carte-nom">{propre(ligne.get("nom", ""))}</p>'
        f'<p class="carte-lieu">{propre(lieu)}</p>'
        f'<div class="jauge-ligne"><span>Score</span><b>{nombre(score, 3)}</b></div>'
        f'<div class="jauge" role="meter" aria-label="Score sur 1" aria-valuemin="0" aria-valuemax="1" '
        f'aria-valuenow="{score:.3f}"><span style="width:{score:.1%}"></span></div>'
        f'<div class="mesures">'
        f'<div><span>Rentabilité prédite</span><b class="{sens}">{rentabilite_texte}{ESPACE_FINE}%</b></div>'
        f'<div><span>Risque élevé</span><b>{float(ligne["proba_risque"]):.0%}</b></div>'
        f'</div></article>'
    )


def cartes_recommandes(recommandes: pd.DataFrame) -> str:
    """Les cartes des projets recommandés, côte à côte."""
    return f'<div class="cartes">{"".join(carte_projet(l) for l in recommandes.to_dict("records"))}</div>'


def alerte_anomalies(anormaux: list[str], limite: int = 15) -> str:
    """Encadré rouge : nombre de dossiers anormaux écartés et leurs identifiants."""
    suite = " …" if len(anormaux) > limite else ""
    if len(anormaux) == 1:
        titre = "1 dossier anormal écarté de la recommandation"
    else:
        titre = f"{len(anormaux)} dossiers anormaux écartés de la recommandation"
    return (f'<div class="alerte" role="alert">{ICONE_ALERTE}<div><strong>{propre(titre)}</strong>'
            f'<span>{propre(", ".join(anormaux[:limite]) + suite)}</span></div></div>')


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
