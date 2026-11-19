"""Graphiques Plotly de l'application, tous dessinés avec le même modèle « papier saumon ».

Trois couleurs de marque, distinctes sur le papier, y compris pour une personne
daltonienne : bleu (cas normal, agent), ocre (projets recommandés, pipeline fixe),
bordeaux (dossiers anormaux). Une forme ou des hachures doublent toujours la couleur.
"""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go

LIBELLES = {
    "rentabilite_predite": "Rentabilité prédite",
    "proba_risque": "Probabilité de risque élevé",
    "cout_total": "Coût total",
    "duree_mois": "Durée",
    "impact_env_social": "Impact environnemental et social",
    "alignement_strategique": "Alignement stratégique",
    "faisabilite": "Faisabilité",
}

# Variables d'entrée du modèle de rentabilité (rôle 1), pour le graphique SHAP.
LIBELLES_VARIABLES = {
    "secteur": "Secteur",
    "duree_mois": "Durée",
    "pays": "Pays",
    "type_financement": "Type de financement",
    "cout_total": "Coût total",
    "region": "Région",
    "annee_approbation": "Année d'approbation",
}
BLEU = "#1d4f8c"
OCRE = "#a8641c"
BORDEAUX = "#8e1b3a"
TEXTE = "#2b2724"
TEXTE_SECOND = "#6b625b"
GRILLE = "rgba(43, 39, 36, 0.12)"
FOND_BULLE = "#fff7f0"
POLICE_TEXTE = "Archivo, Segoe UI, sans-serif"
POLICE_CHIFFRES = POLICE_TEXTE  # Archivo a des chiffres de largeur égale : pas besoin de police à chasse fixe
# Du score le plus faible (clair) au plus fort (bleu foncé) : une seule teinte.
ECHELLE_SCORE = [[0.0, "#e8d3c1"], [0.5, "#7d9cc0"], [1.0, "#1d4f8c"]]
AXES_3D = ("rentabilite_predite", "proba_risque", "impact_env_social")


def modele_papier() -> go.layout.Template:
    """Le modèle commun : fond transparent, grille discrète, polices et couleurs du thème."""
    axe = {"gridcolor": GRILLE, "zerolinecolor": GRILLE, "linecolor": GRILLE,
           "tickfont": {"family": POLICE_CHIFFRES, "size": 12, "color": TEXTE_SECOND},
           "title": {"font": {"size": 13, "color": TEXTE_SECOND}}}
    return go.layout.Template(layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": POLICE_TEXTE, "color": TEXTE, "size": 13},
        "colorway": [BLEU, OCRE, BORDEAUX],
        "xaxis": axe,
        "yaxis": {**axe, "tickfont": {"family": POLICE_TEXTE, "size": 12, "color": TEXTE},
                  "automargin": True, "ticksuffix": "  "},
        "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "font": {"color": TEXTE}},
        "hoverlabel": {"bgcolor": FOND_BULLE, "bordercolor": "#dcc6b3",
                       "font": {"family": POLICE_TEXTE, "color": TEXTE, "size": 13}},
        "margin": {"l": 10, "r": 30, "t": 40, "b": 40},
        "separators": ", ",  # virgule décimale, espace pour les milliers
        "barcornerradius": 0,
        "bargap": 0.35,
    })


MODELE = modele_papier()


def _sans_balise(textes: pd.Series) -> pd.Series:
    """Neutralise les signes < > & : Plotly lit quelques balises HTML dans ses étiquettes."""
    return textes.astype(str).map(lambda texte: html.escape(texte, quote=False))


def _barres(nom: str, valeurs: list[float], etiquettes: list[str], couleur: str, hachures: str = "",
            decimales: int = 2) -> go.Bar:
    """Une série de barres horizontales, avec la valeur écrite au bout de chaque barre."""
    return go.Bar(
        name=nom, x=valeurs, y=etiquettes, orientation="h",
        # Hachures claires dessinées par-dessus la couleur : la série se reconnaît aussi sans la couleur.
        marker={"color": couleur, "pattern": {"shape": hachures, "fillmode": "overlay", "size": 7, "solidity": 0.35,
                                              "fgcolor": "rgba(250, 225, 207, 0.75)"}},
        text=[f"{v:.{decimales}f}".replace(".", ",") for v in valeurs],
        textposition="outside", textfont={"family": POLICE_CHIFFRES, "color": TEXTE, "size": 12},
        cliponaxis=False, hovertemplate="%{y} : %{x:." + str(decimales) + "f}<extra>" + nom + "</extra>")


def figure_poids(poids: dict[str, float]) -> go.Figure:
    """Barres horizontales des poids utilisés pour le classement, du plus fort au plus faible."""
    tries = sorted(poids.items(), key=lambda paire: paire[1])
    figure = go.Figure(_barres("Poids", [p for _, p in tries], [LIBELLES[c] for c, _ in tries], BLEU))
    figure.update_layout(template=MODELE, height=330, showlegend=False, xaxis_title="Poids (somme = 1)")
    return figure


def figure_shap(importances: pd.DataFrame) -> go.Figure:
    """Importance SHAP moyenne de chaque variable dans la rentabilité prédite, de la plus forte à la plus faible."""
    tries = importances.sort_values("importance")
    etiquettes = [LIBELLES_VARIABLES.get(v, v) for v in _sans_balise(tries["variable"])]
    figure = go.Figure(_barres("Importance SHAP", tries["importance"].tolist(), etiquettes, BLEU))
    figure.update_layout(template=MODELE, height=330, showlegend=False,
                         xaxis_title="Effet moyen sur la rentabilité prédite (points de %)")
    return figure


def figure_comparaison_poids(poids_agent: dict[str, float], poids_fixe: dict[str, float],
                             nom_agent: str, nom_fixe: str) -> go.Figure:
    """Poids choisis par l'agent et poids du pipeline fixe, côte à côte pour chaque critère."""
    criteres = list(reversed(LIBELLES))  # Plotly dessine de bas en haut
    etiquettes = [LIBELLES[c] for c in criteres]
    figure = go.Figure([
        _barres(nom_fixe, [poids_fixe[c] for c in criteres], etiquettes, OCRE, hachures="/"),
        _barres(nom_agent, [poids_agent[c] for c in criteres], etiquettes, BLEU),
    ])
    figure.update_layout(template=MODELE, height=460, barmode="group", bargroupgap=0.12,
                         xaxis_title="Poids (somme = 1)", legend_traceorder="reversed")
    return figure


def figure_meilleurs_scores(classement: pd.DataFrame, nombre: int = 10) -> go.Figure:
    """Les meilleurs scores en barres ; les dossiers anormaux sont en bordeaux hachuré."""
    haut = classement.head(nombre).iloc[::-1]  # inversé : le premier se retrouve en haut
    noms = haut["nom"].astype(str).map(lambda nom: nom if len(nom) <= 34 else nom[:33] + "…") if "nom" in haut else ""
    etiquettes = _sans_balise(haut["id_projet"].astype(str) + "  " + noms)
    anormal = haut["anomalie"].astype(bool) if "anomalie" in haut else pd.Series(False, index=haut.index)
    figure = go.Figure()
    for nom, masque, couleur, hachures in (("Dossier normal", ~anormal, BLEU, ""),
                                           ("Dossier anormal", anormal, BORDEAUX, "/")):
        if masque.any():
            figure.add_trace(_barres(nom, haut.loc[masque, "score"].tolist(), etiquettes[masque].tolist(),
                                     couleur, hachures, decimales=3))
    figure.update_layout(template=MODELE, height=440, xaxis_title="Score (de 0 à 1)", xaxis_range=[0, 1.08],
                         yaxis={"categoryorder": "array", "categoryarray": etiquettes.tolist()},
                         showlegend=bool(anormal.any()))
    return figure


def _points_3d(lignes: pd.DataFrame, nom: str, tailles: pd.Series, marqueur: dict,
               avec_texte: bool = False) -> go.Scatter3d:
    """Une famille de points du nuage 3D, avec la fiche du projet au survol."""
    fiche = pd.DataFrame({
        "id": _sans_balise(lignes["id_projet"]),
        "nom": _sans_balise(lignes["nom"].astype(str).str[:60]) if "nom" in lignes else "",
        "rang": lignes["rang"], "score": lignes["score"], "cout": lignes["cout_total"],
    })
    return go.Scatter3d(
        name=nom, x=lignes[AXES_3D[0]], y=lignes[AXES_3D[1]], z=lignes[AXES_3D[2]],
        mode="markers+text" if avec_texte else "markers",
        text=("n°" + lignes["rang"].astype(str)) if avec_texte else None,
        textposition="top center", textfont={"family": POLICE_CHIFFRES, "color": TEXTE, "size": 14},
        marker={"size": tailles.loc[lignes.index], "sizemode": "diameter", "opacity": 0.9, **marqueur},
        customdata=fiche[["id", "nom", "rang", "score", "cout"]],
        hovertemplate=("<b>%{customdata[0]}</b> · rang %{customdata[2]}<br>%{customdata[1]}<br>"
                       "Score : %{customdata[3]:.3f}<br>Rentabilité prédite : %{x:.1f} %<br>"
                       "Risque élevé : %{y:.0%}<br>Impact : %{z}/5<br>"
                       "Coût : %{customdata[4]:,.0f} M$<extra>" + nom + "</extra>"))


def figure_nuage_3d(classement: pd.DataFrame, recommandes: list[str]) -> go.Figure | None:
    """Nuage 3D des projets : rentabilité prédite, probabilité de risque, impact.

    Couleur = score, taille = coût, croix bordeaux = dossiers anormaux, losanges
    ocre = projets recommandés. Rend None si une colonne nécessaire manque.
    """
    if classement.empty or any(c not in classement for c in (*AXES_3D, "score", "rang", "cout_total")):
        return None
    # Diamètre du point : racine du coût, plafonnée pour qu'un coût extrême n'écrase pas les autres.
    repere = max(classement["cout_total"].quantile(0.95), 1e-9)
    tailles = 5 + 17 * (classement["cout_total"] / repere).clip(upper=1) ** 0.5
    anormal = classement["anomalie"].astype(bool) if "anomalie" in classement else pd.Series(False, index=classement.index)
    retenu = classement["id_projet"].isin(recommandes)
    autres = classement[~anormal & ~retenu]

    figure = go.Figure()
    if not autres.empty:
        figure.add_trace(_points_3d(autres, "Projets", tailles, {
            "color": autres["score"], "colorscale": ECHELLE_SCORE, "cmin": 0, "cmax": 1, "symbol": "circle",
            "colorbar": {"title": {"text": "Score", "font": {"color": TEXTE_SECOND}}, "thickness": 12, "len": 0.6,
                         "outlinewidth": 0, "tickfont": {"family": POLICE_CHIFFRES, "color": TEXTE_SECOND}}}))
    if anormal.any():
        figure.add_trace(_points_3d(classement[anormal], "Dossiers anormaux", tailles * 0.8,
                                    {"color": BORDEAUX, "symbol": "x"}))
    if retenu.any():
        figure.add_trace(_points_3d(classement[retenu], "Projets recommandés", tailles.clip(lower=11),
                                    {"color": OCRE, "symbol": "diamond", "line": {"color": TEXTE, "width": 2}},
                                    avec_texte=True))
    axe = {"backgroundcolor": "rgba(0,0,0,0)", "gridcolor": "rgba(43, 39, 36, 0.14)", "showbackground": True,
           "zerolinecolor": "rgba(43, 39, 36, 0.3)", "color": TEXTE_SECOND,
           "tickfont": {"family": POLICE_CHIFFRES, "size": 11}}
    figure.update_layout(
        template=MODELE, height=640, margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={"xaxis": {**axe, "title": {"text": "Rentabilité prédite (%)"}},
               "yaxis": {**axe, "title": {"text": "Probabilité de risque élevé"}, "tickformat": ".0%"},
               "zaxis": {**axe, "title": {"text": "Impact env. et social (/5)"}, "dtick": 1},
               "domain": {"x": [0, 1], "y": [0, 1]},
               "camera": {"eye": {"x": 1.35, "y": -1.5, "z": 0.55}, "center": {"x": 0, "y": 0, "z": -0.12}},
               "aspectmode": "cube"})
    return figure
