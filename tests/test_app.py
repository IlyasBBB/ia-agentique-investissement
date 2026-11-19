"""Lance l'application sans navigateur ni Ollama, avec l'outil de test de Streamlit."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")

if not (Path(APP).parent.parent / "data" / "processed" / "projets.csv").exists():
    pytest.skip("data/processed/projets.csv absent", allow_module_level=True)


def ouvrir() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)  # premier chargement des 2 000 textes : lent
    app.run()
    assert not app.exception
    return app


def test_page_s_affiche():
    app = ouvrir()
    assert app.title[0].value == "InvestMind"
    assert app.button(key="analyser").disabled  # pas de demande, pas d'analyse


def test_analyse_avec_le_pipeline_fixe():
    app = ouvrir()
    app.number_input(key="maximum").set_value(20)
    app.radio(key="mode").set_value("Pipeline fixe (sans LLM)")
    app.text_area(key="demande").set_value("Projets peu risqués")
    app.run()
    app.button(key="analyser").click().run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Recommandation", "Classement", "Vue 3D", "Journal des étapes", "Explication"]
    assert len(app.session_state["resultat"]["classement"]) == 20
    assert "Projets recommandés" in app.session_state["resultat"]["recommandation"]
    blocs = " ".join(m.value for m in app.markdown)
    assert blocs.count('class="carte') >= 3  # les trois cartes des projets recommandés
    assert 'class="journal"' in blocs  # le journal des étapes en ligne du temps


def test_les_textes_des_donnees_ne_sont_pas_lus_comme_du_html():
    import sys

    sys.path.insert(0, str(Path(APP).parent))
    import composants

    carte = composants.carte_projet({"rang": 1, "id_projet": "P1", "nom": "<script>alert(1)</script>",
                                     "secteur": "Eau", "pays": "Maroc", "score": 0.5,
                                     "rentabilite_predite": 4.2, "proba_risque": 0.3})
    assert "<script>" not in carte and "&lt;script&gt;" in carte
    journal = composants.journal_etapes([{"outil": "classer", "arguments": {}, "resultat": "<b>x</b>", "duree_s": 0.1}])
    assert "<b>x</b>" not in journal


def test_les_curseurs_reclassent_sans_relancer_l_analyse():
    app = ouvrir()
    app.number_input(key="maximum").set_value(20)
    app.radio(key="mode").set_value("Pipeline fixe (sans LLM)")
    app.text_area(key="demande").set_value("Projets peu risqués")
    app.run()
    app.button(key="analyser").click().run()
    etapes_avant = app.session_state["resultat"]["etapes"]
    app.slider(key="poids_rentabilite_predite").set_value(1.0).run()
    assert not app.exception
    assert app.session_state["resultat"]["etapes"] is etapes_avant  # l'analyse n'a pas été relancée
    assert "Cartes et classement suivent vos poids" in " ".join(m.value for m in app.markdown)


def test_graphique_shap_du_role_1(tmp_path):
    import sys

    import pandas as pd

    sys.path.insert(0, str(Path(APP).parent))
    import graphiques

    importances = pd.DataFrame({"variable": ["secteur", "duree_mois", "<b>x</b>"], "importance": [1.3, 1.0, 0.2]})
    figure = graphiques.figure_shap(importances)
    barres = figure.data[0]
    assert list(barres.y) == ["&lt;b&gt;x&lt;/b&gt;", "Durée", "Secteur"]  # de la plus faible à la plus forte
    assert list(barres.x) == [0.2, 1.0, 1.3]


def test_onglet_explication_montre_le_shap():
    app = ouvrir()
    app.number_input(key="maximum").set_value(20)
    app.radio(key="mode").set_value("Pipeline fixe (sans LLM)")
    app.text_area(key="demande").set_value("Projets peu risqués")
    app.run()
    app.button(key="analyser").click().run()
    assert not app.exception
    titres = [s.value for s in app.subheader]
    assert "Ce qui pèse sur la rentabilité prédite" in titres
