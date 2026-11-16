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
    assert app.title[0].value == "Atlas Invest"
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
