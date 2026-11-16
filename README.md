# IA agentique pour la décision d'investissement multicritère

Projet du cours « Introduction aux techniques d'apprentissage automatique », École Centrale Casablanca.

Un agent reçoit des dossiers de projets d'investissement, prédit leur rentabilité et leur risque, repère les dossiers anormaux, classe les projets selon plusieurs critères et rédige une recommandation justifiée.

README provisoire. La version finale décrira l'installation, l'exécution et la reproduction des résultats.

## Installation

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

## Structure

```
app/         application Streamlit
data/        raw (brut), processed (préparé), docs (documents texte des projets)
docs/        contrat commun, architecture, consignes, plan
models/      modèles entraînés (non versionnés)
notebooks/   analyses de chaque rôle
prompts/     prompts de l'agent et de l'extraction
reports/     tableaux et figures pour le rapport
src/         code des quatre modules
tests/       tests
```

## Équipe

| Rôle | Responsable | Module |
|---|---|---|
| 1. Apprentissage supervisé | à compléter | `src/supervise.py` |
| 2. Non supervisé et classement multicritère | à compléter | `src/mcda.py` |
| 3. NLP et RAG | à compléter | `src/nlp.py` |
| 4. Agent, application et dépôt | Ilyas Ben Alla | `src/agent.py`, `app/app.py` |
