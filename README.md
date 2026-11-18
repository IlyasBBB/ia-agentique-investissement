# IA agentique pour la décision d'investissement multicritère

Projet du cours « Introduction aux techniques d'apprentissage automatique », École Centrale Casablanca, novembre 2026.

On donne à un agent une liste de projets d'investissement (chiffres et documents texte) et une demande en français, par exemple « Je veux des projets peu risqués avec un fort impact environnemental ». L'agent prédit la rentabilité et le risque de chaque projet, note ses documents, écarte les dossiers anormaux, classe les projets avec des poids tirés de la demande, puis rédige une recommandation qui cite ses chiffres et un passage des documents. Tout tourne en local, y compris le modèle de langage.

## Ce que fait chaque partie

| Rôle | Module | Fonctions | Méthodes |
|---|---|---|---|
| 1. Apprentissage supervisé | `src/supervise.py` | `predict(projet)` | Régression de la rentabilité, classification du risque |
| 2. Non supervisé et multicritère | `src/mcda.py` | `rank(matrice, poids)`, `detect_anomalies(df)` | Classement multicritère pondéré, détection d'anomalies |
| 3. NLP et RAG | `src/nlp.py` | `extract(document)`, `retrieve(question, k)` | Notes qualitatives tirées du texte, recherche de passages par embeddings |
| 4. Agent et application | `src/agent.py`, `app/` | `run_agent(demande, projets)` | Agent piloté par un LLM local (Ollama) avec appel d'outils, application Streamlit |

Le format des entrées et des sorties de chaque fonction est fixé dans `docs/contrat.md` et vérifié par `src/contrat.py`.

L'agent n'appelle jamais les modules directement. Il passe par `src/outils.py`, qui branche soit de fausses fonctions (`src/mocks.py`), soit les vrais modules. Chaque rôle a donc pu avancer sans attendre les autres.

## Installation

Il faut Python 3.11 ou plus récent, et [Ollama](https://ollama.com) pour l'agent. Les commandes ci-dessous sont pour Windows (PowerShell).

```
git clone https://github.com/IlyasBBB/ia-agentique-investissement.git
cd ia-agentique-investissement
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

Sous macOS ou Linux, on active l'environnement avec `source .venv/bin/activate`.

`sentence-transformers` installe PyTorch : le téléchargement est gros, mais un processeur suffit, aucune carte graphique n'est nécessaire.

Au premier appel de `retrieve`, le modèle d'embeddings `all-MiniLM-L6-v2` (80 Mo) est téléchargé, puis les vecteurs des 8 293 passages sont calculés (environ 4 minutes) et enregistrés dans `models/nlp/`. Ensuite tout fonctionne sans Internet.

## Lancer l'application

```
ollama serve                 # dans un premier terminal
streamlit run app/app.py     # dans un second
```

L'application s'ouvre dans le navigateur. On y choisit les projets (jeu de la Banque mondiale ou fichiers importés), on écrit une demande ou on prend un des exemples, puis on clique sur « Analyser ». Elle affiche :

- les trois projets recommandés et l'alerte sur les dossiers anormaux ;
- un relief 3D du portefeuille, une colonne par projet, cliquable ;
- le classement complet ;
- le journal des étapes de l'agent ;
- la comparaison avec le modèle de référence.

Après une analyse, on peut bouger les curseurs des poids : le classement est refait sans relancer l'agent.

Une analyse par l'agent prend environ une minute sur un PC portable, car le LLM tourne sur le processeur. Le mode « pipeline fixe » (les outils dans un ordre fixe, sans LLM) répond en quelques secondes et marche même si Ollama est arrêté.

### Variables d'environnement

| Variable | Valeur par défaut | Effet |
|---|---|---|
| `USE_MOCKS` | `1` | `1` : fausses fonctions ; `0` : vrais modules des rôles 1 à 3 |
| `MODULES_REELS` | vide | Liste des modules à prendre en vrai, les autres restant factices, par exemple `nlp` ou `nlp,supervise`. Passe avant `USE_MOCKS` |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modèle utilisé par l'agent |
| `OLLAMA_HOST` | adresse locale d'Ollama | Adresse du serveur Ollama |
| `OLLAMA_TIMEOUT` | `600` | Attente maximale d'une réponse du LLM, en secondes. Au-delà, l'agent termine sans LLM |
| `NLP_RETRIEVE` | `embeddings` | `tfidf` force le moteur de recherche simple |

Exemple, pour lancer l'application avec les vrais modules :

```
$env:USE_MOCKS="0"
streamlit run app/app.py
```

## Tests

```
python -m pytest            # tous les tests, sans Ollama
python -m pytest -m llm     # en plus, les tests qui appellent le vrai LLM (lents, Ollama doit tourner)
```

Les tests de l'agent remplacent le LLM par un faux client qui rejoue des réponses écrites à l'avance : ils donnent toujours le même résultat et n'ont pas besoin d'Ollama.

## Reproduire les résultats

### Données

Le jeu préparé (`data/processed/projets.csv` et `data/docs/`) est dans le dépôt. Pour le refaire depuis la source :

```
python scripts/telecharger_donnees.py     # télécharge 28 245 projets de la Banque mondiale (52 Mo, hors Git)
python scripts/preparer_donnees.py        # filtre, tire 2 000 projets, simule les cibles, écrit les documents
```

La préparation utilise la graine 42 : à partir du même fichier brut, elle redonne exactement les mêmes fichiers. Le détail est dans `data/README.md`.

### Recherche de passages (rôle 3)

```
python scripts/evaluer_retrieve_nlp.py    # TF-IDF, embeddings et hybride sur 40 requêtes jugées
```

Résultats dans `reports/role3/resultats_retrieve.csv`, analyse dans `notebooks/role3/03_retrieve.ipynb`. Sur les 25 requêtes de test, la précision@3 est de 0,55 pour TF-IDF et de 0,75 pour les embeddings, le moteur retenu.

### Agent (rôle 4)

```
ollama serve
python scripts/evaluer_agent.py --passages 2    # 5 scénarios, agent et pipeline fixe, environ 11 minutes
```

Résultats dans `reports/resultats_scenarios.csv` et `reports/sorties_scenarios.json`, analyse des erreurs dans `reports/tests_agent.md`.

## Données

Projets réels de la Banque mondiale, « Projects & Operations » (https://projects.worldbank.org), licence CC BY 4.0, téléchargés le 16 novembre 2026. On a gardé 2 000 projets : 1 500 terminés et 500 en cours.

La Banque mondiale ne publie ni la rentabilité ni le niveau de risque de ses projets. Les colonnes `rentabilite` et `risque` sont donc **simulées** par une formule décrite dans `data/README.md`, et 40 anomalies ont été ajoutées pour tester leur détection. Le reste (nom, pays, secteur, coût, dates, textes) est réel. Les recommandations de l'application ne disent donc rien de la vraie valeur de ces projets.

## Organisation du dépôt

```
app/          application Streamlit (page, blocs HTML, graphiques, relief 3D, thème)
data/         processed/ (projets.csv), docs/ (un document texte par projet), raw/ (hors Git)
docs/         contrat commun, architecture de l'agent, consignes, plans de travail
models/       modèles entraînés et index de recherche (hors Git)
notebooks/    analyses de chaque rôle
prompts/      prompts de l'agent
reports/      résultats, figures, évaluations et brouillons du rapport
scripts/      téléchargement et préparation des données, évaluations
src/          les modules des quatre rôles, le contrat, les fausses fonctions
tests/        tests automatiques
```

## Limites

- Rentabilité et risque sont simulés : le projet montre une méthode, pas un outil de décision réel.
- Le LLM est petit (3 milliards de paramètres) pour tourner sur un PC portable. Il oublie parfois un outil ou une alerte : le code vérifie et complète sa recommandation.
- Ollama ne tourne pas sur l'hébergement gratuit de Streamlit. L'application se lance donc en local, et la démonstration est filmée.

## Équipe

| Rôle | Responsable |
|---|---|
| 1. Apprentissage supervisé | Salma Akazi |
| 2. Non supervisé et classement multicritère | Zineb Baghzali |
| 3. NLP et RAG | Ayoub Asseffar |
| 4. Agent, application et dépôt | Ilyas Ben Alla |

## Licences

Code : projet étudiant. Données : Banque mondiale, CC BY 4.0. Three.js : licence MIT. Polices Archivo et Source Serif 4 : licence SIL Open Font License (`app/static/polices/LICENCES.md`).
