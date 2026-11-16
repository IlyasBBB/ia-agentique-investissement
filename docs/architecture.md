# Architecture (rôle 4)

## Vue d'ensemble

```
Utilisateur
   |  demande, projets, documents, poids
   v
app/app.py (Streamlit)
   |
   v
src/agent.py  -- run_agent() -->  LLM local (Ollama)
   |                                  |  choisit l'outil suivant
   v                                  v
src/outils.py  (USE_MOCKS=1 : src/mocks.py | USE_MOCKS=0 : vrais modules)
   |-- predict            src/supervise.py   (rôle 1)
   |-- rank               src/mcda.py        (rôle 2)
   |-- detect_anomalies   src/mcda.py        (rôle 2)
   |-- extract            src/nlp.py         (rôle 3)
   |-- retrieve           src/nlp.py         (rôle 3)
```

## Démarche de l'agent

1. Lire la demande et en tirer les priorités (par exemple « faible risque d'abord »), ce qui fixe ou ajuste les poids.
2. Pour chaque projet avec un document : `extract` pour obtenir les critères qualitatifs.
3. Pour chaque projet : `predict` pour la rentabilité et le risque.
4. `detect_anomalies` sur l'ensemble.
5. `rank` avec la matrice des critères et les poids.
6. `retrieve` au besoin pour citer un passage qui justifie la recommandation.
7. Rédiger la recommandation : les trois premiers projets, pourquoi, les alertes, les limites.

L'ordre n'est pas imposé en dur : le LLM choisit les outils. Le prompt système décrit la démarche attendue, et les tests vérifient que l'ordre suivi reste logique.

## Décisions prises

| Sujet | Décision | Raison |
|---|---|---|
| LLM | Ollama en local | Gratuit, choix de l'équipe |
| Modèle | À choisir parmi ceux qui gèrent l'appel d'outils dans Ollama | À tester sur le PC d'Ilyas selon la mémoire disponible |
| Boucle de l'agent | Boucle Python simple avec l'appel d'outils du paquet `ollama` | Plus facile à comprendre et à expliquer qu'un cadriciel. Passage à LangGraph possible ensuite |
| Interface | Streamlit | Cité dans les consignes, rapide à écrire |
| Branchement des modules | `src/outils.py` avec `USE_MOCKS` | Permet d'avancer sans attendre les autres rôles |
| Déploiement | Démonstration locale filmée | Ollama ne tourne pas sur l'hébergement Streamlit gratuit |

## Décisions ouvertes

- Nom exact du modèle Ollama.
- LangGraph ou boucle maison pour la version finale.
- Jeu de données (voir `docs/donnees.md`).

## Application

Trois zones : chargement (CSV des projets, documents texte), réglages (curseurs des poids, demande en langage naturel), résultats (classement, alertes, recommandation, journal des étapes, graphique du classement).
