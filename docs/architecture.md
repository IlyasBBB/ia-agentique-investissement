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

1. Lire la demande et en tirer les priorités (par exemple « faible risque d'abord »), ce qui fixe les poids.
2. `extraire_criteres` : `extract` sur le document de chaque projet, pour les trois notes qualitatives.
3. `predire` : `predict` sur chaque projet, pour la rentabilité et le risque.
4. `detecter_anomalies` : `detect_anomalies` sur l'ensemble.
5. `classer(poids)` : construit la matrice des sept critères et appelle `rank`.
6. `chercher_passages(question, k)` : `retrieve`, pour citer un passage qui justifie la recommandation.
7. Rédiger la recommandation : les trois premiers projets, pourquoi, les alertes, les limites.

L'ordre n'est pas imposé en dur : le LLM choisit les outils. Le prompt système (`prompts/systeme.txt`) décrit la démarche attendue, et les tests vérifient le comportement avec un faux LLM.

## Fonctionnement de `src/agent.py`

- **Les outils agissent sur tout le portefeuille.** Le jeu de données compte 2 000 projets : le LLM ne peut pas appeler `predict` 2 000 fois. Chacun des cinq outils donnés au LLM traite donc tous les projets d'un coup, en passant par `src/outils.py`.
- **Les données restent dans Python.** La classe `Portefeuille` garde les projets, les notes, les prédictions, les anomalies et le classement. Le LLM ne reçoit qu'un résumé court de chaque résultat (quelques lignes de texte). Il ne voit jamais les 2 000 lignes et ne calcule rien : il décide et il rédige.
- **La boucle.** À chaque tour, on envoie la conversation au LLM (`ollama.Client.chat`, température 0, réflexion désactivée). S'il demande un outil, Python l'exécute, ajoute le résumé à la conversation et recommence. S'il répond par du texte, c'est la recommandation finale. Dix tours au maximum.
- **Le journal.** Chaque appel d'outil est noté : outil, arguments, résumé du résultat, durée. Une erreur d'outil (outil inconnu, mauvais argument, module en panne) est notée dans le journal et renvoyée au LLM, sans arrêter l'agent.
- **Les poids.** Ceux que propose le LLM sont nettoyés et ramenés à une somme de 1. Si l'utilisateur a fixé les poids (curseurs de l'application), les siens passent avant ceux du LLM.
- **Les garde-fous.** `classer` lance lui-même les étapes oubliées (notes, prédictions, anomalies). Les dossiers anormaux restent dans le classement avec une colonne `anomalie`, mais ils sont retirés des trois projets à recommander. Un projet sans document reçoit la note neutre 3. Seuls les passages des projets analysés sont cités.
- **Le secours.** Si Ollama ne répond pas, si le plafond de tours est atteint, ou si le LLM répond sans avoir classé les projets, l'agent termine sans LLM : il complète le classement et rédige la recommandation avec un modèle de texte, en indiquant la raison.

## Modèle de référence : `pipeline_fixe`

`pipeline_fixe(demande, projets, documents, poids)` appelle les cinq outils toujours dans le même ordre, sans LLM, et rédige la recommandation avec un modèle de texte. Il a deux usages :

- **comparaison** exigée par la consigne : on compare l'agent (le LLM choisit les outils et les poids) à ce modèle simple (ordre fixe, poids par défaut ou imposés) ;
- **secours** pour la démonstration si Ollama ne tourne pas.

Sur les 2 000 projets avec les fausses fonctions, il répond en moins d'une seconde (hors lecture des 2 000 fichiers texte).

## Décisions prises

| Sujet | Décision | Raison |
|---|---|---|
| LLM | Ollama en local | Gratuit, choix de l'équipe |
| Modèle | `qwen2.5:3b` par défaut (variable `OLLAMA_MODEL`) | Gère l'appel d'outils sans phase de réflexion : environ 1 minute par analyse sur le PC d'Ilyas. `qwen3:4b` a été écarté, trop lent (voir les essais ci-dessous) |
| Contrôle de la recommandation | `controler_recommandation` ajoute au texte du LLM les projets à recommander et les dossiers anormaux qu'il a oubliés | Un modèle de 3 milliards de paramètres oublie des faits ; ce qui peut se vérifier par le code ne doit pas dépendre de lui |
| Outils du LLM | Cinq outils qui traitent tout le portefeuille | Un appel par projet serait impossible avec 2 000 projets |
| Boucle de l'agent | Boucle Python maison avec l'appel d'outils du paquet `ollama`, retenue pour la version finale | Une trentaine de lignes, faciles à expliquer. Un cadriciel comme LangGraph n'apporterait rien ici |
| Modèle de référence | `pipeline_fixe`, sans LLM | Comparaison demandée par la consigne, et secours si Ollama est arrêté |
| Interface | Streamlit | Cité dans les consignes, rapide à écrire |
| Branchement des modules | `src/outils.py` avec `USE_MOCKS` | Permet d'avancer sans attendre les autres rôles |
| Déploiement | Démonstration locale filmée | Ollama ne tourne pas sur l'hébergement Streamlit gratuit |

## Décisions ouvertes

- Aucune pour l'agent à ce stade. La qualité de `qwen2.5:3b` sera mesurée sur les 5 scénarios de la tâche 5.

## Essai du 16 novembre avec `qwen2.5:3b` (modèle retenu)

Même PC, même demande, mêmes cinq projets d'exemple.

- Analyse complète en 58 secondes : un premier tour de 19 s où le modèle demande les cinq outils d'un coup, puis un tour de 38 s pour rédiger.
- Poids bien tirés de la demande : `proba_risque` 0,5 et `impact_env_social` 0,5.
- Le modèle ajoute un argument inventé (`documents`) aux outils qui n'en prennent pas. Corrigé : l'agent ne transmet à un outil que les arguments qu'il accepte, et le journal garde ce que le LLM a envoyé.
- Le texte rédigé cite deux projets au lieu de trois, écrit « aucun dossier anormal » alors que P005 a été écarté, et cite un passage d'un projet non recommandé. D'où le contrôle automatique, qui ajoute en fin de texte le projet et l'alerte oubliés. La phrase fausse du LLM reste visible au-dessus : c'est une limite à montrer dans le rapport.

## Essai du 16 novembre avec `qwen3:4b`

PC : Ryzen 5 7520U, 16 Go de mémoire, calcul sur processeur. Demande : « Je veux des projets peu risqués avec un fort impact environnemental », cinq projets d'exemple, fausses fonctions.

- Le modèle écrit 2 à 8 mots par seconde. Une petite réponse de 60 jetons prend 8 secondes.
- Cette version de `qwen3:4b` réfléchit toujours avant de répondre. `think=False` ne coupe pas la réflexion : elle est seulement versée dans le texte de la réponse, en anglais. La mention `/no_think` dans le prompt n'a pas d'effet non plus.
- Avec le prompt système et les cinq outils, un seul tour a demandé environ 4 600 jetons de réflexion : 10 minutes par tour lors d'un premier essai, 33 minutes lors d'une mesure isolée. L'essai complet a été arrêté après deux tours (20 minutes).
- Les outils choisis étaient les bons (`extraire_criteres`, `predire`, `detecter_anomalies`, `classer`), mais `extraire_criteres` a été appelé deux fois et `classer` sans poids. Le contexte par défaut d'Ollama (4 096 jetons) était dépassé par la réflexion ; il est maintenant fixé à 8 192.

Conclusion : l'agent n'est pas utilisable en démonstration avec ce modèle sur ce PC. Pistes : un modèle sans réflexion de taille voisine (par exemple `qwen2.5:3b` ou `llama3.2:3b`), à mesurer de la même façon. En attendant, un délai de 600 secondes par réponse (`OLLAMA_TIMEOUT`) fait basculer l'agent sur le secours sans LLM.

## Application

Trois zones : chargement (CSV des projets, documents texte), réglages (curseurs des poids, demande en langage naturel), résultats (classement, alertes, recommandation, journal des étapes, graphique du classement).
