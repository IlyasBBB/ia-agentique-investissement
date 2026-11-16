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
- **La relance.** Si le LLM rend une réponse vide (ni texte ni outil), l'agent lui envoie une fois le message de `prompts/relance.txt` et note la relance dans le journal. Une seconde réponse vide déclenche le secours.
- **Le secours.** Si Ollama ne répond pas, si le plafond de tours est atteint, ou si le LLM répond sans avoir classé les projets, l'agent termine sans LLM : il complète le classement et rédige la recommandation avec un modèle de texte, en indiquant la raison. Quand le LLM a répondu par un texte sans rien classer (demande hors sujet, par exemple), ce texte n'est pas repris dans la recommandation, mais le journal le garde.

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

- `chercher_passages` n'est presque jamais demandé par `qwen2.5:3b` (voir `reports/tests_agent.md`). À décider à l'intégration du vrai module `retrieve` : laisser ce choix au LLM, ou lancer la recherche d'office après `classer`.
- Le prompt système est gardé tel quel après deux essais de reformulation qui ont déplacé les erreurs au lieu de les supprimer (même rapport).

## Évaluation du 16 novembre sur cinq scénarios

Cinq demandes, 30 projets en cours, deux passages, agent et `pipeline_fixe` : protocole, résultats et analyse des erreurs dans `reports/tests_agent.md`. Les scénarios et les vérifications automatiques sont dans `src/scenarios.py`, l'évaluation se relance avec `python scripts/evaluer_agent.py --passages 2`.

Deux corrections de `src/agent.py` en sont sorties :

- relance après une réponse vide du LLM (avant : secours immédiat, avec les poids par défaut, alors que la demande donnait des priorités) ;
- réponse du LLM sans outil gardée dans le journal (avant : texte perdu, on ne savait pas pourquoi l'agent n'avait rien classé).

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

L'application ne calcule rien : elle charge les projets, appelle `run_agent` ou `pipeline_fixe` et affiche ce qu'ils rendent.

Trois zones :

- **barre latérale** : source des données (jeu Banque mondiale ou fichiers importés), filtres, mode (agent ou pipeline fixe), comparaison, état d'Ollama ;
- **haut de page** : nom du produit, bandeau d'indicateurs du portefeuille retenu (projets, coût total en milliards de dollars, secteurs, pays, documents), puis la demande, ses trois exemples et le choix des poids ;
- **résultats**, en onglets : Recommandation (cartes des trois projets, alerte sur les dossiers anormaux, texte rédigé), Classement (dix meilleurs scores et tableau complet), Vue 3D, Journal des étapes (ligne du temps), Explication (poids utilisés, graphique SHAP à venir), Comparaison (agent face au pipeline fixe).

### Fichiers

| Fichier | Rôle |
|---|---|
| `app/app.py` | La page : chargement, barre latérale, demande, appel de l'agent, onglets |
| `app/composants.py` | Blocs écrits en HTML : bandeau, cartes, alerte, journal, panneaux de comparaison |
| `app/graphiques.py` | Graphiques Plotly et leur modèle sombre commun, dont le nuage 3D |
| `app/style.css` | Habillage : fond, polices, cartes, journal, animations. Lu par `app.py` et injecté avec `st.markdown` |
| `.streamlit/config.toml` | Thème Streamlit : couleurs de base, polices, barre d'outils réduite |

### Thème

Thème sombre de terminal financier : fond presque noir bleuté avec une grille fine, cyan pour les actions et les repères, vert pour une rentabilité positive, rouge pour les alertes et les dossiers anormaux. Trois polices Google Fonts : Chakra Petch (titres), IBM Plex Sans (texte), IBM Plex Mono (chiffres, journal). Hors ligne, les polices du système prennent le relais (Segoe UI, Consolas) : l'application reste utilisable, seul l'aspect change. Les animations sont courtes et coupées si le système demande de réduire les mouvements.

Les couleurs des graphiques sont plus sourdes que celles de l'interface : cyan `#1298b4`, ambre `#c98500`, rose-rouge `#e0527a`. Elles ont été choisies pour rester distinctes sur fond sombre, y compris en cas de daltonisme. Le vert et le rouge ne sont jamais opposés dans un graphique, et une forme ou des hachures doublent toujours la couleur.

### Vue 3D

Nuage de points Plotly (`scatter3d`), qu'on peut tourner et zoomer. Un point par projet analysé :

- position : rentabilité prédite, probabilité de risque élevé, impact environnemental et social ;
- couleur : score (plus clair = meilleur) ;
- taille : coût total (racine du coût, plafonnée au 95e centile pour qu'un coût extrême n'écrase pas les autres) ;
- losanges ambre avec leur rang : les trois projets recommandés ;
- croix rouges : dossiers anormaux.

Plotly est déjà embarqué par Streamlit : aucun script n'est chargé depuis Internet pour la 3D.

### Sécurité de l'affichage

Le texte du LLM est affiché avec `st.text`, donc jamais lu comme du Markdown ou du HTML. Dans les blocs HTML, tout ce qui vient des données ou du LLM (noms de projets, CSV importé, arguments et résultats d'outils) passe par `composants.propre`, qui neutralise les balises. Un test le vérifie.
