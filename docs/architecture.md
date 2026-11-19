# Architecture de l'agent et de l'application

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
src/outils.py  (par défaut : vrais modules | USE_MOCKS=1 : src/mocks.py)
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
6. `chercher_passages(question, k)` : `retrieve`, pour citer un passage qui justifie la recommandation. La description de l'outil demande une question en anglais, la langue des documents ; si le LLM n'appelle pas l'outil, l'agent le fait lui-même (voir plus bas). L'agent transmet à `retrieve` les documents des projets analysés (paramètre `documents`, contrat du 18 novembre), puis garde par sécurité les `k` premiers passages qui viennent de ces projets. Avec une version de `retrieve` sans ce paramètre, il demande 50 passages sur tout le corpus et filtre de la même façon.
7. Rédiger la recommandation : les trois premiers projets, pourquoi, les alertes, les limites.

L'ordre n'est pas imposé en dur : le LLM choisit les outils. Le prompt système (`prompts/systeme.txt`) décrit la démarche attendue, et les tests vérifient le comportement avec un faux LLM.

## Fonctionnement de `src/agent.py`

- **Les outils agissent sur tout le portefeuille.** Le jeu de données compte 2 000 projets : le LLM ne peut pas appeler `predict` 2 000 fois. Chacun des cinq outils donnés au LLM traite donc tous les projets d'un coup, en passant par `src/outils.py`.
- **Les données restent dans Python.** La classe `Portefeuille` garde les projets, les notes, les prédictions, les anomalies et le classement. Le LLM ne reçoit qu'un résumé court de chaque résultat (quelques lignes de texte). Il ne voit jamais les 2 000 lignes et ne calcule rien : il décide et il rédige.
- **La boucle.** À chaque tour, on envoie la conversation au LLM (`ollama.Client.chat`, température 0, réflexion désactivée). S'il demande un outil, Python l'exécute, ajoute le résumé à la conversation et recommence. S'il répond par du texte, c'est la recommandation finale. Dix tours au maximum.
- **Le journal.** Chaque appel d'outil est noté : outil, arguments, résumé du résultat, durée. Une erreur d'outil (outil inconnu, mauvais argument, module en panne) est notée dans le journal et renvoyée au LLM, sans arrêter l'agent.
- **Les poids.** Ceux que propose le LLM sont nettoyés et ramenés à une somme de 1. Si l'utilisateur a fixé les poids (curseurs de l'application), les siens passent avant ceux du LLM.
- **Les garde-fous.** `classer` lance lui-même les étapes oubliées (notes, prédictions, anomalies). Les dossiers anormaux restent dans le classement avec une colonne `anomalie`, mais ils sont retirés des trois projets à recommander. Un projet sans document reçoit la note neutre 3. Seuls les passages des projets analysés sont cités.
- **La recherche de passages d'office.** Une fois le classement fait, si le LLM n'a pas demandé `chercher_passages`, l'agent le lance lui-même avec les noms des trois projets recommandés comme question (en anglais, comme les documents), et ajoute les passages trouvés au dernier résultat d'outil, avant la rédaction. Le journal le marque (`lance_par : agent`).
- **Le contrôle de la citation.** Un passage cité entre guillemets doit figurer dans un passage rendu par `chercher_passages`. Sinon, le contrôle le signale et donne le vrai passage.
- **La relance.** Si le LLM rend une réponse vide (ni texte ni outil), l'agent lui envoie une fois le message de `prompts/relance.txt` et note la relance dans le journal. Une seconde réponse vide déclenche le secours.
- **Le secours.** Si Ollama ne répond pas, si le plafond de tours est atteint, ou si le LLM répond sans avoir classé les projets, l'agent termine sans LLM : il complète le classement et rédige la recommandation avec un modèle de texte, en indiquant la raison. Quand le LLM a répondu par un texte sans rien classer (demande hors sujet, par exemple), ce texte n'est pas repris dans la recommandation, mais le journal le garde.

## Modèle de référence : `pipeline_fixe`

`pipeline_fixe(demande, projets, documents, poids)` appelle les cinq outils toujours dans le même ordre, sans LLM, et rédige la recommandation avec un modèle de texte. Il a deux usages :

- **comparaison** exigée par la consigne : on compare l'agent (le LLM choisit les outils et les poids) à ce modèle simple (ordre fixe, poids par défaut ou imposés) ;
- **secours** pour la démonstration si Ollama ne tourne pas.

Avec les quatre modules de l'équipe, il analyse 500 projets en 3 à 5 secondes (moins d'une seconde une fois les modèles chargés).

## Décisions prises

| Sujet | Décision | Raison |
|---|---|---|
| LLM | Ollama en local | Gratuit, choix de l'équipe |
| Modèle | `qwen2.5:3b` par défaut (variable `OLLAMA_MODEL`) | Gère l'appel d'outils sans phase de réflexion : environ deux minutes par analyse avec les vrais modules, sur un PC portable sans carte graphique. `qwen3:4b` a été écarté, trop lent (voir les essais ci-dessous) |
| Contrôle de la recommandation | `controler_recommandation` ajoute au texte du LLM les projets à recommander et les dossiers anormaux qu'il a oubliés | Un modèle de 3 milliards de paramètres oublie des faits ; ce qui peut se vérifier par le code ne doit pas dépendre de lui |
| Outils du LLM | Cinq outils qui traitent tout le portefeuille | Un appel par projet serait impossible avec 2 000 projets |
| Boucle de l'agent | Boucle Python maison avec l'appel d'outils du paquet `ollama`, retenue pour la version finale | Une trentaine de lignes, faciles à expliquer. Un cadriciel comme LangGraph n'apporterait rien ici |
| Modèle de référence | `pipeline_fixe`, sans LLM | Comparaison demandée par la consigne, et secours si Ollama est arrêté |
| Interface | Streamlit | Cité dans les consignes, rapide à écrire |
| Branchement des modules | `src/outils.py` avec `USE_MOCKS`, ou `MODULES_REELS` module par module (par exemple `MODULES_REELS=nlp`) | Permet d'avancer sans attendre les autres rôles, et d'essayer chaque module dès sa livraison |
| Déploiement | Démonstration locale filmée | Ollama ne tourne pas sur l'hébergement Streamlit gratuit |

## Décisions prises en cours de route

- 18 nov. : `chercher_passages` presque jamais demandé par `qwen2.5:3b`. La recherche est lancée d'office après `classer` (voir « La recherche de passages d'office »).
- 16 et 19 nov. : le prompt système est gardé dans sa formulation d'origine. Deux reformulations essayées le 16 et une modification du 18 ont déplacé les erreurs au lieu de les supprimer (`reports/role4/tests_agent.md`, erreurs E3 et E8).

## Évaluation sur cinq scénarios (16 et 19 novembre)

Cinq demandes, 30 projets en cours, deux passages, agent et `pipeline_fixe`, avec les fausses fonctions le 16 novembre puis avec les quatre vrais modules le 19 : protocole, résultats et analyse des erreurs dans `reports/role4/tests_agent.md`. Les scénarios et les vérifications automatiques sont dans `src/scenarios.py`, l'évaluation se relance avec `python scripts/evaluer_agent.py --passages 2`.

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

Conclusion : l'agent n'est pas utilisable en démonstration avec ce modèle sur ce PC. `qwen2.5:3b`, sans phase de réflexion, a été mesuré de la même façon et retenu (section précédente). Le délai de 600 secondes par réponse (`OLLAMA_TIMEOUT`) reste en place : au-delà, l'agent bascule sur le secours sans LLM.

## Application

L'application ne calcule rien : elle charge les projets, appelle `run_agent` ou `pipeline_fixe` et affiche ce qu'ils rendent.

Trois zones :

- **barre latérale** (bleu nuit) : source des données (jeu Banque mondiale ou fichiers importés), filtres, mode (agent ou pipeline fixe), comparaison, état d'Ollama ;
- **haut de page** : manchette (nom du produit, date, rappel que rentabilité et risque sont simulés), bandeau de chiffres du portefeuille retenu, puis deux colonnes : à gauche la demande, ses trois exemples, les poids et le bouton « Analyser » ; à droite le relief 3D du portefeuille et la fiche du projet cliqué ;
- **résultats**, en onglets : Recommandation (cartes des trois projets, alerte sur les dossiers anormaux, texte rédigé), Classement (dix meilleurs scores et tableau complet), Vue 3D (nuage Plotly), Journal des étapes, Explication (graphique SHAP du modèle de rentabilité, poids utilisés), Comparaison (agent face au pipeline fixe).

### Fichiers

| Fichier | Rôle |
|---|---|
| `app/app.py` | La page : chargement, barre latérale, demande, poids en direct, relief, onglets |
| `app/composants.py` | Blocs écrits en HTML : bandeau, cartes, fiche du projet, alerte, journal, panneaux de comparaison |
| `app/graphiques.py` | Graphiques Plotly et leur modèle commun « papier saumon », dont le nuage 3D |
| `app/style.css` | Habillage : manchette, cartes, journal, animations. Lu par `app.py` et injecté avec `st.markdown` |
| `app/scene3d/` | Relief 3D : `index.html`, `scene.js` (code de la scène), Three.js 0.160 copié dans le dossier (`three.module.min.js`, `OrbitControls.js`, licence MIT) |
| `app/static/polices/` | Polices Archivo et Source Serif 4 (licence OFL), servies par Streamlit |
| `.streamlit/config.toml` | Thème Streamlit (couleurs, polices, barre latérale) et service des fichiers de `app/static` |

### Thème

Direction choisie le 17 novembre : la page d'un quotidien financier sur papier saumon, pour quitter le look « tableau de bord généré par IA » (fond noir, néon cyan) de la première version.

- Couleurs : papier saumon, encre brun-noir, bleu profond pour l'action et les projets recommandés, bordeaux pour les alertes et les dossiers anormaux, sarcelle pour une rentabilité positive. Barre latérale bleu nuit.
- Polices : Source Serif 4 pour les titres, Archivo pour le texte et les chiffres (chiffres de largeur égale, qui s'alignent dans les tableaux). Elles sont copiées dans le dépôt : plus aucun appel à Google Fonts, l'application ne contacte aucun service externe.
- Filets fins à la place des cartes à ombre, rayons de 2 px. Les animations sont courtes et coupées si le système demande de réduire les mouvements.

Les couleurs des graphiques Plotly restent distinctes sur le papier, y compris en cas de daltonisme : bleu, ocre, bordeaux. Une forme ou des hachures doublent toujours la couleur.

### Fluidité

Streamlit relance tout le script à chaque clic. Pour éviter cet effet de page qui se recharge :

- la demande, les poids, le relief et les résultats sont dans un **fragment** (`@st.fragment`, fonction `atelier`) : un clic dans ce bloc ne relance que lui ;
- **poids en direct** : après une analyse, les curseurs partent des poids utilisés. Les bouger refait le classement avec `rank` sur les sept critères déjà calculés (fonction `reclasser`, via `src/outils.py`), sans relancer l'agent. Les cartes, le classement et le relief suivent ; la recommandation rédigée, le journal et la comparaison restent ceux de l'analyse, et un message le dit ;
- le relief 3D reste le même d'un clic à l'autre : seules ses données changent, et les colonnes glissent vers leur nouvelle place.
- avec les vrais modules (le défaut), les modèles de texte de `extract` et `retrieve` sont chargés dès l'ouverture de l'application (`prechauffer_modeles`, mis en cache) : la première analyse ne paie pas leur chargement, une quinzaine de secondes.

Ce qui reste lent : l'analyse par l'agent (environ deux minutes, le LLM tourne sur le processeur).

### Relief 3D

Composant Streamlit maison (`components.declare_component`, dossier `app/scene3d`), écrit avec Three.js. Une colonne par projet :

- disposition en spirale par rang : le n° 1 au centre, les suivants de plus en plus loin, si bien que les meilleurs projets forment le sommet ;
- hauteur : le score, ramené entre le plus faible et le plus fort du portefeuille et accentué vers le haut pour que le sommet se détache ;
- couleur : encre pour les projets ordinaires, bleu pour les trois recommandés (avec leur numéro), bordeaux pour les dossiers anormaux ;
- avant l'analyse, les colonnes sont au repos, presque plates ; après, elles montent en vague du centre vers le bord ;
- survol : bulle avec le nom, le secteur, le pays, le rang et le score ; clic : l'identifiant revient à Python, qui affiche la fiche du projet sous le relief ;
- la caméra tourne lentement et s'arrête dès qu'on la manipule. Sans WebGL, un message remplace la scène ; si le système demande moins de mouvement, ni rotation ni animation.

Les noms de projets passent dans la scène par `textContent`, jamais par `innerHTML` : ils ne peuvent pas être lus comme du HTML.

### Vue 3D (onglet)

Nuage de points Plotly (`scatter3d`) : rentabilité prédite, probabilité de risque élevé, impact environnemental et social ; couleur = score (plus foncé = meilleur), taille = coût total, losanges ocre = projets recommandés, croix bordeaux = dossiers anormaux.

### Sécurité de l'affichage

Le texte du LLM est affiché avec `st.text`, donc jamais lu comme du Markdown ou du HTML. Dans les blocs HTML, tout ce qui vient des données ou du LLM (noms de projets, CSV importé, arguments et résultats d'outils) passe par `composants.propre`, qui neutralise les balises. Un test le vérifie.
