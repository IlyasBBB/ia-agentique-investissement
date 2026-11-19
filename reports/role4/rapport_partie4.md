# Partie 4 : l'agent, son architecture et son évaluation

Brouillon du 19 novembre 2026. Deux séries d'essais : le 16 novembre avec les fausses fonctions des rôles 1 à 3, le 19 novembre avec les quatre vrais modules (intégration). Détail dans `reports/role4/tests_agent.md`.

## 4.1 Fondements théoriques

### Les grands modèles de langage

Un grand modèle de langage (LLM) est un réseau de neurones qui prédit le mot suivant d'un texte. Plus précisément, il découpe le texte en jetons (des morceaux de mots) et calcule, pour chaque jeton possible, la probabilité qu'il vienne ensuite. Pour écrire une réponse, il choisit un jeton, l'ajoute au texte et recommence.

Ces modèles reposent sur l'architecture Transformer (Vaswani et al., 2017). Son mécanisme central, l'attention, permet à chaque jeton de tenir compte de tous les jetons précédents, quelle que soit leur distance. C'est ce qui permet au modèle de suivre une consigne donnée plusieurs paragraphes plus haut.

Un LLM est construit en trois étapes :

1. Pré-entraînement. Le modèle apprend à prédire le jeton suivant sur un très grand corpus de textes. Il n'y a pas d'étiquettes à fournir : le texte lui-même sert de réponse attendue. C'est de l'apprentissage auto-supervisé. Le modèle obtenu sait continuer un texte, mais pas encore répondre à une demande.
2. Ajustement supervisé (SFT). On l'entraîne ensuite sur des paires demande-réponse écrites par des humains. Il apprend le format d'un échange : une question, puis une réponse utile.
3. Apprentissage par renforcement à partir de préférences humaines (RLHF). Des annotateurs classent plusieurs réponses du modèle à la même demande. Un second modèle apprend à prédire ces préférences, puis sert de récompense pour ajuster le LLM par renforcement (Ouyang et al., 2022).

Deux réglages comptent pour notre usage. La **température** règle le hasard dans le choix du jeton suivant : à 0, le modèle prend toujours le jeton le plus probable, ce qui devrait rendre ses réponses reproductibles. La **fenêtre de contexte** est le nombre de jetons que le modèle lit à la fois : tout ce qui la dépasse est perdu.

### Le prompt

Le prompt est le texte donné au modèle avant la demande de l'utilisateur. Il fixe son rôle, sa démarche et le format de sa réponse. On ne réentraîne pas le modèle : on change seulement ce qu'il lit. C'est le moyen le moins coûteux d'orienter un LLM, mais aussi le plus fragile, comme le montre notre évaluation (section 4.5).

### L'appel d'outils et l'IA agentique

Un LLM seul ne peut ni calculer de façon fiable, ni lire nos données. L'appel d'outils contourne ce problème. On décrit au modèle une liste de fonctions (nom, rôle, arguments). Au lieu de répondre par du texte, il peut demander l'exécution d'une de ces fonctions. Le programme l'exécute, renvoie le résultat au modèle, et le modèle décide de la suite.

Un agent est un programme construit autour de cette boucle : le LLM choisit l'action suivante, le code l'exécute, le LLM lit le résultat, et ainsi de suite jusqu'à ce qu'il juge avoir fini. Cette alternance entre raisonnement et action est décrite par Yao et al. (2022) sous le nom de ReAct. La différence avec un programme classique est que l'enchaînement des étapes n'est pas écrit à l'avance : il dépend de la demande.

La génération augmentée par recherche (RAG, Lewis et al., 2020) est un cas particulier : l'un des outils cherche des passages dans des documents, et le modèle s'appuie dessus pour répondre. Dans notre projet, c'est l'outil `chercher_passages`, construit sur la fonction `retrieve` du rôle 3.

## 4.2 Architecture

### Vue d'ensemble

```
Utilisateur (application Streamlit)
   |  demande, projets, documents, poids éventuels
   v
Agent (src/agent.py)  <---->  LLM local (Ollama, qwen2.5:3b)
   |                            choisit l'outil suivant
   v
src/outils.py
   |-- predict            rôle 1 (supervisé)
   |-- rank               rôle 2 (multicritère)
   |-- detect_anomalies   rôle 2 (non supervisé)
   |-- extract            rôle 3 (NLP)
   |-- retrieve           rôle 3 (RAG)
```

Les quatre rôles ont fixé dès le départ un contrat commun (`docs/contrat.md`) : le nom, les entrées et les sorties de chaque fonction. En attendant les vrais modules, de fausses fonctions qui respectent ce contrat (`src/mocks.py`) ont permis de coder et de tester l'agent et l'application. Une seule variable, `USE_MOCKS`, fait passer des fausses fonctions aux vrais modules.

### Les cinq outils du LLM

| Outil | Ce qu'il fait |
|---|---|
| `extraire_criteres` | Note les documents de tous les projets sur trois critères qualitatifs (`extract`) |
| `predire` | Prédit la rentabilité et la probabilité de risque élevé de chaque projet (`predict`) |
| `detecter_anomalies` | Repère les dossiers suspects (`detect_anomalies`) |
| `classer(poids)` | Classe les projets sur sept critères avec les poids donnés (`rank`) |
| `chercher_passages(question, k)` | Cherche des passages à citer dans les documents (`retrieve`) |

### Trois décisions de conception

**Chaque outil traite tout le portefeuille.** Le jeu de données compte 2 000 projets. Un outil par projet obligerait le LLM à faire des milliers d'appels, ce qui est impossible. Chaque outil traite donc tous les projets d'un coup.

**Les données restent dans Python.** Le LLM ne voit jamais le tableau des projets. Après chaque outil, il reçoit un résumé de quelques lignes, par exemple « 48 dossiers anormaux, à écarter de la recommandation ». Il décide et il rédige, mais tous les calculs sont faits par le code. Le contexte reste court, et le modèle ne peut pas se tromper dans un calcul qu'il ne fait pas.

**Ce qui peut être vérifié par le code ne dépend pas du LLM.** Plusieurs garde-fous rattrapent les erreurs du modèle :

- `classer` lance lui-même les étapes oubliées (notes, prédictions, anomalies) ;
- les arguments inventés par le LLM sont ignorés, seuls ceux que l'outil accepte sont transmis ;
- les poids proposés sont ramenés à une somme de 1, et ceux fixés par l'utilisateur passent avant ;
- les dossiers anormaux sont retirés des trois projets recommandés ;
- si le LLM ne cherche pas de passage après `classer`, l'agent le fait lui-même, avec les noms des projets recommandés ;
- après la rédaction, un contrôle ajoute les projets et les alertes oubliés, et signale une citation qui ne figure pas mot pour mot dans un passage rendu par `chercher_passages` ;
- si le LLM rend une réponse vide, l'agent le relance une fois ;
- si Ollama ne répond pas ou si le LLM ne termine pas en dix tours, l'agent finit sans LLM et le dit.

Chaque appel d'outil est noté dans un journal (outil, arguments, résumé du résultat, durée), affiché dans l'application.

### Le modèle de référence

La consigne demande de comparer un modèle simple à des modèles plus avancés. Pour l'agent, le modèle simple est `pipeline_fixe` : les cinq outils dans le même ordre, sans LLM, avec des poids par défaut (ou ceux de l'utilisateur), et une recommandation rédigée à partir d'un modèle de texte. Il sert aussi de secours pour la démonstration si Ollama est arrêté.

### Le choix du LLM

Contrainte de l'équipe : un LLM local, gratuit, sans service externe. Les essais ont été faits sur un PC portable sans carte graphique (Ryzen 5 7520U, 16 Go de mémoire).

| Modèle | Résultat |
|---|---|
| `qwen3:4b` | Écarté. Le modèle réfléchit toujours avant de répondre, même avec la réflexion désactivée : environ 4 600 jetons par tour, soit 10 minutes. Essai arrêté après deux tours |
| `qwen2.5:3b` | Retenu. Pas de phase de réflexion, il gère l'appel d'outils. Analyse complète en 58 secondes au premier essai |

La boucle de l'agent est écrite à la main avec le paquet `ollama`, en une trentaine de lignes. Un cadriciel comme LangGraph n'apportait rien pour cinq outils et une boucle simple, et aurait rendu le code plus difficile à expliquer.

## 4.3 Méthode d'évaluation

L'évaluation juge le comportement de l'agent, pas la qualité des prédictions, qui relève des autres rôles. Elle sert aussi d'analyse détaillée des erreurs, la méthode d'interprétabilité retenue pour cette partie.

Protocole Cinq demandes, sur les 30 premiers projets en cours du jeu de données, avec leurs documents. Chaque scénario est lancé deux fois, avec l'agent et avec `pipeline_fixe`. Modèle `qwen2.5:3b`, température 0, contexte de 8 192 jetons.

| N° | Demande | Ce qu'on attend |
|---|---|---|
| 1 | « Je veux des projets peu risqués avec un fort impact environnemental » | Poids forts sur le risque et l'impact |
| 2 | « Je cherche les projets les plus rentables, même s'ils sont risqués » | Poids le plus fort sur la rentabilité |
| 3 | « Je veux des projets rapides à réaliser, peu coûteux et faciles à mettre en œuvre » | Poids forts sur la durée, le coût et la faisabilité |
| 4 | « Classe les projets avec les poids que j'ai fixés » | Exactement les poids de l'utilisateur |
| 5 | « Quel temps fera-t-il demain ? » | Rien d'inventé |

Vérifications automatiques (`src/scenarios.py`) : respect du contrat, ordre des outils, cohérence des poids avec la demande, présence des trois projets recommandés et du dossier anormal dans le texte, et surtout fidélité. Chaque nombre décimal du texte doit correspondre à une valeur rendue par un outil, et chaque passage cité doit avoir été rendu par `chercher_passages`.

## 4.4 Résultats

### Avec les fausses fonctions (16 novembre)

Sur les dix exécutions de l'agent :

- contrat respecté, aucun plantage ;
- poids cohérents avec la demande dans les 8 analyses menées par le LLM ;
- aucun chiffre inventé, aucun identifiant inventé, aucune citation inventée dans la sortie finale ;
- durée : 64 secondes en moyenne, 77 secondes sans la demande hors sujet, contre moins de 0,1 seconde pour le pipeline fixe.

| | Agent | Pipeline fixe |
|---|---|---|
| Durée | 64 s en moyenne (12 à 103 s) | moins de 0,1 s |
| Poids adaptés à la demande (scénarios 1 à 3) | 6 sur 6 | 2 sur 6 |
| Outils appelés | 4 sur 5 en général | toujours les 5 |
| Texte | alerte souvent oubliée, ajoutée par le contrôle | toujours complet |
| Même résultat aux deux passages | non pour le scénario 1 | oui |

Aux scénarios 1 et 3, l'agent recommande d'autres projets que le pipeline fixe, parce qu'il a adapté les poids à la demande. Au scénario 3, les trois projets sont entièrement différents.

### Avec les quatre vrais modules (19 novembre)

Mêmes scénarios, mêmes projets, avec `predict` (rôle 1), `rank` et `detect_anomalies` (rôle 2), `extract` et `retrieve` (rôle 3). Sur les dix exécutions : contrat respecté, aucun plantage, aucun chiffre, identifiant ou citation inventé.

| | 16 novembre (fausses fonctions) | 19 novembre (vrais modules) |
|---|---|---|
| Passages cherchés et cités | 1 analyse sur 8 | 8 sur 8 |
| Alerte citée par le LLM lui-même | 2 sur 8 | 6 sur 8 |
| Poids cohérents avec la demande | 8 sur 8 | 6 sur 8 (scénario 3) |
| Mêmes résultats aux deux passages | non pour le scénario 1 | oui pour les 5 scénarios |
| Demande hors sujet | refus, puis secours | refus, puis secours |
| Durée d'une analyse complète | 77 s en moyenne | 110 s en moyenne |

L'agent recommande d'autres projets que le pipeline fixe aux scénarios 1, 2 et 3. Au scénario 1 (« peu risqués avec un fort impact environnemental »), il met en tête un programme de réduction des émissions noté 5/5 en impact, et cite un passage de son document. Le pipeline fixe, lui, rend les trois mêmes projets pour toutes les demandes sans poids imposés. Une analyse prend environ deux minutes, contre 0,3 seconde pour le pipeline fixe : les vrais modules renvoient plus d'informations que le LLM doit lire avant de rédiger.

## 4.5 Analyse des erreurs

| Erreur observée | Cause | Ce qui a été fait |
|---|---|---|
| Réponse vide au premier tour (scénario 1) : l'agent abandonnait et prenait les poids par défaut | Le modèle et son serveur : la même demande donne une réponse normale ou vide selon l'état du serveur | Corrigée : une relance automatique. Le scénario aboutit, mais avec des poids un peu différents |
| Demande hors sujet : la réponse du LLM était perdue | Le code jetait ce texte | Corrigée : le texte est gardé dans le journal |
| Ligne « Alertes » recopiée telle quelle depuis le format demandé (6 textes sur 8) | Le prompt et la taille du modèle | Non corrigée dans le prompt, rattrapée par le contrôle automatique |
| `chercher_passages` presque jamais demandé (1 fois sur 8) ; avec le vrai `retrieve`, citation inventée | Le modèle s'arrête après `classer` | Corrigée : l'agent lance la recherche lui-même après `classer`, et le contrôle vérifie que la citation figure dans un passage rendu. Passages cités dans 8 analyses sur 8 à l'intégration |
| Intégration : poids de l'exemple du prompt recopiés (scénario 3), analyse lancée pour une question de météo (scénario 5) | Une modification du prompt faite la veille pour une autre raison, avec un exemple qui reprenait une demande de test | Corrigée : formulation d'origine rétablie ; le scénario 5 refuse de nouveau, le scénario 3 ne recopie plus l'exemple mais sous-estime encore la faisabilité |
| Arguments inventés, poids proposés malgré ceux de l'utilisateur | Le modèle imite les exemples du prompt | Sans effet : le code les ignore |
| Pas de phrase qui relie chaque projet à la demande | Le modèle recopie les résultats des outils | Non corrigée |

Deux reformulations du prompt ont été essayées pour corriger la ligne « Alertes ». La première a bien fait citer l'alerte, mais elle a faussé les poids du scénario 3 (rentabilité 0,26 pour une demande « rapide, peu coûteux, facile »). La seconde a fait inventer une citation dans les 8 textes rédigés. Le prompt d'origine a été gardé : c'est le seul dont toutes les erreurs sont rattrapées par le code. Avec un modèle de 3 milliards de paramètres, changer deux lignes du prompt déplace les erreurs au lieu de les supprimer. L'intégration l'a confirmé une troisième fois : trois lignes ajoutées pour une autre raison ont suffi à faire recopier un exemple au LLM.

Autre constat : la température 0 ne suffit pas à rendre les réponses reproductibles. Le 16 novembre, au scénario 1, le troisième projet recommandé changeait d'un passage à l'autre. À l'intégration, les deux passages ont donné les mêmes résultats pour les cinq scénarios, mais rien ne le garantit.

## 4.6 Discussion et limites

Sur ces essais, l'apport de l'agent tient au choix des poids. Il traduit une demande en langage courant en importance des critères, et le classement change avec la demande, ce que le pipeline fixe ne fait pas. Il reconnaît aussi une demande hors sujet. Le reste (enchaîner les outils, écrire un texte complet, citer les bons chiffres) est fait aussi bien, ou mieux, par du code. C'est pour cela que l'architecture laisse au LLM la décision et la rédaction, et confie tout ce qui se vérifie au code.

Limites :

- Cibles simulées. La rentabilité et le risque prédits par le rôle 1 sont appris sur des cibles simulées : l'évaluation juge le comportement de l'agent, pas la valeur réelle des projets recommandés.
- Petit modèle. `qwen2.5:3b` recopie plus qu'il ne rédige et réagit fortement à la formulation du prompt. Un modèle plus gros n'était pas utilisable dans un délai raisonnable sur ce matériel.
- Petit échantillon. 30 projets, 5 demandes, 2 passages : assez pour repérer des erreurs, pas pour donner un taux de réussite.
- Vérifications simples. Elles cherchent des identifiants et des nombres dans le texte. Elles ne jugent pas la clarté de la rédaction.
- Démonstration locale. Ollama ne tourne pas sur l'hébergement gratuit de Streamlit : l'application est montrée en local, dans la vidéo.

## 4.7 L'application

L'application Streamlit ne calcule rien : elle charge les projets, appelle l'agent ou le pipeline fixe, et affiche ce qu'ils rendent. On y trouve :

- la demande, avec trois exemples, et des curseurs pour fixer les poids ;
- un relief 3D du portefeuille, une colonne par projet, dont la hauteur est le score et la couleur distingue les projets recommandés et les dossiers anormaux ;
- les trois projets recommandés, l'alerte sur les dossiers anormaux, le classement complet, le journal des étapes, la comparaison avec le pipeline fixe et le graphique SHAP du modèle de rentabilité (rôle 1).

Après une analyse, bouger les curseurs refait le classement sans relancer l'agent. Tout tourne en local : le LLM, les polices et la bibliothèque 3D sont sur la machine, l'application ne contacte aucun service externe.

## Références

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS.
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.

## Diapositives (3 à 4)

1. L'agent en une image. Le schéma de la section 4.2 : l'utilisateur, l'agent, le LLM, les cinq outils et les rôles qui les fournissent. Une phrase : « Le LLM décide et rédige, le code calcule et vérifie. »
2. Comment il fonctionne. La boucle de l'agent (le LLM choisit un outil, le code l'exécute, le LLM lit le résumé) et la liste courte des garde-fous.
3. Ce qu'il apporte. Le tableau agent contre pipeline fixe (section 4.4), avec l'exemple du scénario 3 : des poids adaptés, trois projets différents.
4. Ce qu'on a appris. Les deux essais de prompt qui déplacent les erreurs, la température 0 qui ne suffit pas, et la conclusion : confier au code tout ce qui se vérifie.
