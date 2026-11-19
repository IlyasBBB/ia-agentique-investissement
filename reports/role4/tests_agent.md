# Tests de l'agent sur cinq scénarios

Évaluation du 16 novembre 2026 avec les fausses fonctions, puis intégration du 19 novembre avec les quatre vrais modules (section 9). Toutes les valeurs de ce document viennent d'exécutions réelles, dont les fichiers sont dans `reports/`.

## 1. Objectif

Vérifier que l'agent se comporte correctement sur des demandes variées, et le comparer au modèle de référence `pipeline_fixe` (les cinq outils toujours dans le même ordre, sans LLM, poids par défaut). Ce document sert aussi d'analyse détaillée des erreurs, qui est la méthode d'interprétabilité retenue pour l'agent.

**Ce qui est évalué.** Le comportement de l'agent : outils demandés, poids tirés de la demande, fidélité du texte aux chiffres rendus par les outils, honnêteté quand il ne sait pas.

**Ce qui n'est pas évalué.** La qualité des prédictions, des notes et du classement. Sections 2 à 8 : essais du 16 novembre, quand les modules des rôles 1, 2 et 3 n'étaient pas encore livrés ; les essais tournent avec les fausses fonctions (`USE_MOCKS=1`). Les rentabilités, risques et notes sont donc factices, et aucun projet cité ici n'est réellement « bon » ou « mauvais ».

## 2. Protocole

| Élément | Choix |
|---|---|
| Projets | Les 30 premiers projets en cours de `data/processed/projets.csv`, triés par identifiant (P121507 à P163260), avec leur document `data/docs/<id>.txt`. Règle fixe : on retrouve toujours les mêmes projets |
| Dossier anormal | P121507 (4 694,75 millions de dollars), le seul signalé par la fausse détection sur ces 30 projets |
| Modèle | `qwen2.5:3b` avec Ollama 0.40.16, température 0, contexte de 8 192 jetons |
| Matériel | Ryzen 5 7520U, 16 Go de mémoire, calcul sur processeur |
| Modes | `run_agent` (LLM) et `pipeline_fixe` (référence), sur les mêmes projets |
| Passages | 2 passages complets, pour juger la stabilité |
| Reproduire | `python scripts/evaluer_agent.py --passages 2` (environ 11 minutes) |

### Les cinq scénarios

| N° | Demande | Poids attendus |
|---|---|---|
| 1 | « Je veux des projets peu risqués avec un fort impact environnemental » | `proba_risque` et `impact_env_social` au moins aussi forts que tous les autres |
| 2 | « Je cherche les projets les plus rentables, même s'ils sont risqués » | `rentabilite_predite` le plus fort |
| 3 | « Je veux des projets rapides à réaliser, peu coûteux et faciles à mettre en œuvre » | `cout_total`, `duree_mois` et `faisabilite` au moins aussi forts que tous les autres |
| 4 | « Classe les projets avec les poids que j'ai fixés », avec faisabilité 0,5, coût 0,3, rentabilité 0,2 | Exactement ceux de l'utilisateur |
| 5 | « Quel temps fera-t-il demain ? » (hors sujet) | Aucune priorité : poids par défaut, et surtout rien d'inventé |

### Les vérifications automatiques (`src/scenarios.py`)

| Vérification | Ce qu'elle regarde |
|---|---|
| Contrat | Les trois clés de la sortie, un classement valide, un journal valide |
| Ordre | `extraire_criteres`, `predire` et `detecter_anomalies` demandés avant `classer` |
| Passages | `chercher_passages` demandé |
| Poids | Cohérents avec la demande (tableau ci-dessus) |
| Trois projets | Les trois premiers du classement, dossiers anormaux écartés, cités dans le texte |
| Alerte | Le dossier anormal cité dans le texte |
| Chiffres | Chaque nombre à virgule du texte correspond à une valeur rendue par les outils |
| Citation | Aucun passage cité qui n'ait été rendu par `chercher_passages` |
| Contrôle, secours | Le code a dû compléter le texte du LLM, ou le remplacer |

Pour les trois projets et l'alerte, on distingue le texte écrit par le LLM et la sortie finale, après le contrôle automatique.

**Limite du contrôle des chiffres.** Il compare les nombres à virgule au score, à la rentabilité, à la probabilité de risque et au coût des projets cités, ainsi qu'aux poids, à une unité près sur le dernier chiffre écrit. Il ne contrôle pas les nombres entiers (notes sur 5, rangs). Un chiffre exact mais attribué au mauvais projet cité passerait.

**Limite sur les passages.** La fausse fonction `retrieve` ne connaît que cinq documents d'exemple (P001 à P005). Sur les 30 projets réels, `chercher_passages` ne trouve donc jamais rien. La seule réponse honnête est « Passage cité : aucun ». Ce cas teste bien l'honnêteté, mais pas la qualité des citations.

## 3. Résultats du 16 novembre (fausses fonctions)

Fichier : `reports/role4/scenarios/resultats_scenarios.csv`. Textes complets : `reports/role4/scenarios/sorties_scenarios.json`.

### Agent

« Complété » : le contrôle automatique a ajouté l'alerte oubliée en fin de texte.

| Scénario | Passage | Durée | Ordre | Passages cherchés | Poids cohérents | 3 projets cités par le LLM | Alerte citée par le LLM | Chiffres et citation | Sortie finale |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 76,9 s | oui | oui | oui | oui | non | corrects | complétée |
| 1 | 2 | 103,1 s | oui (après relance) | non | oui | oui | non | corrects | complétée |
| 2 | 1 | 64,9 s | oui | non | oui | oui | non | corrects | complétée |
| 2 | 2 | 70,4 s | oui | non | oui | oui | non | corrects | complétée |
| 3 | 1 | 76,9 s | oui | non | oui | oui | non | corrects | complétée |
| 3 | 2 | 69,1 s | oui | non | oui | oui | non | corrects | complétée |
| 4 | 1 | 78,2 s | oui | non | oui (ceux de l'utilisateur) | oui | oui | corrects | texte du LLM seul |
| 4 | 2 | 75,8 s | oui | non | oui (ceux de l'utilisateur) | oui | oui | corrects | texte du LLM seul |
| 5 | 1 | 11,6 s | aucun outil demandé | non | par défaut | sans objet | sans objet | corrects | secours sans LLM |
| 5 | 2 | 12,3 s | aucun outil demandé | non | par défaut | sans objet | sans objet | corrects | secours sans LLM |

Dans les dix exécutions : contrat respecté, aucun plantage, les trois projets et le dossier anormal présents dans la sortie finale, aucun chiffre non retrouvé, aucun identifiant inventé, aucune citation inventée. Durée moyenne : 63,9 s (76,9 s sans le scénario 5).

### Poids choisis par l'agent et trois projets recommandés

| Scénario | Passage | Poids | Trois projets |
|---|---|---|---|
| 1 | 1 | risque 0,50, impact 0,50 | P159577, P161534, P151739 |
| 1 | 2 | impact 0,35, risque 0,26, alignement 0,17, faisabilité 0,09, les trois autres 0,04 | P161534, P159577, P158003 |
| 2 | 1 et 2 | rentabilité 0,45, risque 0,18, coût 0,14, durée 0,09, les trois autres 0,05 | P161534, P159112, P159577 |
| 3 | 1 et 2 | faisabilité 0,50, durée 0,40, coût 0,10 | P155827, P160427, P159655 |
| 4 | 1 et 2 | faisabilité 0,50, coût 0,30, rentabilité 0,20 (utilisateur) | P151439, P158003, P159655 |
| 5 | 1 et 2 | par défaut | P161534, P159577, P159112 |

### Pipeline fixe

Moins de 0,1 s par scénario. Toujours les cinq outils dans le même ordre, toutes les vérifications de forme réussies, texte produit par un modèle de texte. Poids par défaut (rentabilité 0,25, risque 0,20, impact 0,15, alignement 0,15, coût 0,10, faisabilité 0,10, durée 0,05), sauf au scénario 4 où il prend ceux de l'utilisateur. Trois projets : P161534, P159577, P159112 pour les scénarios 1, 2, 3 et 5 ; P151439, P158003, P159655 pour le scénario 4.

## 4. Analyse détaillée des erreurs

Chaque erreur observée, sa cause probable et ce qui a été fait.

### E1. Réponse vide du LLM au premier tour (scénario 1) : corrigée

- **Observé.** Premier passage avant correction : scénario 1 réussi en 110,0 s avec les poids 0,50 et 0,50. Second passage, même demande : le LLM ne rend ni texte ni outil. L'agent bascule sur le secours en 27,7 s, avec les poids par défaut, alors que la demande donnait des priorités. La recommandation est alors celle du pipeline fixe.
- **Diagnostic.** En rappelant le modèle à la main avec exactement le même message : la première réponse après un changement de demande contient quatre outils (128 jetons produits) ; les suivantes sont vides (223 jetons produits, que le serveur Ollama n'arrive à lire ni comme texte ni comme appel d'outil). Fixer une graine ne change rien. Avec un message de relance ajouté, le modèle appelle `extraire_criteres` en moins de 3 s.
- **Cause.** Le modèle et son serveur, pas le prompt ni le code : à température 0 la réponse devrait être la même, elle ne l'est pas selon que le serveur a déjà lu ce message ou non.
- **Correction (code).** Après une réponse vide, l'agent envoie une fois le message de `prompts/relance.txt` et le note dans le journal. Une seconde réponse vide déclenche le secours. Test sans LLM : `test_reponse_vide_relancee_une_fois`.
- **Après.** Scénario 1 relancé seul deux fois (`resultats_scenarios_correction1.csv`) : relance déclenchée les deux fois, analyse complète en 82,5 s et 82,4 s. Dans l'évaluation finale, la relance s'est déclenchée au passage 2 (103,1 s).
- **Ce qui reste.** Après une relance, le modèle appelle les outils un par un et choisit des poids plus étalés (impact 0,35, risque 0,26) que sans relance (0,50 et 0,50). Le troisième projet recommandé change : P158003, dont la probabilité de risque élevé est 0,67, à la place de P151739 (0,24). Pour une demande « peu risqué », c'est moins bon.

### E2. Réponse du LLM perdue sur la demande hors sujet (scénario 5) : corrigée

- **Observé.** Le LLM répond en une dizaine de secondes, sans outil. L'agent affiche « L'agent n'a pas terminé (le LLM a répondu sans classer les projets) », suivi d'un classement aux poids par défaut. Rien n'est inventé, mais on ne sait pas ce que le LLM a répondu.
- **Diagnostic.** Rappelé à la main, le LLM répond correctement : « Les demandes "Quel temps fera-t-il demain ?" ne sont pas gérées par les outils mis à disposition. […] Je ne peux donc pas les utiliser pour répondre à cette demande. »
- **Cause.** Le code : le texte du LLM était écarté sans laisser de trace.
- **Correction (code).** Ce texte est gardé dans le journal des étapes (ligne `llm`). Il n'est toujours pas repris dans la recommandation, qui ne contient que du vérifié. Test sans LLM : `test_reponse_sans_outil_gardee_dans_le_journal`.
- **Après.** Aux deux passages finaux, le journal contient le refus du LLM, puis le classement de secours.
- **Ce qui reste.** L'utilisateur reçoit quand même trois projets pour une question de météo. Proposition, non appliquée : dans ce cas, afficher le refus du LLM à la place de la recommandation.

### E3. Ligne « Alertes » recopiée telle quelle (scénarios 1, 2, 3) : non corrigée, rattrapée par le contrôle

- **Observé.** Dans 6 exécutions finales sur 8 avec texte du LLM, celui-ci écrit « Alertes : les dossiers anormaux écartés, ou « aucune ». » : il recopie la consigne du format au lieu de la remplir. Au scénario 4, il cite bien P121507.
- **Cause.** Le prompt (une ligne de format qui ressemble à une phrase à recopier) et le modèle (petit, il recopie).
- **Effet.** Le contrôle automatique ajoute l'alerte en fin de texte dans les 6 cas. La phrase fausse du LLM reste visible au-dessus.
- **Deux reformulations du prompt essayées, aucune gardée.** Chacune a été évaluée sur les 5 scénarios et 2 passages.

| Prompt | Alerte citée par le LLM | Poids cohérents | Citation inventée | Scénario 5 |
|---|---|---|---|---|
| D'origine (gardé) | 2 sur 8 | 8 sur 8 | 0 | refus, puis secours |
| Format avec crochets à remplacer (`resultats_scenarios_essai_crochets.csv`) | 10 sur 10 | 6 sur 10 | 0 | analyse complète avec des poids sans rapport avec la demande, sans dire que la question est hors sujet |
| Seule la ligne « Alertes » changée (`resultats_scenarios_essai_alertes.csv`) | 8 sur 8 | 8 sur 8 | 8 sur 8 | refus, puis secours |

  Avec les crochets, les poids du scénario 3 deviennent faux (rentabilité 0,26 pour une demande « rapide, peu coûteux, facile ») et rien dans le code ne le détecte. Avec la ligne « Alertes » seule, le LLM invente un passage dans les 8 textes qu'il rédige, par exemple « Pour un projet à succès, il est essentiel de combiner une forte vision environnementale avec une stratégie d'affaires solide. » ; le contrôle le signale, mais la fausse citation reste dans le texte.
- **Décision.** Le prompt d'origine est gardé : c'est le seul dont toutes les erreurs sont rattrapées par le code sans laisser de citation inventée. Leçon pour le rapport : avec un modèle de cette taille, changer deux lignes du prompt déplace les erreurs au lieu de les supprimer. Proposition : que le contrôle remplace la ligne « Alertes » fautive au lieu d'ajouter un bloc à la fin.

### E4. `chercher_passages` presque jamais demandé : corrigée le 18 novembre

- **Observé.** Demandé 1 fois sur 8 dans l'évaluation finale (scénario 1, passage 1), jamais dans les trois autres séries. Le LLM s'arrête après `classer`.
- **Cause.** Le modèle : il demande les outils d'un seul coup et omet le dernier. Demander dans le prompt de « toujours l'appeler après classer » (essai avec crochets) n'a rien changé.
- **Effet aujourd'hui.** Faible : la fausse recherche ne trouve rien sur ces projets, et le LLM écrit honnêtement « Passage cité : aucun ».
- **Effet à l'intégration.** Important : sans cet appel, la recommandation de l'agent n'aura pas de citation, alors que le pipeline fixe en aura une. Proposition : lancer la recherche d'office après `classer`, comme `classer` lance déjà les étapes oubliées. À décider quand le vrai module sera là.
- **Correction du 18 novembre (code).** Essai avec le vrai `retrieve` (rôle 3), 500 projets : le LLM s'arrête après `classer` et invente une citation. L'agent lance maintenant `chercher_passages` lui-même après `classer`, avec les noms des projets recommandés comme question. Premier essai : le LLM recopie la question au lieu du passage, rattrapé par le contrôle. Après avoir retiré la question de la conversation, le LLM cite mot pour mot un passage du projet recommandé P505887. Le contrôle vérifie aussi, désormais, que la citation figure dans un passage rendu par l'outil. Tests : `test_passages_cherches_d_office_apres_classer`, `test_citation_inventee_signalee_meme_avec_des_passages`.

### E5. Arguments inventés : déjà rattrapée

- **Observé.** Scénario 1, passage 1 : `{'portefeuille': '30'}` ajouté aux cinq outils. Autres séries : `identifiants_projets`, `documents`.
- **Cause.** Le modèle. **Effet.** Aucun : l'agent ne transmet que les arguments que l'outil accepte, et le journal garde ce que le LLM a envoyé.

### E6. Poids proposés malgré les poids de l'utilisateur (scénario 4) : déjà rattrapée

- **Observé.** Aux deux passages, le LLM appelle `classer` avec ses propres poids (rentabilité 0,4, durée 0,3…), alors que le message lui demandait de ne pas en donner.
- **Cause.** Le modèle (il imite l'exemple du prompt). **Effet.** Aucun sur le résultat : les poids de l'utilisateur passent avant, et les trois projets sont ceux du pipeline fixe. Défaut de lisibilité : le journal montre des poids qui n'ont pas servi.

### E7. Texte incomplet par rapport au format demandé : non corrigée

- **Observé.** Aucune des 8 recommandations rédigées ne contient la phrase qui relie chaque projet à la demande. Au scénario 2, les chiffres sont écrits sans leur nom (« 0.846, 13.0 %, 0.22 »).
- **Cause.** Le modèle : il recopie la ligne rendue par l'outil et n'ajoute rien. **Effet.** Le texte de l'agent n'explique pas plus que celui du pipeline fixe. À surveiller au scénario 3 : deux projets recommandés ont une probabilité de risque élevé de 0,91 et 0,95. C'est cohérent avec la demande (le risque a un poids nul), mais le texte ne prévient pas le lecteur.

## 5. Comparaison avec le pipeline fixe

| | Agent | Pipeline fixe |
|---|---|---|
| Durée | 63,9 s en moyenne (11,6 à 103,1 s) | moins de 0,1 s |
| Poids cohérents avec la demande, scénarios 1 à 3 | 6 sur 6 | 2 sur 6 (scénario 2, où les poids par défaut mettent déjà la rentabilité en tête) |
| Poids de l'utilisateur respectés (scénario 4) | oui | oui |
| Outils | 4 sur 5 en général, `chercher_passages` oublié | toujours les 5 |
| Texte | parfois faux sur l'alerte, complété par le contrôle | toujours complet |
| Stabilité | variable au scénario 1 | identique à chaque fois |

**Mêmes trois projets ?**

- Scénario 1 : non. L'agent retient P151739 ou P158003 à la place de P159112.
- Scénario 2 : les mêmes trois projets, dans un ordre différent (P159112 passe devant P159577).
- Scénario 3 : non, trois projets entièrement différents.
- Scénario 4 : oui, identiques (mêmes poids imposés).
- Scénario 5 : oui, par construction (l'agent se rabat sur le même calcul).

**Ce que l'agent apporte.** Il traduit la demande en poids, et le classement change avec la demande. Le pipeline fixe rend le même classement quelle que soit la demande. L'agent reconnaît aussi une demande hors sujet.

**Ce qu'il coûte.** Environ une minute par analyse au lieu d'un instant, un résultat qui peut changer d'un passage à l'autre, un texte qu'il faut contrôler par du code, et un outil oublié. La rédaction, elle, n'apporte rien de plus que le modèle de texte du pipeline.

**Conclusion honnête.** Sur ces essais, tout l'apport de l'agent tient dans le choix des poids. Tout le reste est mieux fait, ou aussi bien fait, par du code.

## 6. Stabilité entre les deux passages

- Scénarios 2, 3, 4 et 5 : mêmes outils, mêmes poids, mêmes trois projets, même texte aux deux passages. Durées voisines (écart de 5,5 s, 7,8 s, 2,4 s et 0,7 s).
- Scénario 1 : instable. Selon l'état du serveur, le LLM répond normalement ou rend une réponse vide. Les poids et le troisième projet changent (voir E1). La relance évite le secours, mais ne rend pas le résultat identique.
- La température 0 ne garantit donc pas la reproductibilité avec ce modèle et ce serveur.
- Les durées varient aussi d'une série à l'autre pour une même demande (scénario 1 : de 76,9 à 110,0 s sans relance).

## 7. Limites

- **Chiffres factices.** Avec les fausses fonctions, seul le comportement de l'agent est jugé. L'évaluation a été refaite avec les vrais modules le 19 novembre (section 9).
- **Citations non testées.** La fausse recherche ne trouve rien sur les projets réels.
- **Petit modèle.** `qwen2.5:3b` (3,1 milliards de paramètres) est sensible à la formulation du prompt et recopie plus qu'il ne rédige. Un modèle plus gros n'a pas pu être essayé sur ce matériel dans un délai utilisable.
- **Petit échantillon.** 30 projets, 5 demandes, 2 passages : assez pour repérer des erreurs, pas pour donner un taux de réussite fiable.
- **Vérifications simples.** Elles reposent sur la présence d'identifiants et de nombres dans le texte. Elles ne jugent ni la clarté de la rédaction ni la pertinence fine des poids (seulement leur ordre).
- **Un seul dossier anormal** dans l'échantillon.

## 8. Fichiers

| Fichier | Contenu |
|---|---|
| `src/scenarios.py` | Les cinq scénarios, les 30 projets, les vérifications |
| `scripts/evaluer_agent.py` | Lance l'évaluation (`--passages`, `--scenarios`, `--suffixe`) |
| `tests/test_agent.py` | Les scénarios en tests : pipeline fixe (sans LLM) et vrai modèle (`python -m pytest -m llm`) |
| `reports/role4/scenarios/resultats_scenarios_integration.csv`, `sorties_scenarios_integration.json` | Évaluation finale, avec les quatre vrais modules (19 novembre, section 9) |
| `reports/role4/scenarios/resultats_scenarios.csv`, `sorties_scenarios.json` | Évaluation du 16 novembre, fausses fonctions, après les corrections E1 et E2 |
| `reports/role4/scenarios/*_avant_corrections.*` | Première évaluation, avant toute correction |
| `reports/role4/scenarios/*_correction1.*` | Scénario 1 relancé après la correction E1 |
| `reports/role4/scenarios/*_essai_crochets.*`, `*_essai_alertes.*` | Les deux reformulations du prompt, non gardées |
| `reports/role4/scenarios/*_integration_prompt18nov.*`, `*_diagnostic*` | Erreur E8 : évaluation avec le prompt du 18 novembre et diagnostic |

## 9. Intégration du 19 novembre : les quatre vrais modules

Mêmes cinq scénarios, mêmes 30 projets, deux passages, avec `USE_MOCKS=0` : `predict` (Ridge et LightGBM, rôle 1), `rank` (TOPSIS) et `detect_anomalies` (Isolation Forest, rôle 2), `extract` et `retrieve` (rôle 3). Reproduire : `USE_MOCKS=0 python scripts/evaluer_agent.py --passages 2 --suffixe integration` (environ 20 minutes). Fichiers : `reports/role4/scenarios/resultats_scenarios_integration.csv`, `reports/role4/scenarios/sorties_scenarios_integration.json`.

### Résultats de l'agent

| Scénario | Durée (passages 1 et 2) | Poids cohérents | Poids choisis | Alerte citée par le LLM | Contrôle automatique |
|---|---|---|---|---|---|
| 1. Peu risqué, fort impact | 132,6 s ; 136,5 s (une relance) | oui | risque 0,33, impact 0,33, rentabilité 0,17 | oui | non |
| 2. Rentable, même risqué | 110,6 s ; 105,0 s | oui | rentabilité 0,45, risque 0,18, coût 0,14 | oui | non |
| 3. Rapide, peu coûteux, facile | 115,8 s ; 96,6 s | non | durée 0,36, rentabilité 0,18, les autres 0,09 | non | oui (alerte ajoutée) |
| 4. Poids imposés | 90,7 s ; 90,4 s | oui (ceux de l'utilisateur) | faisabilité 0,50, coût 0,30, rentabilité 0,20 | oui | oui (projets ajoutés) |
| 5. Hors sujet (météo) | 13,6 s ; 13,6 s | par défaut | aucun outil demandé : refus, puis secours | sans objet | secours sans LLM |

Sur les dix exécutions : contrat respecté, aucun plantage, aucun chiffre ni identifiant inventé, aucune citation inventée. Durée moyenne d'une analyse complète (scénarios 1 à 4) : 109,8 s.

### Ce qui a changé par rapport au 16 novembre (fausses fonctions)

| | 16 novembre | 19 novembre |
|---|---|---|
| Passages cherchés et cités | 1 analyse sur 8 | 8 sur 8 (recherche d'office après `classer`, correction E4) |
| Citations inventées dans la sortie | 0 | 0 (citations vérifiées mot pour mot) |
| Alerte citée par le LLM lui-même | 2 sur 8 | 6 sur 8 |
| Poids cohérents avec la demande | 8 sur 8 | 6 sur 8 (scénario 3) |
| Mêmes résultats aux deux passages | non pour le scénario 1 | oui pour les 5 scénarios |
| Demande hors sujet | refus, puis secours | refus, puis secours |
| Durée moyenne d'une analyse complète | 76,9 s | 109,8 s |

La durée augmente parce que les vrais modules renvoient plus d'informations (notes, passages) que le LLM doit lire avant de rédiger, et parce que la recherche d'office ajoute ses passages à la conversation.

### E8. Exemple du prompt recopié (scénarios 3 et 5) : corrigée le 19 novembre

- **Observé.** Première évaluation d'intégration (`reports/role4/scenarios/resultats_scenarios_integration_prompt18nov.csv`) : au scénario 3, le LLM prend exactement les poids de l'exemple du prompt (rentabilité 0,36, durée 0,27) ; au scénario 5, il lance une analyse complète pour une question de météo au lieu de refuser.
- **Diagnostic.** Les scénarios 3 et 5 relancés avec les fausses fonctions (`reports/role4/scenarios/resultats_scenarios_diagnostic.csv`) donnent la même erreur au scénario 3 : la cause n'est pas dans les modules. Le seul changement du premier tour depuis le 16 novembre est l'étape 5 du prompt, réécrite le 18 novembre pour faire écrire la question de recherche en anglais, avec un exemple qui reprenait mot pour mot la demande du scénario 1.
- **Correction (prompt).** Formulation d'origine de l'étape 5 rétablie. L'indication « en anglais » reste dans la description de l'outil, et la recherche d'office utilise de toute façon les noms des projets, en anglais. Relancés avec les fausses fonctions (`reports/role4/scenarios/resultats_scenarios_diagnostic_prompt.csv`) : le scénario 3 ne recopie plus l'exemple, le scénario 5 refuse de nouveau.
- **Ce qui reste.** Au scénario 3, la durée a bien le poids le plus fort, mais la faisabilité et le coût restent sous la rentabilité. Troisième confirmation de la leçon du point E3 : avec un modèle de 3 milliards de paramètres, une modification du prompt, même faite pour une autre raison, déplace les erreurs.

### Comparaison avec le pipeline fixe

Le pipeline fixe répond en 0,3 s (14 s au premier appel, chargement des modèles compris) et recommande toujours P157929, P155827 et P158003, sauf au scénario 4 (poids imposés). L'agent recommande d'autres projets aux scénarios 1, 2 et 3, parce qu'il adapte les poids à la demande ; au scénario 1, il met en tête P155827, un programme de réduction des émissions noté 5/5 en impact. Aux scénarios 4 et 5, les deux modes donnent les mêmes projets.
