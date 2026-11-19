# Partie 1 : apprentissage supervisé de la rentabilité et du risque

Brouillon du 19 novembre 2026. Tous les chiffres viennent du module livré (`src/supervise.py`) et des notebooks `notebooks/role1/01_exploration.ipynb`, `02_modeles.ipynb` et `03_interpretabilite.ipynb`. Les résultats de bout en bout avec l'agent seront ajoutés après l'intégration du 22 novembre.

Le rôle 1 fournit à l'agent la fonction `predict(projet)`. Elle reçoit la fiche d'un projet d'investissement et rend trois valeurs : la rentabilité prédite (en %), un niveau de risque (`faible`, `moyen` ou `eleve`) et la probabilité que le risque soit élevé. La rentabilité et cette probabilité sont deux des sept critères du classement multicritère du rôle 2. L'agent appelle `predict` sur chacun des projets qu'il analyse, jusqu'à 2 000 par demande.

## 1.1 Fondements théoriques

### Apprentissage supervisé

On dispose d'exemples $(x_i, y_i)$ : $x_i$ décrit un projet (secteur, pays, coût, durée...), $y_i$ est la valeur qu'on veut apprendre à prédire. L'apprentissage supervisé cherche une fonction $f$ telle que $f(x)$ soit proche de $y$ pour des projets **qui ne faisaient pas partie des exemples**. Quand $y$ est un nombre (la rentabilité), c'est une régression ; quand $y$ est une catégorie (le niveau de risque), c'est une classification.

La difficulté tient à cette dernière exigence. Un modèle assez souple peut reproduire parfaitement les exemples d'apprentissage en apprenant aussi leur bruit : c'est le surapprentissage. Il se trompe alors davantage sur des projets nouveaux. À l'inverse, un modèle trop rigide rate des effets réels. Toute la méthode consiste à mesurer la qualité sur des données que le modèle n'a pas vues, et à régler sa souplesse d'après cette mesure (Hastie et al., 2009).

### Régression linéaire

La régression linéaire suppose $y \approx \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$. Les moindres carrés ordinaires choisissent les coefficients qui minimisent la somme des carrés des erreurs sur l'entraînement :

$$\hat\beta = \arg\min_\beta \sum_i \big(y_i - \beta_0 - x_i^\top \beta\big)^2$$

Une variable catégorielle comme le pays entre sous forme d'indicatrices : une colonne par pays, qui vaut 1 pour les projets de ce pays et 0 sinon. Avec 158 pays, le modèle a 158 coefficients rien que pour cette variable, souvent estimés sur un ou deux projets. C'est là que le surapprentissage nous a frappés.

### Ridge, Lasso et ElasticNet

La régularisation ajoute au critère une pénalité sur la taille des coefficients, réglée par $\alpha \geq 0$ :

- **Ridge** (Hoerl et Kennard, 1970) : $\sum_i (y_i - \beta_0 - x_i^\top\beta)^2 + \alpha \sum_j \beta_j^2$. Tous les coefficients sont rapetissés vers 0 sans jamais l'atteindre. Un pays représenté par un seul projet ne reçoit plus un coefficient qui colle exactement à ce projet.
- **Lasso** (Tibshirani, 1996) : la pénalité devient $\alpha \sum_j |\beta_j|$. La forme de la valeur absolue pousse une partie des coefficients à exactement 0 : le Lasso sélectionne des variables en même temps qu'il les estime.
- **ElasticNet** (Zou et Hastie, 2005) mélange les deux pénalités ; `l1_ratio` donne la part du Lasso.

Avec $\alpha = 0$, on retrouve la régression linéaire ; avec $\alpha$ très grand, tous les coefficients s'écrasent et le modèle prédit la moyenne. Comme la pénalité porte sur la taille des coefficients, les variables doivent être sur la même échelle : on les centre et on les réduit avant.

### Classification

La **régression logistique** multinomiale calcule un score linéaire par classe et le transforme en probabilités qui somment à 1 (fonction softmax). Elle se régularise comme Ridge ; le paramètre `C` est l'inverse de la force de la pénalité.

Un **arbre de décision** découpe les données par seuils successifs (« durée > 90 mois ? », « secteur = Énergie ? ») et prédit dans chaque feuille la classe majoritaire. Il capte seul les effets non linéaires et les interactions, mais un arbre profond apprend le bruit. La **forêt aléatoire** (Breiman, 2001) fait voter des centaines d'arbres, chacun appris sur un tirage des projets et ne regardant qu'une partie des colonnes à chaque découpage. Le **boosting de gradient** ajoute au contraire de petits arbres un par un, chacun corrigeant les erreurs des précédents ; LightGBM (Ke et al., 2017) en est une version rapide.

Quand une classe est rare, un modèle peut gagner en exactitude en l'ignorant. L'option `class_weight="balanced"` donne à chaque erreur sur une classe un poids inversement proportionnel à sa fréquence.

### Validation croisée

Pour régler $\alpha$, `C` ou la profondeur des arbres sans toucher aux jeux de validation et de test, on coupe l'entraînement en $k$ parts (ici 5). Pour chaque valeur à essayer, on apprend sur 4 parts, on mesure sur la cinquième, on recommence 5 fois en changeant de part, et on fait la moyenne. On garde la valeur qui donne la meilleure moyenne (`GridSearchCV` de scikit-learn, Pedregosa et al., 2011). Pour la classification, les plis sont **stratifiés** : chacun garde les proportions des classes. Le prétraitement est placé dans le même `Pipeline` que le modèle, si bien qu'il est réappris sur les 4 parts d'apprentissage à chaque pli.

### Mesures

| Mesure | Définition | Lecture |
|---|---|---|
| RMSE | $\sqrt{\frac1n \sum (y_i - \hat y_i)^2}$ | erreur typique, en points de rentabilité ; punit fort les grosses erreurs |
| MAE | $\frac1n \sum \lvert y_i - \hat y_i \rvert$ | erreur moyenne en points, moins sensible aux valeurs extrêmes |
| R² | $1 - \sum (y_i - \hat y_i)^2 / \sum (y_i - \bar y)^2$ | part de la variance expliquée ; 0 = pas mieux que la moyenne |
| Exactitude | part de projets bien classés | trompeuse si les classes sont déséquilibrées |
| F1 macro | moyenne, sur les trois classes, du F1 (moyenne harmonique de la précision et du rappel) | chaque classe compte autant, même la plus rare |
| AUC | probabilité qu'un projet à risque élevé reçoive une `proba_risque` plus haute qu'un projet qui ne l'est pas | 0,5 = hasard, 1 = parfait |

## 1.2 Les données

Le fichier `data/processed/projets.csv` contient 2 000 projets réels de la Banque mondiale (licence CC BY 4.0), dont 1 500 terminés et 500 en cours. Les entrées du modèle sont réelles : secteur (11 valeurs), pays (158), région (8), type de financement (4), coût total, durée prévue, année d'approbation. Les deux cibles, elles, sont **simulées** par une formule écrite dans `data/README.md`, avec du bruit : la Banque mondiale ne publie ni rentabilité ni niveau de risque. Elles ne sont connues que pour les 1 500 projets terminés ; les 500 projets en cours sont ceux que l'agent évalue. Un modèle qui apprend ces cibles retrouve la formule, pas la réalité économique, et aucune conclusion sur les vrais projets ne peut en être tirée.

L'analyse exploratoire n'a trouvé ni doublon ni valeur manquante chez les projets terminés. Elle a fixé plusieurs choix :

- **Le coût est très asymétrique** (asymétrie de 18,5, de 1 à 37 707 millions de dollars). Son logarithme décimal est presque symétrique (asymétrie 0,06) ; on utilise donc $\log_{10}$ du coût.
- **Les classes de risque sont déséquilibrées** : 40 % `faible`, 40 % `moyen`, 20 % `eleve`. Répondre toujours la classe la plus fréquente donne déjà 40 % d'exactitude.
- **Le pays a beaucoup de valeurs rares** : 42 pays ont moins de 5 projets, et 3 pays des projets en cours n'apparaissent jamais parmi les projets terminés.
- **Les projets en cours ne ressemblent pas tout à fait aux projets terminés** : plus récents (année médiane 2022 contre 2009) et plus chers (107 contre 46 millions de dollars en médiane). Le modèle prédit donc en partie hors de ce qu'il a appris.
- **40 anomalies ont été fabriquées** dans les données, dont 25 parmi les projets terminés : coûts multipliés par 10 à 50, durées remplacées par 1 à 3 mois, et 10 rentabilités remplacées par une valeur au hasard entre 35 et 60 %, positive ou négative. Ces dernières suffisent à faire passer l'écart-type de la rentabilité de 3,1 à 5,1 points. Leur détection est le travail du rôle 2, mais elles sont dans nos données d'apprentissage.

## 1.3 Méthode

### Séparation des données

Les 1 500 projets terminés ont été séparés **avant tout prétraitement** (`scripts/separer_jeux_supervise.py`) : 1 050 pour l'entraînement, 225 pour la validation, 225 pour le test, avec un tirage stratifié sur le risque (40 / 40 / 20 % dans chaque jeu) et la graine 42. Seuls les identifiants sont enregistrés (`reports/role1/separation.csv`). Les rôles des trois jeux ne se mélangent pas : l'entraînement sert à apprendre et à régler par validation croisée, la validation à comparer les modèles déjà réglés, le test à mesurer une seule fois le modèle choisi.

### Prétraitement

Le prétraitement (`supervise.creer_pretraitement`) est un `ColumnTransformer` placé dans un `Pipeline` avec chaque modèle :

- coût : $\log_{10}$, valeur manquante remplacée par la médiane, puis centrage et réduction ;
- durée et année : médiane pour les manquants, centrage et réduction ;
- secteur, pays, région, financement : une indicatrice par catégorie vue à l'entraînement, avec `handle_unknown="ignore"`. Une catégorie jamais vue donne des zéros partout au lieu de faire planter. Une catégorie manquante est remplacée par « inconnu », donc traitée comme une catégorie jamais vue. Remplacer un pays manquant par le pays le plus fréquent (la Chine) lui aurait donné à tort l'effet propre à la Chine.

On obtient 175 colonnes. Deux défauts sont apparus en testant des fiches piégées : `SimpleImputer` ne reconnaît pas `None` comme valeur manquante et plante sur `pd.NA`. Une première étape ramène ces valeurs, et tout texte non numérique, à `NaN`.

### Rentabilités aberrantes

Les rentabilités fabriquées ne dépendent de rien : aucun modèle ne peut les prédire, et une régression par moindres carrés, qui élève les erreurs au carré, se laisse tirer par elles. Une première version utilisait la liste des anomalies pour les retirer de l'entraînement. Elle a été remplacée à la relecture : cette liste ne doit pas servir au modèle, et en situation réelle on ne l'aurait pas. La règle retenue est apprise sur l'entraînement seulement. Une rentabilité est aberrante si elle sort de $[Q_1 - 3\,\mathrm{IQR} ; Q_3 + 3\,\mathrm{IQR}]$, où $Q_1$ et $Q_3$ sont les quartiles et $\mathrm{IQR} = Q_3 - Q_1$. Sur l'entraînement, cela donne $[-6{,}1\,\% ; 23{,}3\,\%]$. Les quartiles bougent à peine à cause de quelques valeurs extrêmes, donc la règle n'est pas faussée par ce qu'elle doit repérer. Les mêmes bornes sont appliquées à la validation et au test.

La liste des anomalies a seulement servi à vérifier après coup : la règle attrape les 10 rentabilités fabriquées des projets terminés, et aucun autre projet. Elle ne voit pas les coûts et durées modifiés, puisque ceux-ci faussent une entrée et non la rentabilité. Les modèles de rentabilité apprennent sur les 1 044 projets d'entraînement dans les bornes ; ceux du risque gardent les 1 050 projets, car le risque a été calculé avant l'injection des anomalies. Les mesures de régression sont données deux fois, sur la validation complète et sur la validation dans les bornes.

### Modèles comparés et règle de choix

Pour la rentabilité : la moyenne (`DummyRegressor`), la régression linéaire, Ridge, Lasso, ElasticNet, une forêt aléatoire et LightGBM. Pour le risque : la classe la plus fréquente et un tirage au hasard (`DummyClassifier`), la régression logistique, une forêt aléatoire et LightGBM. Les réglages sont choisis par validation croisée à 5 plis sur l'entraînement, avec la RMSE pour la régression et le F1 macro pour la classification ; l'option `class_weight` fait partie des réglages essayés.

La règle de choix a été fixée avant d'ouvrir le test : pour la rentabilité, la plus petite RMSE sur la validation dans les bornes, et le modèle le plus simple en cas d'égalité à 0,01 point près ; pour le risque, le meilleur F1 macro sur la validation.

## 1.4 Résultats

### Rentabilité

| Modèle | Réglage | RMSE validation croisée | R² entraînement | RMSE validation (bornes) | MAE validation (bornes) | R² validation (bornes) | R² validation complète |
|---|---|---|---|---|---|---|---|
| Moyenne | | | 0 | 2,78 | 2,24 | −0,03 | −0,01 |
| Régression linéaire | | | 0,67 | 2,14 | 1,63 | 0,38 | 0,08 |
| Ridge | α ≈ 3,2 | 2,12 | 0,64 | **2,05** | 1,60 | **0,44** | 0,09 |
| Lasso | α ≈ 0,003 | 2,14 | 0,64 | 2,07 | 1,63 | 0,42 | 0,09 |
| ElasticNet | α ≈ 0,003, l1_ratio 0,1 | 2,12 | 0,64 | 2,05 | 1,61 | 0,44 | 0,09 |
| Forêt aléatoire | profondeur libre | 2,25 | 0,93 | 2,12 | 1,72 | 0,40 | 0,06 |
| LightGBM | 500 arbres de 7 feuilles | 2,21 | 0,70 | 2,16 | 1,75 | 0,37 | 0,04 |

Quatre constats. D'abord, la régression linéaire apprend par cœur : R² de 0,67 sur l'entraînement contre 0,38 en validation. La régularisation réduit cet écart et fait gagner environ 5 points de R² (0,44 pour Ridge et ElasticNet). Ensuite, les modèles à arbres font moins bien que Ridge, en validation croisée comme en validation. La rentabilité simulée est surtout une somme d'effets (secteur, pays, financement, durée), la forme même d'un modèle linéaire ; l'effet pays porte sur 158 catégories souvent représentées par un ou deux projets, qu'un modèle linéaire régularisé estime par un coefficient rapetissé, alors qu'un arbre doit isoler chaque pays par des découpages successifs. Un modèle plus avancé n'est donc pas meilleur ici.

Troisième constat : les rentabilités fabriquées pèsent très lourd. Les 6 de l'entraînement font tomber le R² de validation de la régression linéaire de 0,38 à 0,22 si on les garde ; les 2 de la validation (+38,1 % et −56,0 %) font plus que doubler la RMSE et ramènent le R² de tous les modèles sous 0,1. Enfin, on reste sous le R² d'environ 0,55 annoncé par le contrôle du script de préparation. L'analyse des erreurs (section 1.5) en montre une raison : la formule contient une courbure en durée qu'un modèle linéaire ne suit pas.

Le Lasso met 66 des 175 coefficients à zéro, dont 63 pays sur 149, et `secteur_Agriculture` : dans la formule, l'agriculture est justement le secteur d'effet nul. Ses plus gros coefficients retrouvent la formule sans qu'on la lui donne : Énergie +2,9 (formule : +2,5), Protection sociale −2,0 (−2), Gouvernance −2,0 (−2), Transport +1,6 (+1,5), durée −1,4 par écart-type (−1,2), financement IBRD +1,1 (+1), coût −0,7 (−0,8).

Ridge (α ≈ 3,2) a été retenu : sa RMSE (2,050) est à égalité avec celle d'ElasticNet (2,051), et il n'a qu'un réglage au lieu de deux.

### Risque

| Modèle | F1 macro validation croisée | Exactitude entraînement | Exactitude validation | F1 macro validation | AUC élevé | Rappel `eleve` |
|---|---|---|---|---|---|---|
| Classe la plus fréquente | | | 40 % | 0,19 | | 0 % |
| Tirage selon les proportions | | | 36 % | 0,33 | | |
| Régression logistique (C = 0,1) | 0,65 ± 0,02 | 70 % | 55 % | 0,53 | 0,86 | 38 % |
| Forêt aléatoire (équilibrée) | 0,62 ± 0,04 | 76 % | 58 % | 0,57 | 0,86 | 53 % |
| LightGBM (équilibré) | 0,62 ± 0,02 | 77 % | **59 %** | **0,59** | 0,86 | 58 % |

Les trois modèles battent nettement les références et atteignent le niveau du contrôle de préparation (58 %). Les erreurs se font presque toujours entre classes voisines : confondre `faible` et `eleve` n'arrive que 1 à 5 fois sur 225 projets. C'est attendu, puisque les trois classes sont des tranches d'un même score continu. L'avance des arbres vient de la classe `eleve` : réglés avec `class_weight="balanced"`, ils en retrouvent 53 à 58 %, contre 38 % pour la régression logistique, que la validation croisée a réglée sans pondération. Pour `proba_risque`, les trois modèles se valent (AUC de 0,86).

La validation croisée et la validation ne classent pas les modèles dans le même ordre : la régression logistique gagne la première, LightGBM la seconde. Les écarts sont du même ordre que le bruit de mesure. L'écart-type entre plis est de 0,02 à 0,04 en F1 macro, et sur 225 projets une exactitude de 58 % a un intervalle de confiance à 95 % d'environ ±6,5 points. Selon la règle fixée, LightGBM a été retenu (300 arbres de 7 feuilles, pas d'apprentissage 0,03, au moins 30 projets par feuille, classes équilibrées).

### Évaluation sur le test

Le test a été ouvert une seule fois, sur les deux modèles choisis appris sur les 1 050 projets d'entraînement.

| | Validation | Test | Référence sur le test |
|---|---|---|---|
| Rentabilité, R² (bornes) | 0,44 | 0,47 | −0,01 |
| Rentabilité, RMSE (bornes) | 2,05 | 2,24 | 3,09 |
| Rentabilité, MAE (bornes) | 1,60 | 1,86 | 2,46 |
| Rentabilité, R² (complet) | 0,09 | 0,12 | 0,00 |
| Risque, exactitude | 59 % | 67 % | 40 % |
| Risque, F1 macro | 0,59 | 0,66 | 0,19 |
| Risque, AUC élevé | 0,86 | 0,88 | |
| Risque, rappel `eleve` | 58 % | 62 % | 0 % |

Ridge réduit l'erreur typique d'environ 28 % par rapport à la moyenne. Le R² du test est un peu plus haut qu'en validation alors que la RMSE est plus grande : la rentabilité varie davantage dans le test, il y a donc plus de variance à expliquer. Les deux rentabilités fabriquées du test (+50,5 % et −55,0 %) sont prédites autour de 10 à 11 %, comme n'importe quel projet de ce type, et suffisent à ramener le R² complet à 0,12. Les prédictions sont resserrées vers le centre : les projets très rentables sont sous-estimés et les moins rentables surestimés, ce qui est attendu quand une partie de la cible est du bruit.

Les 67 % du risque dépassent le plafond annoncé, ce qui nous a fait chercher une fuite. Aucun identifiant n'est commun aux jeux, aucun projet du test n'a les mêmes entrées qu'un projet d'entraînement, et le prétraitement n'a vu que l'entraînement. Si la vraie exactitude est d'environ 62 % (moyenne de la validation croisée et de la validation), observer au moins 67 % sur 225 projets arrive environ 7 fois sur 100. Nous lisons donc ce résultat comme un tirage favorable : la meilleure estimation de la qualité du modèle de risque est d'environ 60 %, à quelques points près. Le test ne dit pas non plus si la régression logistique aurait fait mieux ; le lui demander après coup reviendrait à choisir sur le test.

### Modèles livrés

Une fois le choix fait et le test mesuré, Ridge et LightGBM ont été **réappris sur les 1 500 projets terminés**, avec exactement les mêmes réglages. La règle IQR, recalculée sur ces 1 500 projets ($[-6{,}1\,\% ; 22{,}8\,\%]$), écarte les mêmes 10 rentabilités, et Ridge apprend sur 1 490 projets. **Les chiffres du test décrivent le modèle appris sur les 1 050 projets d'entraînement.** Le modèle livré dispose de 43 % de projets en plus ; ses performances devraient être au moins aussi bonnes, mais elles ne sont pas mesurées, faute de données neuves.

## 1.5 Interprétabilité et erreurs

L'interprétabilité porte sur le modèle à 1 050 projets, celui qui a été évalué, analysé sur les 450 projets de validation et de test.

### SHAP

Pour un projet, la valeur SHAP d'une variable est sa contribution à l'écart entre la prédiction et la prédiction moyenne ; les contributions s'additionnent (Lundberg et Lee, 2017). Pour Ridge, elle vaut coefficient × (valeur du projet − valeur moyenne) ; pour LightGBM, `TreeExplainer` la calcule exactement à partir des arbres. Les contributions des 175 colonnes sont regroupées par variable d'origine.

| Variable | SHAP rentabilité (points) | Permutation rentabilité (hausse de RMSE) | SHAP risque élevé (log-cote) | Permutation risque (baisse de F1 macro) |
|---|---|---|---|---|
| secteur | 1,30 | 0,80 | 0,18 | 0,03 |
| durée | 1,14 | 0,70 | 0,91 | 0,22 |
| pays | 0,82 | 0,44 | 0,04 | 0,01 |
| financement | 0,59 | 0,22 | 0,13 | 0,00 |
| coût | 0,53 | 0,17 | 0,59 | 0,08 |
| région | 0,36 | 0,01 | 0,44 | 0,07 |
| année | 0,10 | 0,02 | 0,09 | 0,02 |

Pour la rentabilité, l'ordre est celui de la formule : effets de secteur de −2 à +2,5 points, durée, effet pays d'écart-type 1,5, financement ±1, coût, et l'année absente. Pour le risque, la durée, le coût et la région dominent, comme dans la formule du risque (0,5 × durée + 0,4 × coût + effet de région − 0,6 × rentabilité). La durée y compte deux fois, directement et à travers la rentabilité.

### Importance par permutation

On mélange au hasard les valeurs d'une variable dans les données d'évaluation, 20 fois, et on mesure la dégradation du modèle. Les deux méthodes s'accordent sur l'essentiel, sauf pour la région dans le modèle de rentabilité : SHAP lui attribue 0,36 point, mais la mélanger ne coûte que 0,01 point de RMSE. Ridge s'en sert comme d'un résumé des pays voisins, alors que le pays contient déjà l'information. SHAP mesure l'usage d'une variable, la permutation sa nécessité.

### Les plus grosses erreurs

Les 10 plus grosses erreurs de rentabilité se rangent en trois familles. Les 4 premières (28 à 66 points) sont les rentabilités fabriquées, que la règle IQR marque comme aberrantes. La huitième est une durée fabriquée : 1 mois au lieu de la vraie durée, le modèle voit un projet très court et surestime sa rentabilité de 5,5 points ; seul un détecteur d'anomalies placé avant `predict` pourrait l'éviter. Les 5 autres sont des projets normaux, avec des erreurs de 5 à 6,5 points, soit environ 3 fois l'erreur typique : surtout de petits projets (3 à 25 millions de dollars) dans des pays rares à l'entraînement (Dominica n'y a qu'un projet), plus le bruit propre à chaque projet dans la simulation (écart-type 1,8 point).

Une erreur est systématique : le modèle surestime la rentabilité des projets très longs (+1,7 point en moyenne au-delà de 120 mois) et des projets très courts (+1,1 point sous 30 mois), et il est juste entre les deux. C'est le terme −0,5 × durée² de la formule, qui pénalise les durées éloignées de la moyenne dans les deux sens. Ajouter la durée au carré comme variable corrigerait probablement ce défaut ; nous ne l'avons pas fait, pour ne pas rouvrir le test.

Pour le risque, les 10 erreurs les plus sûres d'elles (au moins 88 % de probabilité donnée à la mauvaise classe) concernent toutes des projets vraiment `moyen`, la tranche étroite entre les deux seuils. Deux sont des durées fabriquées. Pour 6 des 8 autres, la rentabilité réelle explique l'erreur : un projet moins rentable que prévu est plus risqué que ses caractéristiques ne le laissent penser, et le modèle ne peut pas le savoir, puisque la rentabilité réelle est inconnue au moment de prédire.

## 1.6 La fonction `predict`

`predict` charge une seule fois les deux modèles enregistrés avec `joblib` dans `models/supervise.joblib` (hors Git). Si le fichier manque, est illisible ou a été écrit avec une autre version de scikit-learn, elle réapprend les modèles et les enregistre. Elle ne lit que les 7 entrées du modèle : les cibles éventuellement présentes dans la fiche sont ignorées.

Le premier essai prenait 33 secondes pour 2 000 projets : prétraiter une seule ligne coûte environ 15 ms, surtout en vérifications internes de scikit-learn. Au chargement, les prédictions des 2 000 projets de `projets.csv` sont donc calculées en une fois (0,2 s) et rangées selon les valeurs de leurs entrées. Les 2 000 appels prennent ensuite 0,005 s ; un projet inconnu (fichier importé, valeur modifiée) passe par le calcul complet en 18 ms environ. Un test vérifie que les deux chemins donnent exactement le même résultat.

Les 17 tests de `tests/test_supervise.py` couvrent le contrat (la sortie passe le modèle `Prediction` de `src/contrat.py` pour les 500 projets en cours), les pays et secteurs inconnus, les valeurs manquantes sous toutes leurs formes, une fiche vide, la rapidité, le rechargement des modèles, la séparation des jeux et l'absence de fuite dans le prétraitement. Branchée dans l'agent (`MODULES_REELS=supervise`), l'étape de prédiction traite les 500 projets en cours en 0,21 s. Elle en classe 95 à risque faible, 320 moyen et 85 élevé : plus de `moyen` que parmi les projets terminés, ce qui va avec des projets plus récents et plus chers.

## 1.7 Limites

- Les cibles sont simulées : les modèles retrouvent une formule, et rien ne dit qu'ils feraient aussi bien sur de vraies rentabilités.
- Le bruit de la simulation fixe un plafond : environ 2 points d'erreur typique sur la rentabilité, environ 60 % de bonnes classes de risque.
- Les mesures reposent sur 225 projets par jeu : quelques points d'écart entre modèles ne sont pas significatifs.
- Les chiffres du test décrivent le modèle à 1 050 projets ; le modèle livré, réappris sur 1 500 projets, n'a pas de mesure propre.
- Les projets en cours sont plus récents et plus chers que les projets terminés ; le coût n'est pas corrigé de l'inflation (1987 à 2026).
- Les entrées fausses (coût ou durée fabriqués) donnent des prédictions fausses ; il faudrait que les dossiers repérés par le rôle 2 soient signalés avant d'utiliser la prédiction.
- La courbure de la durée n'est pas modélisée.

## Références

- Banque mondiale, *Projects & Operations*, https://projects.worldbank.org, consulté le 16 novembre 2026, licence CC BY 4.0. Données modifiées par l'équipe (filtres, regroupements, cibles simulées).
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Hastie, T., Tibshirani, R. et Friedman, J. (2009). *The Elements of Statistical Learning* (2e éd.). Springer.
- Hoerl, A. E. et Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
- Lundberg, S. M. et Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267-288.
- Zou, H. et Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society, Series B*, 67(2), 301-320.

## Diapositives (3 à 4)

1. Données et méthode. 2 000 projets réels, cibles simulées, 1 500 terminés séparés en 1 050 / 225 / 225 avant tout prétraitement. Le schéma du `Pipeline` (log10 du coût, imputation, standardisation, indicatrices `handle_unknown="ignore"`) et la règle IQR apprise sur l'entraînement, qui attrape les 10 rentabilités fabriquées et rien d'autre.
2. Rentabilité. La figure `comparaison_modeles.png` (partie gauche) : la régularisation passe de 0,38 à 0,44 de R², les arbres ne font pas mieux. La figure `regularisation_alpha.png` pour le choix de α, et le tableau des coefficients du Lasso face à la formule.
3. Risque. Les matrices de confusion (`confusion_risque.png`) : erreurs entre classes voisines ; le rappel de la classe `eleve` (38 % contre 58 %) grâce aux classes équilibrées. Le test : 67 %, l'enquête sur la fuite, et l'estimation honnête d'environ 60 %.
4. Interprétabilité. `shap_rentabilite.png` et `importance_permutation.png` : l'ordre de la formule retrouvé, le cas de la région (utilisée mais inutile), et `erreurs_duree.png` pour l'erreur systématique aux deux extrémités de la durée.
