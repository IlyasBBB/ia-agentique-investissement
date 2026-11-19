# Partie 2 : apprentissage non supervisé et classement multicritère

Brouillon du 19 novembre 2026. Tous les chiffres viennent du module livré (`src/mcda.py`), du script `scripts/evaluer_anomalies_mcda.py` et des notebooks `notebooks/role2/01_non_supervise.ipynb` et `02_mcda_sensibilite.ipynb`. Les résultats de bout en bout avec l'agent seront ajoutés après l'intégration du 22 novembre.

Le rôle 2 fournit à l'agent deux fonctions. `rank(matrice, poids)` classe les projets selon sept critères pondérés : la rentabilité prédite et la probabilité de risque élevé (rôle 1), le coût, la durée, et trois notes qualitatives tirées des documents (rôle 3). `detect_anomalies(df)` signale les dossiers dont les chiffres sont suspects : l'agent les écarte de sa recommandation et les affiche en alerte. Le rôle 2 a aussi mené l'analyse non supervisée des 2 000 projets (ACP, quatre familles de clustering), pour savoir si les projets forment des familles distinctes.

## 2.1 Fondements théoriques

### Analyse en composantes principales (ACP)

On dispose de $n$ projets décrits par $p$ variables centrées et réduites, rangées dans une matrice $X$ de taille $n \times p$. L'ACP cherche une direction $u_1$ (un vecteur de norme 1) telle que les projections $X u_1$ soient le plus dispersées possible, puis une direction $u_2$ perpendiculaire à $u_1$ qui maximise la dispersion restante, et ainsi de suite. Ces directions sont les vecteurs propres de la matrice des corrélations $R = \frac{1}{n} X^\top X$ :

$$R\, u_k = \lambda_k u_k, \qquad \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_p$$

La valeur propre $\lambda_k$ est la variance des projets le long de l'axe $k$. Comme les variables sont réduites, $\sum_k \lambda_k = p$ : la part de variance expliquée par l'axe $k$ vaut $\lambda_k / p$. Pour lire un axe, on calcule la corrélation entre chaque variable $j$ et l'axe, $\sqrt{\lambda_k}\, u_{kj}$ ; une variable très corrélée à l'axe lui donne son nom (Jolliffe et Cadima, 2016).

Le critère de Kaiser garde les axes de valeur propre supérieure à 1 : un axe doit résumer plus de variance qu'une seule variable d'origine. Le graphique des valeurs propres (« coude ») et la part de variance cumulée servent de vérification.

La standardisation est indispensable avant l'ACP comme avant toute méthode fondée sur des distances : sans elle, une variable exprimée en grandes unités (l'année, de 1987 à 2026) écraserait une variable bornée entre 0 et 1 (une probabilité).

### Clustering : quatre familles

Le clustering cherche des groupes de projets semblables sans aucune étiquette. Nous avons comparé une méthode de chaque grande famille.

**Partition : K-means** (Lloyd, 1982). On fixe le nombre de groupes $k$. L'algorithme cherche des centres $\mu_1, \dots, \mu_k$ qui minimisent l'inertie, somme des carrés des distances de chaque projet au centre de son groupe :

$$\min_{\mu_1,\dots,\mu_k} \sum_{i=1}^{n} \min_{c} \lVert x_i - \mu_c \rVert^2$$

Il alterne deux étapes jusqu'à stabilité : chaque projet rejoint le centre le plus proche, puis chaque centre devient la moyenne de ses projets. Le résultat dépend du point de départ : on relance l'algorithme plusieurs fois (10 ici) et on garde la meilleure inertie. K-means produit des groupes compacts, de forme à peu près sphérique.

**Hiérarchique : méthode de Ward** (Ward, 1963). On part de $n$ groupes d'un seul projet. À chaque étape, on fusionne les deux groupes dont la réunion augmente le moins l'inertie totale. Le dendrogramme garde la trace de toutes les fusions et de leur coût : le couper à une hauteur donnée fournit une partition. Aucun point de départ aléatoire n'intervient.

**Densité : DBSCAN** (Ester et al., 1996). Un projet est un point « central » s'il a au moins `min_samples` voisins à une distance inférieure à $\varepsilon$. Les points centraux voisins forment un même groupe, les points proches d'un point central y sont rattachés, et les autres sont déclarés **bruit**. DBSCAN ne demande pas de nombre de groupes, trouve des groupes de forme quelconque, et met à part les points isolés. Il faut en revanche choisir $\varepsilon$ : on trie les distances de chaque point à son `min_samples`-ième voisin et on prend le coude de cette courbe.

**Modèle probabiliste : mélange gaussien.** Les données sont vues comme tirées d'un mélange de $K$ lois normales multivariées :

$$p(x) = \sum_{c=1}^{K} \pi_c\, \mathcal{N}(x \mid \mu_c, \Sigma_c)$$

Les paramètres sont estimés par l'algorithme EM, et chaque projet reçoit une probabilité d'appartenir à chaque composante. Avec des covariances $\Sigma_c$ libres, les groupes peuvent être allongés ou inclinés. Le nombre de composantes se choisit par le critère BIC, $\mathrm{BIC} = -2 \ln \hat L + q \ln n$, où $q$ est le nombre de paramètres : il récompense la vraisemblance et pénalise la complexité (Schwarz, 1978).

**Mesures de qualité.** La silhouette d'un projet $i$ vaut $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$, où $a_i$ est sa distance moyenne aux projets de son groupe et $b_i$ sa distance moyenne aux projets du groupe voisin le plus proche (Rousseeuw, 1987). Elle va de −1 à 1 : proche de 1, le projet est bien à sa place ; proche de 0, il est entre deux groupes. L'indice de Davies-Bouldin (plus bas = mieux) et celui de Calinski-Harabasz (plus haut = mieux) comparent la dispersion à l'intérieur des groupes et l'écart entre groupes. L'indice de Rand ajusté mesure l'accord entre deux partitions : 1 si elles sont identiques, 0 si leur accord n'est pas meilleur que le hasard (Hubert et Arabie, 1985).

### Détection d'anomalies : Isolation Forest

L'Isolation Forest (Liu, Ting et Zhou, 2008) part d'une idée simple : un point anormal est **facile à isoler**. On construit des arbres en coupant les données au hasard : à chaque nœud, on tire une variable au hasard, puis un seuil au hasard entre son minimum et son maximum. Un point isolé, loin des autres, se retrouve seul après très peu de coupes ; un point au milieu de la masse demande beaucoup de coupes. Le score d'anomalie d'un point $x$ se calcule à partir de la profondeur moyenne $E[h(x)]$ à laquelle il est isolé, sur tous les arbres :

$$s(x) = 2^{-E[h(x)]/c(n)}$$

où $c(n)$ est la profondeur moyenne attendue pour $n$ points. Un score proche de 1 signale une anomalie. Le paramètre de **contamination** fixe la part des points d'apprentissage déclarés anormaux, donc le seuil sur ce score. La méthode ne fait aucune hypothèse sur la forme des données et reste rapide sur des milliers de points.

### Classement multicritère

On classe $m$ projets selon $p$ critères. Chaque critère $j$ a un poids $w_j \geq 0$, avec $\sum_j w_j = 1$, et un sens : à maximiser (rentabilité, notes) ou à minimiser (risque, coût, durée).

**Somme pondérée, la référence.** Chaque critère est ramené entre 0 et 1 par min-max sur les projets classés, et inversé s'il est à minimiser :

$$r_{ij} = \frac{x_{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}} \;\text{(à maximiser)}, \qquad r_{ij} = \frac{\max_i x_{ij} - x_{ij}}{\max_i x_{ij} - \min_i x_{ij}} \;\text{(à minimiser)}$$

Le score est $S_i = \sum_j w_j r_{ij}$, entre 0 et 1. La méthode est transparente : chaque critère ajoute sa part. Elle est aussi **compensatoire** (un très bon critère rattrape un très mauvais), et la normalisation min-max dépend entièrement des deux valeurs extrêmes de chaque critère.

**AHP, pour fixer les poids** (Saaty, 1980). Plutôt que d'écrire sept poids directement, le décideur compare les critères deux à deux sur l'échelle de 1 à 9 (1 : aussi important ; 3 : un peu plus ; 5 : nettement plus ; 9 : extrêmement plus). On obtient une matrice $A$ positive et réciproque : $a_{ji} = 1/a_{ij}$. Si les jugements étaient parfaitement cohérents, on aurait $a_{ij} = w_i / w_j$, et $w$ serait vecteur propre de $A$ pour la valeur propre $p$. On prend donc pour poids le vecteur propre associé à la plus grande valeur propre $\lambda_{\max}$, ramené à une somme de 1. L'écart à la cohérence se mesure par :

$$IC = \frac{\lambda_{\max} - p}{p - 1}, \qquad RC = \frac{IC}{IA_p}$$

où $IA_p$ est l'indice aléatoire de Saaty, cohérence moyenne de matrices remplies au hasard (0,58 pour 3 critères, 1,32 pour 7). On accepte les jugements si $RC < 0{,}1$.

**TOPSIS** (Hwang et Yoon, 1981), en cinq étapes :

1. normalisation vectorielle : $r_{ij} = x_{ij} / \sqrt{\sum_i x_{ij}^2}$ ;
2. pondération : $v_{ij} = w_j\, r_{ij}$ ;
3. solution idéale $A^+$ (meilleure valeur de chaque critère : maximum ou minimum selon le sens) et anti-idéale $A^-$ (la pire) ;
4. distances euclidiennes de chaque projet à l'idéale et à l'anti-idéale : $d_i^+ = \lVert v_i - A^+ \rVert$, $d_i^- = \lVert v_i - A^- \rVert$ ;
5. score de proximité $C_i = d_i^- / (d_i^+ + d_i^-)$, entre 0 (l'anti-idéal) et 1 (l'idéal).

Un bon projet est à la fois proche du meilleur possible et loin du pire possible.

## 2.2 Données et préparation

Les 2 000 projets réels de la Banque mondiale (1 500 terminés, 500 en cours) sont décrits dans la partie Données. Pour l'analyse non supervisée, nous avons retenu **8 variables** :

- les trois colonnes chiffrées réelles : le coût (en log10), la durée et l'année d'approbation ;
- les cinq critères que l'agent calcule pour chaque projet : la rentabilité prédite et la probabilité de risque élevé (`predict`, rôle 1), et les notes d'impact environnemental et social, d'alignement stratégique et de faisabilité (`extract`, rôle 3).

Le secteur, la région et le type de financement ne servent qu'à décrire les groupes, pas à les former. Les cibles `rentabilite` et `risque` sont exclues : elles sont vides pour les 500 projets en cours et simulées par une formule.

Deux remarques sur ces variables. D'abord, la rentabilité et le risque prédits sont calculés par le rôle 1 à partir du coût, de la durée, de l'année, du secteur et du pays : ils sont en partie redondants avec les trois chiffres (corrélation de 0,67 entre risque prédit et durée, 0,44 entre risque prédit et coût, −0,46 entre rentabilité prédite et durée). Ensuite, pour les 1 500 projets terminés, ce sont des prédictions sur des projets vus pendant l'apprentissage du rôle 1 ; elles collent donc mieux aux vraies valeurs que pour un projet nouveau. Cela ne gêne pas une analyse descriptive, qui ne mesure aucune performance de prédiction.

Les 8 variables n'ont aucune valeur manquante. Le coût est très asymétrique (de 1 à 37 707 millions de dollars, asymétrie 18,5) : son log10 ramène l'asymétrie à 0,06. Toutes les variables sont ensuite centrées et réduites (`StandardScaler`). Aucun projet n'est alors à plus de 4 écarts-types sur une variable : les anomalies fabriquées ne se voient pas variable par variable (un coût multiplié par 10 à 50 ne déplace le log10 que de 1,4 à 2,3 écarts-types).

La carte des corrélations montre deux blocs : la durée, le coût et les prédictions du rôle 1 d'un côté ; les trois notes du texte de l'autre, liées entre elles (0,45 à 0,64) mais presque pas aux chiffres.

Le fichier `anomalies_injectees.csv` (vérité terrain des 40 anomalies fabriquées) n'est lu nulle part dans l'analyse : il ne sert qu'à évaluer la détection, après coup (section 2.5).

## 2.3 ACP

La règle a été fixée avant le calcul : critère de Kaiser. Il retient **3 axes**, de valeurs propres 2,60, 1,93 et 1,06, qui résument **70 %** de la variance (57 % pour les deux premiers). Le quatrième axe passe sous 1 (0,85).

| Axe | Variance | Variables les plus corrélées (au moins 0,5 en valeur absolue) | Nom proposé |
|---|---|---|---|
| CP1 | 32 % | risque (+0,70), faisabilité (+0,68), durée (+0,66), impact et alignement (+0,55), rentabilité (−0,62) | gros programme long et bien documenté ↔ opération courte et rentable |
| CP2 | 24 % | alignement (+0,67), impact (+0,64), année (+0,53), durée (−0,56), risque (−0,53) | récent, court, bien noté ↔ ancien, long, risqué |
| CP3 | 13 % | coût (+0,69), année (+0,66) | taille et récence (le coût n'est pas corrigé de l'inflation) |

Les projets aux extrémités confirment ces noms. À droite de CP1 se trouvent de longs programmes (147 à 175 mois), au risque prédit supérieur à 0,93, notés 5 sur les trois notes. À gauche, de vieux prêts d'ajustement structurel de 6 à 16 mois, rentables, peu risqués et mal notés, parce que leurs documents sont très courts.

Nous attendions un axe « chiffres et prédictions » et un axe « notes du texte ». Le cercle des corrélations montre bien ces deux blocs, presque perpendiculaires, mais **en diagonale** des axes : l'ACP place son premier axe entre les deux, là où la dispersion totale est la plus grande, et chaque axe mélange donc les deux blocs. La redondance des prédictions se voit nettement : le risque prédit et la durée ont presque la même flèche.

Les projets en cours sont plutôt en haut à droite du plan (plus récents, plus chers, mieux notés : leurs documents ont plus souvent un résumé). Un bord net apparaît en haut à droite : ce sont les projets notés 5 partout, alignés par le plafond de l'échelle. Les secteurs se recouvrent largement ; on ne voit que des tendances (l'énergie à gauche de CP1, l'eau et l'éducation à droite, le transport en bas de CP2).

*Figures : `reports/role2/figures/acp_variance.png`, `acp_cercle.png`, `acp_projection.png`.*

## 2.4 Clustering

Règles fixées avant les calculs : pour K-means et Ward, le $k$ entre 2 et 8 qui maximise la silhouette ; pour le mélange gaussien, le nombre de composantes entre 1 et 8 qui minimise le BIC ; pour DBSCAN, `min_samples` = 16 (deux fois le nombre de variables) et $\varepsilon$ au coude de la courbe des distances au 16e voisin. Une solution est écartée si un groupe compte moins de 5 % des projets ou si plus de 20 % des projets sont du bruit.

| Méthode | Réglage | Groupes | Bruit | Plus petit groupe | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|---|---|---|
| K-means | k = 2 | 2 | 0 % | 47 % | **0,20** | **1,80** | **566** |
| Ward | k = 2 | 2 | 0 % | 40 % | 0,17 | 2,08 | 410 |
| DBSCAN | ε = 1,93 | 1 | 4,2 % | — | — | — | — |
| Mélange gaussien | 7 composantes | 7 | 0 % | 0,4 % | 0,05 | 2,97 | 120 |

**Il n'y a pas de groupes naturels bien séparés.** Les quatre méthodes le disent chacune à sa manière :

- la meilleure silhouette n'est que de 0,20 et elle baisse quand $k$ augmente ; l'inertie de K-means décroît régulièrement, sans coude net ;
- DBSCAN ne trouve qu'**un seul nuage dense** et met 4 % des projets à part, surtout en bordure (anciens, courts, mal notés) ;
- le mélange gaussien est **piégé par les notes entières** : ses composantes collent aux valeurs exactes d'une note (une composante ne contient que des projets notés 5 en faisabilité, une autre que des projets notés 3). Une loi normale posée sur une seule valeur a une variance quasi nulle, donc une vraisemblance énorme, ce qui fait chuter artificiellement le BIC (17 289 avec 7 composantes contre 26 737 avec 4). Le modèle suppose des variables continues ; nos notes de 1 à 5 ne le sont pas ;
- K-means et Ward ne s'accordent pas : indice de Rand ajusté de 0,29. K-means coupe le nuage le long de CP1, Ward en diagonale.

La solution retenue, par la règle, est celle de K-means avec deux groupes :

- **groupe 0, « petites opérations courtes et rentables »** (943 projets) : coût médian de 28 M$, 47 mois en moyenne, rentabilité prédite de 9,4 %, risque prédit de 0,10, notes moyennes (3,4 à 3,6) ; 16 % de projets en cours ;
- **groupe 1, « grands programmes longs, risqués et bien notés »** (1 057 projets) : coût médian de 100 M$, 84 mois, rentabilité prédite de 7,2 %, risque prédit de 0,40, notes élevées (4,3 à 4,7) ; 33 % de projets en cours.

L'année moyenne est la même (2011), et la gouvernance est le secteur le plus fréquent des deux groupes sans en dominer aucun (23 % et 16 %) : le secteur n'explique pas les groupes.

Le clustering confirme donc l'ACP : les projets s'étalent le long d'un continuum, des opérations courtes et rentables aux grands programmes longs, sans familles distinctes. Les deux groupes de K-means sont une coupe commode de ce continuum, pas des types naturels. Pour le produit, cela veut dire qu'aucun segment de projets ne demande un traitement à part dans le classement.

*Figures : `clustering_choix_k.png`, `clustering_dendrogramme.png`, `clustering_dbscan_eps.png`, `clustering_plan_acp.png` ; tableaux : `reports/role2/clustering_comparaison.csv`, `clustering_groupes.csv`.*

## 2.5 Détection d'anomalies

### Méthode

L'agent appelle `detect_anomalies` sur la sélection de l'utilisateur, qui peut ne compter que 20 projets, voire un seul. Un Isolation Forest réappris sur 20 projets n'aurait aucun sens. Le détecteur est donc appris **une fois** sur les 2 000 projets de référence (sans aucune étiquette), enregistré, puis appliqué tel quel aux projets reçus : un projet reçoit la même réponse seul ou parmi les 2 000, ce que vérifie un test.

Variables : log10 du coût, log10 de la durée et log10 du coût par mois. Elles ne demandent que les colonnes obligatoires d'une fiche projet et fonctionnent donc aussi sur un fichier importé dans l'application. Un coût ou une durée nul ou négatif est signalé d'office. Réglages : 300 arbres, graine 42.

La **contamination de 5 %** (valeur du plan commun) a été fixée avant toute mesure sur la vérité terrain ; le commit qui l'enregistre précède l'évaluation dans l'historique Git. Le taux de 2 % annoncé dans la documentation des données n'a pas été utilisé, puisqu'il vient de la vérité terrain.

Deux limites étaient connues avant l'évaluation : les 10 rentabilités extrêmes sont indétectables, puisque la rentabilité n'existe pas pour les projets en cours que l'agent évalue ; et un coût multiplié par 10 à 50 sur un petit projet peut tomber dans la plage des grands projets réels (34 projets dépassent 2 000 M$).

### Évaluation avec la vérité terrain

| Périmètre | Signalés | Vraies anomalies trouvées | Faux positifs | Précision | Rappel |
|---|---|---|---|---|---|
| 2 000 projets | 100 | 19 sur 40 | 81 | 0,19 | 0,48 |
| 2 000 projets, sans les rentabilités extrêmes | 100 | 18 sur 30 | 82 | 0,18 | 0,60 |
| 500 projets en cours (ceux de l'agent) | 24 | 9 sur 15 | 15 | 0,38 | 0,60 |

Par type : **13 durées incohérentes sur 13**, 5 coûts aberrants sur 17, 1 rentabilité extrême sur 10 (trouvée par hasard : son coût de 1 M$, le plus bas possible, est atypique en lui-même).

Les 81 faux positifs sont surtout de **vrais projets rares** : 37 durées courtes (6 à 8 mois, au bord du filtre de 6 mois appliqué à la préparation), 27 coûts extrêmes réels (de 1 à 22 000 M$) et 17 coûts par mois inhabituels. Le détecteur repère des projets **rares**, pas forcément **faux**. Pour l'agent, cela veut dire qu'un dossier signalé est à vérifier, pas à rejeter : écarter les 24 projets signalés parmi les 500 en cours retire 15 dossiers sains de la recommandation.

Chaque projet signalé reçoit une explication : la variable la plus éloignée de la médiane, en écarts interquartiles. Les premiers de la liste sont un projet terminé de 2 114 M$ sur 2 mois (durée très faible), un projet de 1 M$ sur 170 mois (coût par mois très faible, faux positif), et le projet en cours de 37 707 M$ (coût aberrant fabriqué, qui joue un rôle important dans la section 2.6).

### Sensibilité à la contamination

Mesurée après coup, à titre d'analyse seulement : `detect_anomalies` garde 5 %.

| Contamination | 1 % | 2 % | 3 % | 5 % | 8 % | 10 % |
|---|---|---|---|---|---|---|
| Signalés | 20 | 40 | 60 | 100 | 160 | 200 |
| Précision | 0,45 | 0,38 | 0,28 | 0,19 | 0,13 | 0,11 |
| Rappel | 0,23 | 0,38 | 0,43 | 0,48 | 0,50 | 0,55 |

Le compromis est net : chaque point de rappel gagné coûte de nombreux faux positifs. Au-delà de 5 %, le rappel ne progresse presque plus, car les anomalies restantes (coûts multipliés sur de petits projets, rentabilités) ne sont pas atypiques dans les variables du détecteur. Une contamination plus basse donnerait moins de fausses alertes ; le chiffre de 5 % a été gardé pour ne pas régler le seuil sur la vérité terrain.

*Tableaux : `reports/role2/anomalies_evaluation.csv`, `anomalies_sensibilite.csv`, `anomalies_detectees.csv`.*

## 2.6 Classement multicritère

### Somme pondérée et TOPSIS

Les deux méthodes sont implémentées de façon vectorisée (numpy). Elles traitent de la même façon les cas difficiles : une valeur manquante reçoit la pire valeur du critère (un risque inconnu ne doit pas faire monter un projet) ; une colonne constante ne départage personne (0,5 pour la somme pondérée ; idéal et anti-idéal confondus pour TOPSIS) ; un seul projet reçoit 0,5. Une rentabilité négative ne pose pas de problème à TOPSIS, car la division par la norme conserve l'ordre.

TOPSIS a été vérifié à la main sur 4 projets et 3 critères, avec des chiffres choisis pour que les normes tombent juste :

| Projet | Rentabilité (à max., norme 5) | Coût (à min., norme 5) | Risque (à min., norme 1) |
|---|---|---|---|
| P1 | 4 | 1 | 0,6 |
| P2 | 2 | 2 | 0 |
| P3 | 2 | 2 | 0,8 |
| P4 | 1 | 4 | 0 |

Avec les poids 0,5, 0,3 et 0,2, la matrice pondérée vaut P1 (0,40 ; 0,06 ; 0,12), P2 (0,20 ; 0,12 ; 0), P3 (0,20 ; 0,12 ; 0,16), P4 (0,10 ; 0,24 ; 0). L'idéal est (0,40 ; 0,06 ; 0) et l'anti-idéal (0,10 ; 0,24 ; 0,16). Pour P1, $d^+ = 0{,}12$ et $d^- = \sqrt{0{,}3^2 + 0{,}18^2 + 0{,}04^2} = 0{,}352$, d'où $C = 0{,}746$. Les quatre scores sont **0,746 ; 0,517 ; 0,373 ; 0,314** : la fonction redonne exactement ces valeurs (test automatique).

### Poids par AHP

`ahp_poids` calcule les poids à partir d'une matrice de comparaisons. Deux exemples vérifiés à la main :

| Matrice 3 × 3 | Poids | $\lambda_{\max}$ | RC |
|---|---|---|---|
| cohérente : A = 2 B, B = 2 C, A = 4 C | 4/7, 2/7, 1/7 | 3 | 0 |
| circulaire : A > B, B > C, C > A (tous à 3) | 1/3 chacun | 13/3 | 1,149 (rejetée) |

Pour la matrice circulaire, la symétrie donne directement le vecteur propre $(1/3, 1/3, 1/3)$ et $\lambda_{\max} = 1 + 3 + 1/3$, d'où $IC = 2/3$ et $RC = (2/3)/0{,}58 = 1{,}149$, très au-dessus du seuil de 0,1.

La matrice d'exemple des 7 critères compare, par exemple, la rentabilité à la durée avec un 5 (« nettement plus importante ») et à l'impact avec un 2. Elle donne des poids proches des poids par défaut du contrat, avec un ratio de cohérence de **0,006** :

| Critère | Rentabilité | Risque | Coût | Durée | Impact | Alignement | Faisabilité |
|---|---|---|---|---|---|---|---|
| Poids AHP | 0,265 | 0,188 | 0,087 | 0,049 | 0,162 | 0,162 | 0,087 |
| Poids par défaut | 0,25 | 0,20 | 0,10 | 0,05 | 0,15 | 0,15 | 0,10 |

*[À compléter par Zineb : justification de chaque comparaison de la matrice des 7 critères, ou matrice ajustée.]*

### Règle de choix, sensibilité et décision

Le plan prévoyait que `rank` utilise TOPSIS. Pour ne pas choisir après coup, la règle a été écrite et enregistrée **avant** toute comparaison :

- 500 projets en cours, critères calculés par `predict` et `extract` ; comme l'agent, `rank` sur les 500 projets, anomalies comprises, puis les trois premiers projets **non signalés** par `detect_anomalies` (retirer les anomalies avant le classement changerait la normalisation) ;
- 14 perturbations : chaque poids multiplié par 1,2 puis par 0,8, les autres ramenés pour que la somme reste 1 ;
- garde-fou : une méthode doit changer ses trois premiers pour au moins 5 des 7 profils franchement différents (0,70 sur un critère, 0,05 sur les autres), sinon elle est écartée, car une méthode stable parce qu'elle ignore les poids ne convient pas à une application où l'utilisateur règle les poids ;
- décision : la méthode dont les trois premiers changent le moins souvent ; si l'écart est de moins de 5 points, TOPSIS.

| | Somme pondérée | TOPSIS |
|---|---|---|
| Perturbations ± 20 % qui changent les trois premiers | 0 sur 14 | 0 sur 14 |
| 1 000 tirages où tous les poids bougent à la fois (complément) | 0 % | 3,8 % |
| Profils extrêmes qui changent les trois premiers (garde-fou) | **4 sur 7** | **7 sur 7** |

Les deux méthodes sont très stables : aucune retouche de ± 20 % ne change les trois projets recommandés. C'est le garde-fou qui tranche : avec la somme pondérée, mettre 0,70 sur le coût, l'impact ou l'alignement ne change pas les trois premiers. **La règle désigne TOPSIS**, que `rank` utilise depuis.

L'explication tient à un seul projet. Le plus grand coût des 500 projets en cours, 37 707 M$, est un coût aberrant fabriqué (signalé par `detect_anomalies`, mais gardé dans le classement comme le fait l'agent). La normalisation min-max se cale sur lui : tous les projets non signalés coûtent moins de 2 800 M$ et reçoivent presque 1 sur le critère du coût, qui ne les départage plus. Les notes qualitatives, souvent au maximum (médiane de 5 en faisabilité), laissent aussi peu de marge à la somme pondérée. TOPSIS, qui mesure des distances à l'idéal après une normalisation par la norme, reste sensible à ces critères.

Les deux méthodes ne classent pas les projets de la même façon : la corrélation de Spearman entre leurs classements des 500 projets vaut 0,74 ; 7 projets sont communs dans les 10 premiers, 29 dans les 50 premiers, et l'écart de rang médian est de 72 places. Le haut du classement est proche (même premier projet ; même deuxième, P512688, signalé comme anomalie et donc écarté de la recommandation), le milieu beaucoup moins. La corrélation de Spearman tient lieu ici d'analyse d'interprétabilité : elle dit à quel point le choix de la méthode change le résultat.

Pistes non essayées, pour ne pas ajuster la méthode après avoir vu le résultat : une somme pondérée sur le log du coût, ou des bornes de normalisation calculées sans les projets signalés.

*Figures : `rangs_topsis_somme.png`, `reactivite_profils.png` ; tableaux : `reports/role2/sensibilite_*.csv`.*

## 2.7 Les fonctions livrées

**`rank(matrice, poids)`** vérifie les poids (critères connus, positifs, somme 1), calcule le score TOPSIS, trie du meilleur au moins bon et numérote les rangs de 1 à $n$. Les égalités de score sont départagées par `id_projet`, pour que le classement ne dépende pas de l'ordre des lignes reçues et que les rangs soient uniques. Les colonnes en plus sont gardées, l'entrée n'est pas modifiée, et la sortie passe `verifier_classement` du contrat. L'application refait le classement à chaque mouvement d'un curseur de poids : `rank` prend **2,3 ms** sur 2 000 projets.

**`detect_anomalies(df)`** ajoute la colonne booléenne `anomalie`. Le détecteur est chargé une seule fois, et réappris automatiquement s'il manque ou s'il a été enregistré avec une autre version de scikit-learn.

Le module contient aussi `somme_ponderee`, `topsis`, `ahp_poids` et `score_anomalie`. Il est couvert par **37 tests** (`tests/test_mcda.py`) : cas du contrat (un seul projet, colonne constante, égalités, colonnes en plus, tableau vide, valeurs manquantes ou impossibles), vérifications à la main de TOPSIS et d'AHP, indépendance de la détection vis-à-vis de la sélection, et temps de calcul.

## 2.8 Limites

- **Variables de l'analyse non supervisée** : cinq des huit variables sont des sorties des modèles des rôles 1 et 3. Les axes et les groupes décrivent donc les projets à travers ces modèles, et la rentabilité et le risque prédits recopient en partie le coût et la durée.
- **Notes discrètes** : les notes de 1 à 5 sont traitées comme continues par l'ACP et le clustering. Le mélange gaussien en est directement victime.
- **Pas de structure en groupes** : le clustering donne un résultat négatif (silhouette de 0,20 au mieux). Il est honnête, mais il n'apporte pas de segmentation exploitable.
- **Détection** : précision de 0,19 sur les 2 000 projets (0,38 sur les projets en cours). Le détecteur repère des projets rares ; les coûts aberrants sur de petits projets et les rentabilités extrêmes lui échappent. Un détecteur qui comparerait le coût à celui des projets du même secteur et du même pays pourrait faire mieux.
- **Normalisation sur la sélection** : les deux méthodes de classement normalisent sur les projets reçus. Le score d'un projet dépend donc des autres projets de la sélection, et un seul dossier aberrant peut écraser un critère, comme on l'a vu pour la somme pondérée.
- **Sensibilité** : 14 perturbations, c'est peu ; avec 0 changement des deux côtés, la mesure de stabilité ne départage rien. Le résultat dépend aussi de `predict` et `extract` : si ces modules changent, il faut relancer le notebook 02.
- **Poids AHP** : la matrice d'exemple a été construite pour redonner les poids par défaut. Une vraie utilisation demanderait les comparaisons d'un décideur.

## Références

- Ester, M., Kriegel, H.-P., Sander, J. et Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of KDD-96*, 226-231.
- Hubert, L. et Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.
- Hwang, C.-L. et Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.
- Jolliffe, I. T. et Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065).
- Liu, F. T., Ting, K. M. et Zhou, Z.-H. (2008). Isolation forest. *Proceedings of the IEEE International Conference on Data Mining*, 413-422.
- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236-244.

## Diapositives (3 à 4)

1. **Les projets n'ont pas de familles naturelles.** Le cercle des corrélations (`acp_cercle.png`) : 3 axes pour 70 % de la variance, deux blocs de variables en diagonale, la redondance du risque prédit et de la durée. Puis `clustering_plan_acp.png` : silhouette de 0,20 au mieux, DBSCAN ne voit qu'un nuage, le mélange gaussien piégé par les notes entières. Message : un continuum, des opérations courtes et rentables aux grands programmes longs.
2. **Détecter les dossiers suspects, et mesurer les fausses alertes.** Le principe de l'Isolation Forest (un point anormal s'isole en peu de coupes), appris une fois sur les 2 000 projets. Le tableau précision / rappel : 13 durées incohérentes sur 13, mais 81 faux positifs sur 100 signalés, de vrais projets rares. Le tableau de sensibilité à la contamination, et le choix de 5 % fait avant la mesure.
3. **TOPSIS et AHP, vérifiés à la main.** Le schéma de TOPSIS (idéal, anti-idéal, distances) avec l'exemple des 4 projets (0,746 ; 0,517 ; 0,373 ; 0,314). La matrice AHP des 7 critères, ses poids et son ratio de cohérence de 0,006.
4. **Choisir la méthode par une règle écrite à l'avance.** Le tableau stabilité / réactivité : 0 changement sur 14 pour les deux méthodes, mais 4 profils sur 7 seulement pour la somme pondérée. `reactivite_profils.png` et l'explication par le coût aberrant de 37 707 M$ qui écrase la normalisation min-max. Spearman de 0,74 entre les deux classements (`rangs_topsis_somme.png`).
