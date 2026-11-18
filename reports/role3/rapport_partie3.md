# Partie 3 : traitement du texte et recherche de passages

Brouillon du 18 novembre 2026. Tous les chiffres viennent du module livré (`src/nlp.py`) et des notebooks `notebooks/role3/01_exploration.ipynb`, `02_extract.ipynb` et `03_retrieve.ipynb`. Les résultats de bout en bout avec l'agent seront ajoutés après l'intégration du 22 novembre.

Le rôle 3 fournit deux fonctions à l'agent. `extract(document)` lit le texte d'un projet et lui donne trois notes de 1 à 5 : impact environnemental et social, alignement stratégique, faisabilité. Ces notes entrent dans le classement multicritère du rôle 2. `retrieve(question, k)` retrouve dans les 2 000 documents les passages les plus proches d'une question, pour que la recommandation de l'agent cite les dossiers au lieu de les paraphraser.

## 3.1 Fondements théoriques

### Représenter un texte par des nombres

Un modèle d'apprentissage ne lit pas des mots, il manipule des vecteurs. Toute la question est de choisir comment passer de l'un à l'autre.

Le sac de mots est la représentation la plus simple. On fixe un vocabulaire, puis on décrit chaque texte par le nombre de fois où chaque mot y apparaît. L'ordre des mots est perdu : « le projet finance l'école » et « l'école finance le projet » ont le même vecteur. Ces vecteurs sont très longs (un coefficient par mot du vocabulaire) et presque entièrement nuls.

TF-IDF corrige le principal défaut du sac de mots : les mots fréquents partout, comme *project* ou *development* dans nos documents, ne disent rien d'un texte en particulier. Le poids d'un mot dans un texte est le produit de deux termes :

- la fréquence du terme (TF), son nombre d'occurrences dans le texte, que nous prenons sous forme logarithmique (1 + log tf) pour qu'un mot répété dix fois ne compte pas dix fois plus ;
- la fréquence inverse de document (IDF), log(N / df), où N est le nombre de textes et df le nombre de textes qui contiennent le mot. Un mot présent partout a un IDF proche de zéro. La bibliothèque scikit-learn, que nous utilisons, en prend une variante lissée, log((1 + N) / (1 + df)) + 1, qui évite les divisions par zéro.

Chaque vecteur est ensuite normé à la longueur 1. On peut ajouter des paires de mots consécutifs (bigrammes) au vocabulaire pour garder un peu d'ordre : *private sector* ou *capacity building* ont un sens que leurs mots isolés n'ont pas (Salton et Buckley, 1988 ; Manning et al., 2008).

Les plongements (embeddings) remplacent ces vecteurs creux par des vecteurs denses de quelques centaines de nombres, appris par un réseau de neurones de façon que deux textes de même sens soient proches, même s'ils n'ont aucun mot en commun. Les premiers modèles donnaient un vecteur par mot (Mikolov et al., 2013). Les modèles récents, construits sur l'architecture Transformer présentée en partie 4, donnent un vecteur par phrase. Nous utilisons `all-MiniLM-L6-v2`, un petit modèle de la famille Sentence-BERT (Reimers et Gurevych, 2019) : 22 millions de paramètres, des vecteurs de 384 nombres, assez léger pour tourner sur le processeur d'un portable. Il a été entraîné à rapprocher des paires de phrases équivalentes (questions et réponses, paraphrases), ce qui convient à la recherche. Sa limite : il ne lit que les 256 premiers jetons d'un texte, soit environ 180 mots.

### Recherche d'information

Chercher des passages revient à calculer un score de proximité entre la question et chaque passage, puis à trier. Avec des vecteurs normés, le score usuel est la similarité cosinus, qui se réduit à un produit scalaire. Pour 8 293 passages, c'est un seul produit matrice-vecteur : pas besoin de base de données spécialisée.

Pour mesurer la qualité d'un moteur, il faut des requêtes de test et la liste des documents pertinents pour chacune. On en tire trois mesures :

- la précision@k, part de documents pertinents parmi les k premiers résultats ;
- le rappel@k, part des documents pertinents connus retrouvés dans les k premiers ;
- le MRR (rang réciproque moyen), moyenne de 1 / rang du premier résultat pertinent : 1 si le premier résultat est toujours bon, 0,5 s'il arrive en moyenne en deuxième position.

Juger chaque document pour chaque requête est impossible sur 2 000 documents. La méthode des campagnes TREC est la mise en commun des candidats (pooling) : on réunit les premiers résultats de plusieurs moteurs et on ne juge que ceux-là (Voorhees et Harman, 2005). Un document pertinent qu'aucun moteur n'a trouvé est compté comme non pertinent : le rappel est surestimé, mais la comparaison reste équitable si chaque moteur fournit autant de candidats.

### RAG et place du rôle 3

La génération augmentée par recherche (RAG, Lewis et al., 2020), présentée en partie 4, sépare deux tâches : un moteur de recherche trouve des passages, puis le LLM rédige à partir d'eux. `retrieve` est ce moteur. Sa qualité compte directement pour l'agent : si les passages sont mauvais, le LLM cite des dossiers hors sujet ou n'a rien à citer.

### Mesurer l'accord entre deux notateurs

Pour `extract`, une note juste à un point près vaut mieux qu'une note fausse de trois points, et l'exactitude seule ne voit pas cette différence. Nous utilisons le kappa de Cohen pondéré quadratique (Cohen, 1968). Il compare l'accord observé entre deux séries de notes à l'accord qu'on obtiendrait par hasard, en pénalisant chaque désaccord par le carré de l'écart. Il vaut 0 pour un accord dû au hasard et 1 pour un accord parfait. Il sert ici à la fois à comparer un modèle aux notes de référence et à comparer le LLM à ces mêmes notes.

Les intervalles de confiance sont obtenus par bootstrap (Efron et Tibshirani, 1993) : on retire plusieurs milliers de fois, avec remise, autant de requêtes ou de documents qu'il y en a, on recalcule l'écart entre deux modèles, et on garde les quantiles 2,5 % et 97,5 %. Si l'intervalle ne contient pas 0, l'écart ne tient pas au choix de l'échantillon.

## 3.2 Les documents et leur préparation

Chaque projet a un document texte (`data/docs/<id_projet>.txt`) : un en-tête rédigé à partir des chiffres (pays, secteur, coût, durée), puis l'objectif de développement (PDO) et le résumé, recopiés de la Banque mondiale, en anglais (licence CC BY 4.0). Nous n'analysons que l'objectif et le résumé : l'en-tête répète des chiffres que les rôles 1 et 2 utilisent déjà.

L'analyse exploratoire a guidé le prétraitement :

| Constat | Conséquence |
|---|---|
| 526 documents (26 %) sans résumé, surtout des projets anciens : il ne reste qu'une ou deux phrases | `extract` donne la note neutre 3 à tout texte de moins de 15 mots utiles |
| 285 objectifs (14 %) entièrement en majuscules, mots coupés par un tiret de fin de ligne (« UTILIZA- TION ») | nettoyage avant tout traitement |
| 211 mots en médiane, de 11 à 1 259 | des textes courts : une note fondée sur quelques indices explicites |
| 434 318 mots, 15 675 différents, dont 6 957 vus une seule fois (loi de Zipf) | mots trop rares filtrés dans les modèles appris |
| les mots les plus fréquents sont du jargon (*project, component, development, management*) | pondération TF-IDF et retrait des mots vides |
| chaque secteur a son vocabulaire (*electricity, power* pour l'énergie, *poor, vulnerable, households* pour la protection sociale) | les textes contiennent de l'information exploitable |

Pour la recherche, chaque texte est découpé en fenêtres de trois phrases qui avancent de deux, si bien que deux passages voisins partagent une phrase et qu'une idée n'est pas coupée en deux. Une « phrase » de plus de 60 mots (une énumération sans point) est coupée, donc un passage fait au plus 180 mots, ce qui correspond à la limite de lecture du modèle de plongements. On obtient 8 293 passages, de 74 mots en médiane. Le nom du projet est ajouté au texte indexé, pas au passage rendu à l'agent : il aide à situer un passage isolé.

## 3.3 Recherche de passages (`retrieve`)

### Méthode

Trois moteurs sont comparés :

- TF-IDF, modèle de référence : mots et bigrammes, mots vides anglais retirés, similarité cosinus ;
- plongements, modèle avancé : `all-MiniLM-L6-v2`, similarité cosinus ;
- hybride : α × plongements + (1 − α) × TF-IDF, chaque score ramené entre 0 et 1 pour chaque question.

`retrieve` rend un seul passage par projet, pour que l'agent voie k projets différents. Aucun de ces moteurs n'apprend à partir d'étiquettes : le seul réglage est α, fait sur des requêtes séparées de celles du test.

Nous avons écrit 40 requêtes en anglais, couvrant les 11 secteurs, volontairement reformulées pour ne pas recopier les mots des documents (« getting more girls to attend secondary school », « helping farmers cope with drought »). Quinze servent au réglage, 25 au test. Les projets attendus ont été établis par mise en commun des candidats : les 10 premiers projets de TF-IDF, les 10 premiers des plongements et jusqu'à 10 projets dont le titre contient un synonyme choisi à l'avance, soit 886 candidats. Chacun a été jugé à la main avec une règle fixe : pertinent si son objectif principal ou une composante majeure traite directement le sujet de la requête. Cela donne 489 projets pertinents, de 4 à 21 par requête.

### Résultats

Sur les 15 requêtes de réglage, α = 0,6 obtient la meilleure précision@3 (0,933), un seul bon résultat devant les plongements seuls (0,911). Sur les 25 requêtes de test :

| Moteur | Précision@3 | Précision@5 | MRR | Rappel@10 |
|---|---|---|---|---|
| TF-IDF | 0,55 | 0,54 | 0,72 | 0,48 |
| Plongements | **0,75** | **0,78** | **0,93** | **0,65** |
| Hybride (α = 0,6) | 0,71 | 0,67 | 0,88 | 0,62 |

Les plongements font mieux que TF-IDF de 0,20 en précision@3, avec un intervalle de confiance de [+0,08 ; +0,33]. Ils gagnent sur 12 requêtes et en perdent 3. L'hybride ne fait pas mieux que les plongements seuls (−0,04, intervalle [−0,19 ; +0,11]) : le gain vu sur le réglage ne s'est pas reproduit sur le test. À égalité, nous gardons le modèle le plus simple, les plongements, avec TF-IDF en repli automatique si le modèle ne peut pas être chargé.

Côté vitesse, une question prend environ 20 ms avec les plongements et 5 ms avec TF-IDF. Le calcul des vecteurs des 8 293 passages prend environ 4 minutes, une seule fois : ils sont enregistrés sur disque et rechargés en une dizaine de secondes.

### Analyse des erreurs

Les plongements l'emportent quand la question est une reformulation. Pour « helping farmers cope with drought », leurs trois premiers projets parlent d'adaptation à la sécheresse et d'agriculture résiliente au climat, alors que TF-IDF remonte un projet d'enseignement supérieur qui partage quelques mots avec la question. TF-IDF résiste mieux sur un terme technique précis : pour « highway construction between major cities », le mot *highway* lui suffit, tandis que les plongements proposent des projets routiers généraux (entretien, routes urbaines). Les erreurs des plongements sont presque toujours des voisins de sujet. Pour « reducing deaths of mothers and newborns », un projet de nutrition de la petite enfance arrive en deuxième position.

### Intégration avec l'agent

L'agent ne doit citer que les projets qu'il analyse. Au début, il cherchait dans tout le corpus puis filtrait : avec k = 3, il ne lui restait en moyenne que 0,9 passage, et rien du tout pour 17 requêtes sur 40. Une première correction consistait à demander 50 passages puis à filtrer. Elle ne réglait pas un second problème : les documents importés dans l'application ne sont pas dans l'index, et `retrieve` ne les voyait jamais.

Le contrat a donc été étendu le 18 novembre : `retrieve(question, k, documents)` accepte les documents à fouiller. Si ces documents sont déjà dans l'index de `data/docs/` (même identifiant, même texte), la recherche réutilise cet index en ne gardant que ces projets, sans aucun calcul supplémentaire : 24 ms par question pour les 500 projets en cours. Seuls les documents inconnus, comme des fichiers importés, reçoivent un petit index construit à la volée et gardé en mémoire. Sur les 40 requêtes, les trois premiers passages sont les mêmes qu'avec l'astuce des 50 passages, mais pour k = 10 l'astuce en rendait moins de 10 dans 4 cas sur 40, alors que la nouvelle recherche en rend toujours 10. Le LLM de l'agent écrit ses questions en anglais, la langue des documents et du modèle de plongements.

## 3.4 Extraction des critères (`extract`)

### Construire une vérité terrain

Personne n'a jamais noté ces projets sur nos trois critères. Nous avons donc d'abord écrit une grille de notation (`reports/role3/grille_notation.md`) qui définit chaque note avec des indices concrets. Par exemple, une faisabilité de 5 demande des composantes détaillées, un montage institutionnel précis, un suivi et une évaluation, du renforcement de capacités ou une expérience antérieure ; une faisabilité de 2 correspond à un contexte de crise ou de fragilité avec peu d'éléments de mise en œuvre. La note 3 signifie « pas assez d'indices », et c'est la note de tout texte trop court.

Avant toute annotation et tout modèle, les documents ont été séparés une fois pour toutes, avec une répartition équilibrée par secteur (graine 42) :

| Jeu | Documents | Notes | Usage |
|---|---|---|---|
| Entraînement | 640 | LLM local | apprendre la régression logistique |
| Validation | 160 | à la main, selon la grille | régler et choisir le modèle |
| Test | 100 | à la main, selon la grille | mesure finale, une seule fois |

Les 640 notes d'entraînement viennent d'un LLM local, `qwen2.5:3b` (Qwen Team, 2024), lancé avec Ollama. Il reçoit la grille traduite en anglais, répond en JSON avec trois entiers, à température 0 pour qu'il réponde toujours pareil. Une réponse invalide est redemandée jusqu'à trois fois. Il lui faut environ 11 secondes par document, près de trois heures pour 900 documents. C'est acceptable une fois, pas pour l'agent, qui devrait attendre six heures pour les 2 000 projets. Le LLM sert donc à produire des étiquettes, et le modèle livré doit être rapide.

### Le LLM note-t-il comme la grille ?

Sur les 260 documents notés à la main, l'accord entre le LLM et les notes de référence est modéré : kappa de 0,38 en impact, 0,51 en alignement, 0,52 en faisabilité. Surtout, le LLM n'emploie pas l'échelle de la même façon. Il met 5 en alignement à plus de la moitié des documents, contre environ un quart pour les notes de référence, donne plus de 2 et de 3 en impact, et met surtout 4 en faisabilité là où la référence met surtout 5.

Cette différence a changé notre méthode. Une première évaluation choisissait le modèle sur une validation notée par le LLM : elle retenait TF-IDF, c'est-à-dire le modèle qui imitait le mieux le LLM, et ce modèle s'est révélé moins bon que de simples mots-clés face aux notes de référence. Nous avons donc noté la validation à la main, pour que le choix du modèle se fasse sur la même échelle que la mesure finale.

### Les modèles comparés

Cinq candidats, tous réglés sur la validation :

1. Mots-clés (référence simple) : une liste de mots positifs et négatifs par critère, écrite à partir de la grille. Le solde (positifs moins négatifs) donne la note par des seuils réglés sur la validation.
2. TF-IDF + régression logistique apprise sur les 640 notes du LLM, un modèle par critère, vocabulaire appris sur l'entraînement seulement, régularisation C réglée sur la validation.
3. TF-IDF recalé : le même modèle, dont la note attendue (moyenne des notes pondérée par les probabilités) est convertie en note par des seuils qui redonnent, sur la validation, la répartition des notes de référence.
4. Combinaison : moyenne du solde de mots-clés et de la note attendue TF-IDF, chacun centré-réduit sur la validation, puis recalée de la même façon.
5. TF-IDF appris sur les 160 notes de référence, évalué par validation croisée à 5 plis.

La règle de choix a été fixée avant d'ouvrir le test : le meilleur kappa moyen sur la validation, et le plus simple si un modèle plus simple fait presque aussi bien (moins de 0,02 d'écart). La combinaison obtient 0,649, devant le TF-IDF appris sur les notes de référence (0,629), les mots-clés (0,602), le TF-IDF recalé (0,528) et le TF-IDF appris sur les notes du LLM (0,520). C'est le modèle livré.

### Résultats sur le test

| Modèle | Impact | Alignement | Faisabilité | Kappa moyen | Exactitude | Erreur moyenne |
|---|---|---|---|---|---|---|
| Classe majoritaire | 0,04 | 0,08 | 0,07 | 0,06 | 41 % | 0,87 |
| LLM (`qwen2.5:3b`) | 0,47 | 0,52 | 0,57 | 0,52 | 50 % | 0,62 |
| TF-IDF (notes du LLM) | 0,46 | 0,40 | 0,62 | 0,49 | 51 % | 0,64 |
| TF-IDF recalé | 0,61 | 0,41 | 0,63 | 0,55 | 54 % | 0,55 |
| TF-IDF (notes de référence) | 0,49 | 0,54 | 0,62 | 0,55 | 55 % | 0,56 |
| Mots-clés | 0,58 | 0,55 | 0,74 | 0,62 | 58 % | 0,49 |
| **Combinaison (livrée)** | **0,69** | **0,55** | **0,74** | **0,66** | **59 %** | **0,46** |

Le modèle livré reste le meilleur sur le test, ce qui montre que son avance sur la validation n'était pas un hasard. Il fait mieux que le LLM dont il a appris les étiquettes (+0,14, intervalle [+0,05 ; +0,23]) et que TF-IDF seul (+0,17, intervalle [+0,09 ; +0,25]). Son avance sur les mots-clés seuls (+0,04, intervalle [−0,03 ; +0,10]) n'est en revanche pas démontrée : sur 100 documents, on ne peut pas affirmer que la partie apprise améliore la règle écrite, même si elle semble aider en impact. Entre 95 et 97 % des notes sont justes à un point près.

`extract` traite les 2 000 documents en une dizaine de secondes, justifications comprises. Le modèle est appris au premier appel en deux secondes environ, puis gardé sur disque.

### Interprétabilité

Le modèle est lisible de bout en bout. La partie mots-clés est une liste écrite à la main. Pour la partie TF-IDF, chaque mot a un poids sur la note attendue, calculé à partir des coefficients de la régression logistique. La plupart des mots qui font monter les notes sont cohérents avec la grille, alors que le modèle ne l'a jamais vue : *water*, *biodiversity*, *poor*, *children* pour l'impact ; *component*, *management*, *institutional*, *capacity* pour la faisabilité. D'autres trahissent les habitudes du LLM : *wastewater* et *road* font baisser l'impact, *private sector* et *financial* font baisser l'alignement. La liste de mots-clés, écrite à partir de la grille, corrige une partie de ces biais, ce qui explique que la combinaison fasse mieux que chacune de ses parties.

Chaque note est accompagnée d'une justification : la phrase du document qui pèse le plus dans le sens de la note, selon les mêmes poids. Pour le projet de résilience agricole en Bosnie-Herzégovine (P171266), la faisabilité de 5 est justifiée par la phrase qui décrit l'unité de gestion du projet et le spécialiste du suivi et de l'évaluation.

### Analyse des erreurs

Sur 300 notes de test, 12 sont fausses de deux points ou plus. Elles ont quatre causes :

1. Un mot ambigu. Dans un projet de modernisation de l'administration fiscale (P099376), le mot-clé *climate* est compté dans *investment climate*, le climat des affaires. L'impact passe à 4 au lieu de 2.
2. L'effet de longueur. Un document de 1 205 mots sur la Somalie (P179775) accumule les mots de mise en œuvre (solde de 24). Les mots *conflict*, *crisis* et *emergency* sont bien comptés, mais noyés dans la masse : la faisabilité monte à 5 alors que le texte décrit une famine.
3. Des signaux absents du lexique. Un projet de pêche au Liberia (P172012) se dit *overly ambitious*, avec des travaux *no longer feasible* ; un appui budgétaire en Afghanistan (P107921) évoque l'incertitude et des élections. Aucun de ces mots n'est dans la liste négative, et TF-IDF ne les a pas appris sur 640 exemples.
4. Des textes courts notés 1. Deux textes de 16 et 42 mots, sans aucun mot-clé, obtiennent un score plus bas que tout ce qui a été vu en validation et tombent à 1, une note que l'annotateur n'a jamais donnée.

## 3.5 Discussion et limites

- Un seul annotateur. La grille, la liste de mots-clés et les notes de référence viennent de la même personne. Les mots-clés partagent sa lecture de la grille, ce qui les avantage par rapport au LLM. Faire noter une partie des documents par un second annotateur donnerait l'accord entre humains, le plafond réaliste pour un modèle.
- Des notes extrapolées. Le modèle donne la note 1 à 108 notes sur 6 000 (1,8 %), surtout pour des textes courts, alors qu'aucune note de référence ne vaut 1. Nous avons choisi de ne pas corriger ce défaut après avoir ouvert le test, pour que le score annoncé reste propre. La correction est simple pour une version suivante : limiter les notes à celles utilisées par l'annotateur et diviser le solde de mots-clés par la longueur du texte, ce qui traiterait aussi l'effet de longueur.
- Des échantillons petits. 25 requêtes de test pour `retrieve`, 160 et 100 documents pour `extract` : les intervalles de confiance restent larges.
- Des jugements de pertinence incomplets. La mise en commun des candidats surestime le rappel de `retrieve`, et une règle de pertinence plus souple ferait monter tous les scores.
- Un petit LLM. `qwen2.5:3b` tourne sur un portable mais applique la grille moins bien qu'un grand modèle. De meilleures étiquettes d'apprentissage profiteraient directement à la partie TF-IDF.
- Des notes sans lien avec la rentabilité. La rentabilité et le risque du jeu de données sont simulés à partir des chiffres seulement : les notes qualitatives servent au classement multicritère, elles n'aident pas à prédire le risque.

Le principal enseignement de cette partie tient à la qualité des étiquettes. Un modèle appris reproduit les habitudes de celui qui l'a étiqueté : le choisir sur des étiquettes du LLM menait au mauvais modèle. Une petite validation notée à la main, quelques règles explicites tirées de la grille et un recalage de l'échelle ont suffi à faire mieux que le LLM lui-même, pour un coût de calcul négligeable.

## Références

- Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213-220.
- Efron, B. et Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Manning, C. D., Raghavan, P. et Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Mikolov, T. et al. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
- Qwen Team (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.
- Reimers, N. et Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.
- Salton, G. et Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.
- Voorhees, E. M. et Harman, D. K. (dir.) (2005). *TREC: Experiment and Evaluation in Information Retrieval*. MIT Press.
- Banque mondiale, *Projects & Operations*, https://projects.worldbank.org, consulté le 16 novembre 2026, licence CC BY 4.0.

## Diapositives (3 à 4)

1. Le texte en nombres. Sac de mots, TF-IDF, plongements, sur un même exemple de phrase. Chiffres du corpus : 2 000 documents, 26 % sans résumé, 8 293 passages.
2. `retrieve`. Tableau des trois moteurs (précision@3 : 0,55, 0,75, 0,71) et un exemple de reformulation (« helping farmers cope with drought ») où les plongements trouvent les bons projets et TF-IDF non. Une phrase sur l'agent : de 0,9 à 3 passages utiles grâce à `retrieve(question, 50)`.
3. `extract` sans vérité terrain. La grille, la séparation 640 / 160 / 100, et la figure LLM contre notes de référence : le LLM n'utilise pas l'échelle comme l'annotateur.
4. Résultats et erreurs. Kappa sur le test (majoritaire 0,06, LLM 0,52, mots-clés 0,62, modèle livré 0,66), les mots les plus influents, et deux erreurs typiques (*investment climate*, texte de 1 205 mots).
