# Grille de notation des critères qualitatifs

Cette grille sert à noter à la main le jeu de test et à formuler la consigne donnée au LLM local pour le jeu d'entraînement. Les deux sources de notes suivent donc les mêmes règles.

Principes communs :

- On note **uniquement le texte** (objectif de développement et résumé). L'en-tête (pays, secteur, coût, durée) n'est pas utilisé : ces chiffres sont déjà des critères du classement, les compter deux fois fausserait le classement.
- **3 = neutre** : le texte ne donne pas assez d'indices dans un sens ou dans l'autre. C'est la note attendue pour un texte vide ou très court.
- On monte au-dessus de 3 pour des indices **explicites** dans le texte, et on descend en dessous de 3 pour des signaux contraires explicites ou pour un texte qui porte sur le sujet sans rien en dire.

## `impact_env_social` : bénéfices environnementaux et sociaux attendus

| Note | Indices dans le texte |
|---|---|
| 5 | Objectif principal social ou environnemental, explicite et chiffré ou ciblé : réduction de la pauvreté, populations vulnérables, femmes, santé, accès à l'eau ou à l'énergie, climat, biodiversité, avec des bénéficiaires ou des résultats précis |
| 4 | Bénéfices sociaux ou environnementaux explicites, mais secondaires ou sans cible précise |
| 3 | Rien de clair, ou texte trop court |
| 2 | Projet purement financier, administratif ou d'infrastructure, sans aucun bénéfice social ou environnemental mentionné |
| 1 | Effets négatifs évoqués sans mesure d'atténuation (déplacement de population, pollution, déforestation) |

## `alignement_strategique` : cohérence avec les priorités de développement

| Note | Indices dans le texte |
|---|---|
| 5 | Le projet s'inscrit explicitement dans une stratégie nationale ou sectorielle, un programme de réforme du gouvernement ou un cadre international (objectifs de développement durable, accord sur le climat), et vise un enjeu prioritaire (capital humain, climat, transition énergétique, numérique, emploi) |
| 4 | Enjeu prioritaire clairement visé, ou rattachement explicite à un programme du gouvernement, mais pas les deux |
| 3 | Objectif de développement général, sans rattachement ni priorité nette, ou texte trop court |
| 2 | Objectif flou, très étroit ou ponctuel, sans lien avec une politique plus large |
| 1 | Objectif en contradiction avec les priorités affichées, ou incohérent |

## `faisabilite` : capacité du projet à être mis en œuvre comme prévu

| Note | Indices dans le texte |
|---|---|
| 5 | Composantes détaillées, montage institutionnel précis (unité de gestion du projet, agence d'exécution), suivi et évaluation, renforcement des capacités, phasage ou expérience antérieure (deuxième phase, projet pilote réussi) |
| 4 | Composantes décrites et au moins un élément de mise en œuvre (gestion, suivi, capacités) |
| 3 | Objectif seul, sans description de la mise en œuvre, ou texte trop court |
| 2 | Contexte difficile mentionné (conflit, crise, urgence, fragilité, faible capacité des institutions) avec peu d'éléments de mise en œuvre |
| 1 | Contexte très difficile et objectifs très ambitieux, sans aucun dispositif de mise en œuvre |

## Exemples

- *P171266, Agriculture Resilience and Competitiveness Project* : climat et compétitivité (impact 4) ; enjeu prioritaire, le climat, sans rattachement explicite à une stratégie nationale (alignement 4) ; quatre composantes détaillées et une unité de gestion du projet (faisabilité 5).
- *P045546, Public Finance Structural Adjustment Credit* : réformes des finances publiques, aucun bénéfice social mentionné (impact 2) ; programme de réforme du gouvernement après la guerre (alignement 4) ; contexte d'après-guerre, aucun dispositif décrit (faisabilité 2).
