# Données

Jeu hybride : des projets réels de la Banque mondiale, avec deux cibles simulées.

> **Mise en garde.** Les colonnes `rentabilite` et `risque` sont **simulées** par une formule écrite plus bas. Ce ne sont pas des résultats réels de la Banque mondiale, qui ne publie ni rentabilité ni niveau de risque dans ces données. Les anomalies sont elles aussi fabriquées. Tout le reste (nom, pays, secteur, coût, dates, textes) est réel.

## Source

- Banque mondiale, « Projects & Operations » : https://projects.worldbank.org
- API utilisée : `https://search.worldbank.org/api/v3/projects` (pagination par `os` et `rows`)
- Licence : Creative Commons Attribution 4.0 (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Citation : World Bank, *Projects & Operations*, https://projects.worldbank.org, consulté le 16 novembre 2026, licence CC BY 4.0. Données modifiées par l'équipe (filtres, regroupements, cibles simulées).
- Date de téléchargement : 16 novembre 2026 (28 245 projets). Elle est aussi écrite dans le fichier brut, champ `date_telechargement`.

## Reproduire

```
python scripts/telecharger_donnees.py     # écrit data/raw/banque_mondiale_projets.json (52 Mo, hors Git)
python scripts/preparer_donnees.py        # écrit data/processed/ et data/docs/
python -m pytest tests/test_donnees.py -q
```

Le téléchargement ne se relance pas si le fichier brut existe (option `--force` pour le refaire). La préparation utilise la graine 42 : avec le même fichier brut, elle redonne exactement les mêmes fichiers. Un nouveau téléchargement peut changer l'échantillon, car la Banque mondiale met ses projets à jour.

## Fichiers

| Fichier | Contenu | Dans Git |
|---|---|---|
| `data/raw/banque_mondiale_projets.json` | Réponse de l'API, non modifiée | non |
| `data/processed/projets.csv` | 2 000 projets, un par ligne (0,3 Mo) | oui |
| `data/processed/anomalies_injectees.csv` | Les 40 anomalies fabriquées (vérité terrain) | oui |
| `data/docs/<id_projet>.txt` | Un document texte par projet (2 000 fichiers, 3,9 Mo) | oui |

## Filtres

Sur les 28 245 projets du brut, on garde ceux qui remplissent toutes ces conditions (12 656 projets) :

- statut `Closed` (terminé) ou `Active` (en cours) ; les projets abandonnés ou en préparation sont écartés ;
- objectif de développement (PDO) d'au moins 80 caractères ;
- coût d'au moins 1 million de dollars ;
- dates d'approbation et de clôture valides, durée entre 6 et 180 mois ;
- secteur principal, pays et région connus.

On tire ensuite au hasard 1 500 projets terminés et 500 projets en cours.

## Colonnes de `projets.csv`

| Colonne | Type | Origine | Description |
|---|---|---|---|
| `id_projet` | texte | réel | Identifiant Banque mondiale, par exemple `P123456` |
| `nom` | texte | réel | Nom du projet (en anglais) |
| `secteur` | catégorie | réel, regroupé | Un des 11 secteurs ci-dessous |
| `pays` | catégorie | réel | Nom court du pays en anglais (`countryshortname`). Quelques projets régionaux ont une zone à la place, par exemple `Africa` |
| `region` | catégorie | réel, traduit | Région de la Banque mondiale |
| `cout_total` | nombre | réel | Millions de dollars. Coût total du projet (`lendprojectcost`), ou à défaut montant engagé par la Banque (`totalamt`) |
| `duree_mois` | entier | réel | Mois entre l'approbation et la clôture prévue |
| `annee_approbation` | entier | réel | Année d'approbation par le conseil |
| `type_financement` | catégorie | réel, simplifié | `IBRD` (prêt aux pays à revenu intermédiaire), `IDA` (pays les plus pauvres), `Dons`, `Autre`. Si plusieurs types, le premier de cette liste |
| `statut` | catégorie | réel | `termine` ou `en_cours` |
| `rentabilite` | nombre | **simulé** | Cible de la régression, en %. Vide pour les projets en cours |
| `risque` | catégorie | **simulé** | Cible de la classification : `faible`, `moyen`, `eleve`. Vide pour les projets en cours |

Les projets en cours sont les dossiers à évaluer par l'agent : leurs cibles sont vides.

`impact_env_social`, `alignement_strategique` et `faisabilite` ne sont pas dans ce fichier : le rôle 3 les extrait des documents de `data/docs/`.

L'API ne fournit pas l'instrument de prêt : `type_financement` le remplace.

## Regroupement des secteurs

Le secteur retenu est le grand secteur (`major_sectors`) qui a le plus gros pourcentage dans le projet. Les préfixes `FY17 - ` et `(Historic)` sont retirés.

| Secteur | Grands secteurs de la Banque mondiale |
|---|---|
| Agriculture | Agriculture, Fishing and Forestry |
| Énergie | Energy and Extractives ; Electric Power & Other Energy ; Energy and Mineral Resources ; Oil & Gas ; Mining |
| Transport | Transportation |
| Eau et assainissement | Water, Sanitation and Waste Management |
| Santé | Health ; Health and other social services |
| Éducation | Education |
| Finance | Financial Sector |
| Numérique | Information and Communications Technologies ; Digital Development |
| Gouvernance | Public Administration ; Economic Policy |
| Protection sociale | Social Protection ; Social Sustainability and Inclusion |
| Industrie et commerce | Industry, Trade and Services ; Industry ; Tourism ; Private Sector Development |

Les projets des autres grands secteurs (Multisector, Urban Development, Environment) sont écartés.

## Simulation des cibles

Calculée sur les 2 000 projets tirés, avant l'injection des anomalies. Notations :

- `z_cout` : log10 de `cout_total`, centré et réduit (moyenne 0, écart-type 1) ;
- `z_duree` : `duree_mois`, centré et réduit ;
- `N(0, s)` : bruit gaussien de moyenne 0 et d'écart-type `s`, graine 42.

### Rentabilité (en %)

```
rentabilite = 9
            + effet_secteur
            + effet_pays                      # un tirage N(0, 1.5) par pays
            + effet_financement
            - 0.8 * z_cout
            + 1.0 * z_cout  si secteur = Énergie ou Transport
            - 1.2 * z_duree
            - 0.5 * z_duree²
            + N(0, 1.8)                       # bruit propre à chaque projet
```

arrondie à une décimale.

| Secteur | Effet | Secteur | Effet |
|---|---|---|---|
| Énergie | +2,5 | Eau et assainissement | −0,5 |
| Numérique | +2,0 | Éducation | −1,0 |
| Transport | +1,5 | Santé | −1,5 |
| Finance | +1,0 | Gouvernance | −2,0 |
| Industrie et commerce | +1,0 | Protection sociale | −2,0 |
| Agriculture | 0 | | |

Effet du financement : IBRD +1, IDA 0, Dons −1, Autre 0.

### Risque

```
score = - 0.6 * rentabilite centrée et réduite
        + 0.5 * z_duree
        + 0.4 * z_cout
        + effet_region                        # un tirage N(0, 0.4) par région
        + N(0, 0.8)
```

Le score est coupé en trois avec deux seuils calculés sur les projets terminés : les 40 % de scores les plus bas donnent `faible`, les 40 % suivants `moyen`, les 20 % les plus hauts `eleve` (600, 600 et 300 projets).

### Ce que l'on peut espérer d'un modèle

Contrôle fait par le script (modèle linéaire, 1 200 projets terminés pour l'entraînement, 300 pour le test, anomalies non encore injectées) : R² de 0,55 sur la rentabilité, 58 % de bonnes classes de risque contre 43 % en répondant toujours la classe la plus fréquente. Le bruit empêche tout modèle d'être parfait : c'est voulu.

## Anomalies injectées

40 projets (2 %) tirés au hasard reçoivent une valeur fabriquée, après la simulation des cibles :

| Type | Modification | Nombre |
|---|---|---|
| `cout_aberrant` | `cout_total` multiplié par un nombre tiré entre 10 et 50 | 17 |
| `duree_incoherente` | `duree_mois` remplacée par 1, 2 ou 3 mois | 13 |
| `rentabilite_extreme` | `rentabilite` remplacée par une valeur entre 35 et 60 %, positive ou négative (projets terminés seulement) | 10 |

Les types alternent dans cet ordre. Un projet en cours tiré pour `rentabilite_extreme` reçoit `cout_aberrant` à la place.

`anomalies_injectees.csv` donne pour chacune : `id_projet`, `type_anomalie`, `colonne`, `valeur_initiale`, `valeur_modifiee`. Ce fichier sert à évaluer la détection d'anomalies (rôle 2). Il ne faut pas le donner au modèle. `projets.csv` ne contient aucune colonne qui signale les anomalies.

Pour le rôle 1 : retirer ces 40 projets de l'entraînement, ou vérifier que le modèle y résiste.

## Documents texte

Chaque fichier de `data/docs/` contient : nom, identifiant, pays, région, secteur, coût, durée, année, statut, puis l'objectif de développement (PDO) et le résumé (`project_abstract`) recopiés tels quels de la Banque mondiale, en anglais. 526 projets n'ont pas de résumé : le fichier indique « Non disponible. ». Le coût et la durée de l'en-tête sont ceux de `projets.csv`, donc modifiés pour les projets anormaux.

## Limites

- Les cibles suivent une formule : un modèle retrouve la formule, pas la réalité économique. Aucune conclusion sur les vrais projets ne peut en être tirée.
- Les textes ne dépendent pas des cibles : les notes du rôle 3 n'aident pas à prédire `rentabilite` ni `risque`.
- La durée va jusqu'à la date de clôture publiée par la Banque, qui peut avoir été révisée en cours de projet.
- Le coût n'est pas corrigé de l'inflation (projets approuvés de 1987 à 2026).
