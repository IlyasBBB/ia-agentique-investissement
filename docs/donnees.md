# Jeu de données

État au 16 novembre : pas encore choisi. C'est la tâche la plus urgente, car les rôles 1, 2 et 3 en dépendent.

## Ce qu'il faut

- Au moins 500 projets, un par ligne.
- Des colonnes chiffrées (coût, durée) et catégorielles (secteur, pays).
- Une variable résultat pour la régression (rentabilité ou équivalent) et une pour la classification (niveau de risque, ou réussite et échec).
- Une licence qui autorise la réutilisation, et une source à citer.
- Si possible un texte de description par projet, utile au rôle 3.

## Pistes à examiner

- Projets de la Banque mondiale (Projects & Operations) : données ouvertes, texte de description disponible. À vérifier : présence d'une note de résultat exploitable comme cible.
- Jeux Kaggle sur des projets financés (financement participatif, startups, prêts à des projets). À vérifier : licence et qualité.
- Données simulées en dernier recours ou en complément, avec les hypothèses écrites dans `data/README.md`. Le professeur l'accepte si le mode de génération est décrit.

Aucune de ces pistes n'a encore été vérifiée. Avant de retenir un jeu, contrôler la licence, le nombre de lignes utilisables et le taux de valeurs manquantes.

## Marche à suivre

1. Télécharger le jeu dans `data/raw/` sans le modifier.
2. Écrire `scripts/preparer_donnees.py` qui lit le fichier brut, renomme les colonnes selon `docs/contrat.md` et écrit `data/processed/projets.csv`.
3. Remplir `data/README.md` : source, date, licence, description des colonnes, traitements effectués.
4. Mettre à jour `docs/contrat.md` et `src/contrat.py` si des colonnes diffèrent, puis prévenir l'équipe.
5. Partager le fichier avec les rôles 1, 2 et 3.

Si le jeu ne contient pas de rentabilité ni de risque au sens strict, choisir les variables les plus proches et expliquer ce choix dans le rapport.
