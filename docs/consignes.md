# Consignes du professeur et livrables

## Le cours

« Introduction aux techniques d'apprentissage automatique » :

1. Introduction, panorama des paradigmes d'apprentissage.
2. Apprentissage supervisé : régression (Ridge, Lasso, ElasticNet), classification binaire et multiclasse, évaluation, optimisation des hyperparamètres.
3. Apprentissage non supervisé : réduction de dimension, clustering (partition, hiérarchique, densité, modèle probabiliste), détection d'anomalies.
4. Semi-supervisé et renforcement.
5. NLP : représentation du texte, tâches classiques.
6. IA générative : LLM, pré-entraînement, SFT, RLHF, prompt engineering, RAG, IA agentique.

Sujet choisi par le groupe : IA agentique pour la décision d'investissement multicritère, évaluation et sélection de projets d'investissement.

## Consignes générales

- Partie théorique présentant les fondements de tous les concepts, méthodes et algorithmes utilisés.
- Prétraitement et analyse exploratoire des données obligatoires.
- Comparaison de modèles, dont un modèle de référence simple et d'autres plus avancés.
- Prévention des fuites de données : séparer entraînement, validation et test avant tout prétraitement.
- Interprétabilité obligatoire avec au moins une méthode adaptée : SHAP, importance par permutation ou analyse détaillée des erreurs.
- Données : privilégier les jeux ouverts, sinon des données synthétiques générées par l'équipe. Préciser la source et décrire la préparation.
- Solution finale automatisée et déployée si possible (Streamlit, Flask, API).
- Dépôt Git avec le code, les données ou les scripts pour les obtenir, et un README décrivant installation, exécution et reproduction des résultats.
- Python et bibliothèques adaptées : scikit-learn, LightGBM, XGBoost, PyTorch, Hugging Face, Gymnasium ou autres.

## Livrables (date limite : 30 novembre 2026)

1. Rapport écrit : contexte, problématique, objectifs, fondements théoriques, méthodologie, résultats et discussion.
2. Vidéo de 15 à 20 minutes avec démonstration des fonctionnalités et des résultats.
3. Présentation PowerPoint.
4. Code source et fichiers nécessaires à l'exécution et à la reproduction des expériences.
5. Données utilisées, sources et description des traitements.
6. Déploiement de la solution, ou à défaut une démonstration permettant de tester son fonctionnement.

## Ce que cela implique pour le rôle 4

- Interprétabilité côté agent : analyse détaillée des erreurs sur les scénarios de test (`reports/tests_agent.md`).
- Modèle de référence côté agent : comparer l'agent à un enchaînement fixe des outils sans LLM (même ordre à chaque fois). Cela montre ce que l'agent apporte ou non.
- Déploiement : Ollama ne tourne pas sur l'hébergement Streamlit gratuit. La démonstration se fait en local et sera filmée, ce que la consigne accepte.
- Partie théorique à rédiger : LLM et pré-entraînement, IA agentique, appel d'outils.
