"""Réglages communs des tests.

Par défaut, les tests utilisent les fausses fonctions de src/mocks.py : ils restent rapides et
tournent sans modèle d'embeddings ni Ollama, y compris sur GitHub Actions. Pour tester avec les
vrais modules : USE_MOCKS=0 python -m pytest. La variable est fixée ici, avant que src/outils.py
ne soit importé par un test.
"""
import os

os.environ.setdefault("USE_MOCKS", "1")
