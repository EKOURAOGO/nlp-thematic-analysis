"""
preprocessing.py
----------------
Nettoyage et prétraitement du texte pour l'analyse thématique.
Fonctions pures, sans état, avec typage complet et docstrings.
"""

import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer

# Télécharger les ressources NLTK si absentes
for resource in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_STEMMER = FrenchStemmer()
_STOPWORDS_FR = set(stopwords.words("french"))

# Stopwords supplémentaires spécifiques au domaine
_EXTRA_STOPWORDS = {
    "projet", "programme", "action", "actions", "dispositif", "objectif",
    "objectifs", "cadre", "territoire", "partenaires", "notamment", "afin",
    "permettre", "développer", "mettre", "oeuvre", "également",
    "ainsi", "cet", "cette", "dont", "lors", "niveau",
}
STOPWORDS = _STOPWORDS_FR | _EXTRA_STOPWORDS


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut : minuscule, suppression ponctuation/chiffres,
    normalisation des espaces.

    Args:
        text: Texte brut d'entrée.

    Returns:
        Texte nettoyé.
    """
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenise un texte nettoyé en liste de mots.

    Args:
        text: Texte nettoyé.

    Returns:
        Liste de tokens.
    """
    return text.split()


def remove_stopwords(tokens: List[str], extra: Optional[set] = None) -> List[str]:
    """
    Supprime les mots vides d'une liste de tokens.

    Args:
        tokens: Liste de tokens.
        extra: Ensemble optionnel de mots vides supplémentaires.

    Returns:
        Liste de tokens filtrés.
    """
    sw = STOPWORDS | (extra or set())
    return [t for t in tokens if t not in sw and len(t) > 2]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Applique le stemming français (Snowball) à une liste de tokens.

    Args:
        tokens: Liste de tokens.

    Returns:
        Liste de racines (stems).
    """
    return [_STEMMER.stem(t) for t in tokens]


def preprocess(
    text: str,
    apply_stemming: bool = False,
    extra_stopwords: Optional[set] = None,
) -> str:
    """
    Pipeline complet de prétraitement : nettoyage -> tokenisation ->
    suppression stopwords -> (stemming optionnel) -> reconstruction.

    Args:
        text: Texte brut d'entrée.
        apply_stemming: Si True, applique le stemming.
        extra_stopwords: Mots vides supplémentaires à retirer.

    Returns:
        Texte prétraité prêt pour la vectorisation.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens, extra=extra_stopwords)
    if apply_stemming:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_corpus(
    texts: List[str],
    apply_stemming: bool = False,
) -> List[str]:
    """
    Applique le pipeline de prétraitement à une liste de textes.

    Args:
        texts: Liste de textes bruts.
        apply_stemming: Si True, applique le stemming.

    Returns:
        Liste de textes prétraités.
    """
    return [preprocess(t, apply_stemming=apply_stemming) for t in texts]
