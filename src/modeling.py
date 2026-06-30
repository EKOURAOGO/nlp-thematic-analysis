"""
modeling.py
-----------
Clustering KMeans + Topic Modeling LDA sur corpus de textes.
Toutes les fonctions retournent des structures sérialisables (dict/list)
pour faciliter l'affichage dans Streamlit.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    classification_report,
)
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Vectorisation
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_matrix(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple:
    """
    Construit la matrice TF-IDF à partir d'une liste de textes prétraités.

    Args:
        texts: Textes prétraités.
        max_features: Nombre maximum de features.
        ngram_range: Plage de n-grammes.

    Returns:
        (matrice TF-IDF sparse, vectorizer fitté)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# KMeans Clustering
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(
    matrix,
    k_range: range = range(2, 12),
) -> Dict:
    """
    Calcule le score de silhouette pour chaque k dans k_range
    afin d'identifier le nombre optimal de clusters.

    Args:
        matrix: Matrice TF-IDF.
        k_range: Plage de valeurs de k à tester.

    Returns:
        Dict {k: silhouette_score}
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        try:
            score = silhouette_score(matrix, labels, sample_size=min(1000, matrix.shape[0]))
            scores[k] = round(float(score), 4)
        except Exception:
            scores[k] = 0.0
    return scores


def run_kmeans(
    matrix,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    Applique KMeans sur la matrice TF-IDF.

    Args:
        matrix: Matrice TF-IDF.
        n_clusters: Nombre de clusters.
        random_state: Graine aléatoire.

    Returns:
        (labels, modèle KMeans fitté)
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(matrix)
    return labels, km


def get_cluster_top_terms(
    km: KMeans,
    vectorizer: TfidfVectorizer,
    n_terms: int = 10,
) -> Dict[int, List[str]]:
    """
    Retourne les n termes les plus représentatifs de chaque cluster
    à partir des centres du modèle KMeans.

    Args:
        km: Modèle KMeans fitté.
        vectorizer: Vectorizer TF-IDF fitté.
        n_terms: Nombre de termes par cluster.

    Returns:
        Dict {cluster_id: [terme1, terme2, ...]}
    """
    feature_names = vectorizer.get_feature_names_out()
    top_terms = {}
    for i, center in enumerate(km.cluster_centers_):
        top_indices = center.argsort()[::-1][:n_terms]
        top_terms[i] = [feature_names[j] for j in top_indices]
    return top_terms


# ─────────────────────────────────────────────────────────────────────────────
# Réduction dimensionnelle pour visualisation
# ─────────────────────────────────────────────────────────────────────────────

def reduce_dimensions(
    matrix,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Réduit la matrice TF-IDF à 2 dimensions via SVD tronquée (LSA).
    Équivalent à PCA pour les matrices sparse.

    Args:
        matrix: Matrice TF-IDF sparse.
        n_components: Nombre de dimensions cibles.
        random_state: Graine aléatoire.

    Returns:
        Matrice 2D normalisée.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(matrix)
    return normalize(reduced)


# ─────────────────────────────────────────────────────────────────────────────
# LDA Topic Modeling
# ─────────────────────────────────────────────────────────────────────────────

def run_lda(
    texts_preprocessed: List[str],
    n_topics: int = 6,
    n_words: int = 10,
    passes: int = 10,
    random_state: int = 42,
) -> Tuple[LdaModel, corpora.Dictionary, List, Dict[int, List[Tuple[str, float]]]]:
    """
    Entraîne un modèle LDA sur le corpus prétraité.

    Args:
        texts_preprocessed: Textes prétraités.
        n_topics: Nombre de topics.
        n_words: Nombre de mots par topic à retourner.
        passes: Nombre de passes d'entraînement.
        random_state: Graine aléatoire.

    Returns:
        (modèle LDA, dictionnaire, corpus BoW, topics {id: [(mot, score)]})
    """
    tokenized = [text.split() for text in texts_preprocessed]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    bow_corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

    lda = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=passes,
        random_state=random_state,
        alpha="auto",
        eta="auto",
    )

    topics = {}
    for topic_id in range(n_topics):
        words = lda.show_topic(topic_id, topn=n_words)
        topics[topic_id] = [(word, round(float(score), 4)) for word, score in words]

    return lda, dictionary, bow_corpus, topics


def get_document_topics(
    lda: LdaModel,
    bow_corpus: List,
) -> List[int]:
    """
    Retourne le topic dominant pour chaque document.

    Args:
        lda: Modèle LDA fitté.
        bow_corpus: Corpus en Bag-of-Words.

    Returns:
        Liste des topics dominants (un entier par document).
    """
    dominant = []
    for bow in bow_corpus:
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        best = max(dist, key=lambda x: x[1])
        dominant.append(best[0])
    return dominant


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clustering(
    true_labels: List[str],
    predicted_labels: np.ndarray,
) -> Dict:
    """
    Calcule les métriques d'évaluation du clustering.

    Args:
        true_labels: Labels réels (noms de thèmes).
        predicted_labels: Labels prédits par KMeans.

    Returns:
        Dict avec ARI et silhouette (si disponible).
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return {
        "adjusted_rand_index": round(float(ari), 4),
        "interpretation": (
            "excellent (> 0.7)" if ari > 0.7
            else "bon (> 0.4)" if ari > 0.4
            else "modéré (> 0.2)" if ari > 0.2
            else "faible"
        ),
    }


def save_results(results: Dict, output_path: str) -> None:
    """
    Sauvegarde les résultats d'analyse au format JSON.

    Args:
        results: Dictionnaire de résultats.
        output_path: Chemin du fichier de sortie.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
