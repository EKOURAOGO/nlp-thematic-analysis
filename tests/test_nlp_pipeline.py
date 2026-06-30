"""
test_nlp_pipeline.py
--------------------
Tests unitaires du pipeline NLP :
preprocessing, vectorisation, clustering, LDA.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import clean_text, tokenize, remove_stopwords, preprocess, preprocess_corpus
from modeling import (
    build_tfidf_matrix,
    find_optimal_k,
    run_kmeans,
    get_cluster_top_terms,
    reduce_dimensions,
    run_lda,
    get_document_topics,
    evaluate_clustering,
)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_clean_text_lowercase(self):
        assert clean_text("Bonjour MONDE") == "bonjour monde"

    def test_clean_text_removes_numbers(self):
        assert "2024" not in clean_text("En 2024, le projet démarre.")

    def test_clean_text_removes_punctuation(self):
        result = clean_text("Santé, bien-être ! Oui.")
        for p in [".", ",", "!", "-"]:
            assert p not in result

    def test_clean_text_normalizes_spaces(self):
        result = clean_text("Mot1   Mot2  Mot3")
        assert "  " not in result

    def test_tokenize_splits_correctly(self):
        tokens = tokenize("formation professionnelle emploi")
        assert len(tokens) == 3
        assert "formation" in tokens

    def test_remove_stopwords_removes_fr_stopwords(self):
        tokens = ["le", "la", "formation", "est", "importante"]
        filtered = remove_stopwords(tokens)
        for sw in ["le", "la", "est"]:
            assert sw not in filtered
        assert "formation" in filtered

    def test_remove_stopwords_min_length(self):
        tokens = ["à", "de", "en", "formation"]
        filtered = remove_stopwords(tokens)
        assert "formation" in filtered
        # Short tokens removed
        for t in ["à", "de", "en"]:
            assert t not in filtered

    def test_preprocess_returns_string(self):
        result = preprocess("Le projet de formation professionnelle.")
        assert isinstance(result, str)

    def test_preprocess_corpus_length(self):
        texts = ["Texte un sur la santé publique.",
                 "Projet emploi formation insertion.",
                 "Logement social quartiers prioritaires."]
        result = preprocess_corpus(texts)
        assert len(result) == 3

    def test_preprocess_empty_string(self):
        result = preprocess("")
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Modeling tests
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "formation professionnelle emploi compétences qualification",
    "santé publique prévention soins hôpital médecin",
    "logement social HLM loyer quartier urbain",
    "énergie renouvelable écologie biodiversité recyclage",
    "numérique internet compétences alphabétisation fracture",
    "éducation scolaire jeunes lycée décrochage réussite",
    "insertion professionnelle chômage reconversion emploi",
    "vaccination dépistage épidémiologie santé mentale",
    "rénovation urbaine habitat quartiers prioritaires logement",
    "solaire éolien émissions carbone transition écologique",
    "dématérialisation services ligne administration numérique",
    "orientation scolaire bourses étudiantes citoyenneté",
]


class TestModeling:

    def test_tfidf_matrix_shape(self):
        matrix, vectorizer = build_tfidf_matrix(SAMPLE_TEXTS)
        assert matrix.shape[0] == len(SAMPLE_TEXTS)
        assert matrix.shape[1] > 0

    def test_tfidf_matrix_not_empty(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        assert matrix.nnz > 0

    def test_find_optimal_k_returns_dict(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        scores = find_optimal_k(matrix, k_range=range(2, 5))
        assert isinstance(scores, dict)
        assert len(scores) == 3

    def test_find_optimal_k_all_floats(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        scores = find_optimal_k(matrix, k_range=range(2, 4))
        for v in scores.values():
            assert isinstance(v, float)

    def test_kmeans_labels_length(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        labels, _ = run_kmeans(matrix, n_clusters=3)
        assert len(labels) == len(SAMPLE_TEXTS)

    def test_kmeans_labels_range(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        labels, _ = run_kmeans(matrix, n_clusters=4)
        assert min(labels) >= 0
        assert max(labels) <= 3

    def test_get_cluster_top_terms_keys(self):
        matrix, vectorizer = build_tfidf_matrix(SAMPLE_TEXTS)
        labels, km = run_kmeans(matrix, n_clusters=3)
        top = get_cluster_top_terms(km, vectorizer, n_terms=5)
        assert set(top.keys()) == {0, 1, 2}
        for terms in top.values():
            assert len(terms) == 5

    def test_reduce_dimensions_shape(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        reduced = reduce_dimensions(matrix, n_components=2)
        assert reduced.shape == (len(SAMPLE_TEXTS), 2)

    def test_lda_returns_correct_number_of_topics(self):
        _, _, _, topics = run_lda(SAMPLE_TEXTS, n_topics=3, passes=3)
        assert len(topics) == 3

    def test_lda_topics_have_words(self):
        _, _, _, topics = run_lda(SAMPLE_TEXTS, n_topics=3, passes=3)
        for topic_words in topics.values():
            assert len(topic_words) > 0
            word, score = topic_words[0]
            assert isinstance(word, str)
            assert 0 <= score <= 1

    def test_get_document_topics_length(self):
        _, _, bow_corpus, _ = run_lda(SAMPLE_TEXTS, n_topics=3, passes=3)
        # Re-run LDA to get model
        lda, _, bow_corpus, _ = run_lda(SAMPLE_TEXTS, n_topics=3, passes=3)
        dominant = get_document_topics(lda, bow_corpus)
        assert len(dominant) == len(SAMPLE_TEXTS)

    def test_evaluate_clustering_ari_range(self):
        true = ["a", "a", "b", "b", "c", "c"]
        pred = np.array([0, 0, 1, 1, 2, 2])
        result = evaluate_clustering(true, pred)
        assert 0 <= result["adjusted_rand_index"] <= 1

    def test_evaluate_clustering_perfect(self):
        true = ["a", "a", "b", "b"]
        pred = np.array([0, 0, 1, 1])
        result = evaluate_clustering(true, pred)
        assert result["adjusted_rand_index"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
