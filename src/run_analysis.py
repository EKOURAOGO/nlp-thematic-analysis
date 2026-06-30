"""
run_analysis.py
---------------
Pipeline principal d'analyse thématique NLP :
  1. Chargement du corpus
  2. Prétraitement
  3. Vectorisation TF-IDF
  4. Clustering KMeans (avec sélection du k optimal)
  5. Topic Modeling LDA
  6. Évaluation et sauvegarde des résultats
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_corpus
from modeling import (
    build_tfidf_matrix,
    find_optimal_k,
    run_kmeans,
    get_cluster_top_terms,
    reduce_dimensions,
    run_lda,
    get_document_topics,
    evaluate_clustering,
    save_results,
)


def load_corpus(path: str) -> List[Dict]:
    """Charge le corpus depuis un fichier JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base = Path(__file__).parent.parent

    # ── 1. Chargement ────────────────────────────────────────────────────────
    corpus_path = base / "data" / "corpus.json"
    if not corpus_path.exists():
        print("Corpus non trouvé — génération en cours...")
        from generate_corpus import generate_corpus
        corpus = generate_corpus(n_per_theme=50)
        corpus_path.parent.mkdir(exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
    else:
        corpus = load_corpus(str(corpus_path))

    texts = [doc["text"] for doc in corpus]
    true_labels = [doc["theme"] for doc in corpus]
    true_labels_display = [doc["label"] for doc in corpus]
    ids = [doc["id"] for doc in corpus]

    print(f"Corpus chargé : {len(texts)} documents")

    # ── 2. Prétraitement ─────────────────────────────────────────────────────
    print("Prétraitement des textes...")
    processed = preprocess_corpus(texts, apply_stemming=False)

    # ── 3. Vectorisation TF-IDF ──────────────────────────────────────────────
    print("Vectorisation TF-IDF...")
    matrix, vectorizer = build_tfidf_matrix(processed, max_features=5000)
    print(f"  Matrice TF-IDF : {matrix.shape[0]} docs × {matrix.shape[1]} features")

    # ── 4. Sélection du k optimal ────────────────────────────────────────────
    print("Recherche du k optimal (scores de silhouette)...")
    silhouette_scores = find_optimal_k(matrix, k_range=range(2, 10))
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"  k optimal = {best_k} (silhouette = {silhouette_scores[best_k]})")

    # ── 5. KMeans avec k=6 (nb de thèmes vrais) ──────────────────────────────
    # On utilise k=6 pour évaluer la cohérence avec les vrais labels
    print("Clustering KMeans (k=6)...")
    km_labels, km_model = run_kmeans(matrix, n_clusters=6)
    top_terms = get_cluster_top_terms(km_model, vectorizer, n_terms=10)

    # ── 6. Réduction 2D pour visualisation ──────────────────────────────────
    print("Réduction dimensionnelle (SVD 2D)...")
    coords_2d = reduce_dimensions(matrix, n_components=2)

    # ── 7. LDA Topic Modeling ────────────────────────────────────────────────
    print("Topic Modeling LDA (6 topics)...")
    lda_model, dictionary, bow_corpus, lda_topics = run_lda(
        processed, n_topics=6, n_words=10, passes=15
    )
    lda_dominant = get_document_topics(lda_model, bow_corpus)
    print("  Topics LDA extraits.")

    # ── 8. Évaluation ────────────────────────────────────────────────────────
    eval_kmeans = evaluate_clustering(true_labels, km_labels)
    eval_lda = evaluate_clustering(true_labels, lda_dominant)
    sil_k6 = silhouette_scores.get(6, None)

    print(f"\n{'='*50}")
    print("RÉSULTATS D'ÉVALUATION")
    print(f"{'='*50}")
    print(f"KMeans ARI    : {eval_kmeans['adjusted_rand_index']} ({eval_kmeans['interpretation']})")
    print(f"LDA ARI       : {eval_lda['adjusted_rand_index']} ({eval_lda['interpretation']})")
    print(f"Silhouette k=6: {sil_k6}")
    print(f"{'='*50}\n")

    # ── 9. Sauvegarde ────────────────────────────────────────────────────────
    # Données pour la visualisation
    viz_data = [
        {
            "id": ids[i],
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "km_cluster": int(km_labels[i]),
            "lda_topic": int(lda_dominant[i]),
            "true_theme": true_labels[i],
            "true_label": true_labels_display[i],
            "text_preview": texts[i][:120] + "...",
        }
        for i in range(len(texts))
    ]

    results = {
        "n_documents": len(texts),
        "n_themes": 6,
        "evaluation": {
            "kmeans": eval_kmeans,
            "lda": eval_lda,
            "silhouette_k6": sil_k6,
        },
        "silhouette_scores": silhouette_scores,
        "kmeans_top_terms": {str(k): v for k, v in top_terms.items()},
        "lda_topics": {
            str(k): [{"word": w, "score": s} for w, s in v]
            for k, v in lda_topics.items()
        },
    }

    outputs = base / "outputs"
    outputs.mkdir(exist_ok=True)
    save_results(results, str(outputs / "analysis_results.json"))
    save_results(viz_data, str(outputs / "viz_data.json"))

    print("Résultats sauvegardés dans outputs/")
    print("  - outputs/analysis_results.json")
    print("  - outputs/viz_data.json")


if __name__ == "__main__":
    main()
