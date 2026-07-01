"""
app.py — Dashboard Analyse Thématique NLP
Politiques publiques françaises · TF-IDF · KMeans · LDA
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocessing import preprocess_corpus
from modeling import (
    build_tfidf_matrix,
    run_kmeans,
    get_cluster_top_terms,
    reduce_dimensions,
    run_lda,
    get_document_topics,
    find_optimal_k,
)
from generate_corpus import generate_corpus

# ── Configuration page ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Analyse Thématique NLP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Fond global */
.stApp { background-color: #0f1117; }

/* Métriques */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #2d3550;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* Titres */
h1 { color: #e2e8f0 !important; font-weight: 800 !important; }
h2, h3 { color: #cbd5e1 !important; font-weight: 600 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #141624 !important; border-right: 1px solid #2d3550; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Tabs */
button[data-baseweb="tab"] { color: #94a3b8 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }

/* Cards */
.info-card {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #2d3550;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Palette de couleurs ───────────────────────────────────────────────────────

COLORS = ["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa", "#fb923c"]
THEME_LABELS = {
    "insertion_professionnelle": "Insertion pro.",
    "sante_publique": "Santé publique",
    "transition_ecologique": "Transition éco.",
    "inclusion_numerique": "Inclusion num.",
    "logement_social": "Logement social",
    "education_jeunesse": "Éducation",
}

# ── Chargement / traitement (cached) ─────────────────────────────────────────

@st.cache_data
def load_corpus():
    path = Path(__file__).parent / "data" / "corpus.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return generate_corpus(n_per_theme=50)


@st.cache_data
def run_pipeline(n_clusters: int, n_topics: int):
    corpus = load_corpus()
    texts = [d["text"] for d in corpus]
    labels_true = [d["theme"] for d in corpus]
    labels_display = [d["label"] for d in corpus]

    processed = preprocess_corpus(texts)
    matrix, vectorizer = build_tfidf_matrix(processed, max_features=5000)
    km_labels, km_model = run_kmeans(matrix, n_clusters=n_clusters)
    top_terms = get_cluster_top_terms(km_model, vectorizer, n_terms=12)
    coords = reduce_dimensions(matrix, n_components=2)
    _, _, bow_corpus, lda_topics = run_lda(processed, n_topics=n_topics, passes=15)
    lda_labels = get_document_topics(
        __import__("gensim").models.LdaModel.load(str(Path(__file__).parent / "tmp_lda"))
        if False else
        _run_lda_model(processed, n_topics),
        bow_corpus,
    )
    sil_scores = find_optimal_k(matrix, k_range=range(2, 10))

    return {
        "corpus": corpus,
        "processed": processed,
        "matrix": matrix,
        "km_labels": km_labels,
        "top_terms": top_terms,
        "coords": coords,
        "lda_topics": lda_topics,
        "lda_labels": lda_labels,
        "labels_true": labels_true,
        "labels_display": labels_display,
        "sil_scores": sil_scores,
    }


def _run_lda_model(processed, n_topics):
    from gensim import corpora
    from gensim.models import LdaModel
    tokenized = [t.split() for t in processed]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    bow = [dictionary.doc2bow(t) for t in tokenized]
    lda = LdaModel(corpus=bow, id2word=dictionary, num_topics=n_topics,
                   passes=15, random_state=42, alpha="auto", eta="auto")
    return lda, bow


@st.cache_data
def get_pipeline_data(n_clusters: int, n_topics: int):
    corpus = load_corpus()
    texts = [d["text"] for d in corpus]
    labels_true = [d["theme"] for d in corpus]
    labels_display = [d["label"] for d in corpus]

    processed = preprocess_corpus(texts)
    matrix, vectorizer = build_tfidf_matrix(processed, max_features=5000)
    km_labels, km_model = run_kmeans(matrix, n_clusters=n_clusters)
    top_terms = get_cluster_top_terms(km_model, vectorizer, n_terms=12)
    coords = reduce_dimensions(matrix, n_components=2)

    lda, bow_corpus, lda_labels = _run_lda_full(processed, n_topics)
    from modeling import run_lda as _rl
    _, _, _, lda_topics = _rl(processed, n_topics=n_topics, passes=15)
    sil_scores = find_optimal_k(matrix, k_range=range(2, 10))

    return {
        "corpus": corpus, "processed": processed,
        "km_labels": km_labels, "top_terms": top_terms,
        "coords": coords, "lda_topics": lda_topics,
        "lda_labels": lda_labels, "labels_true": labels_true,
        "labels_display": labels_display, "sil_scores": sil_scores,
    }


def _run_lda_full(processed, n_topics):
    from gensim import corpora
    from gensim.models import LdaModel
    tokenized = [t.split() for t in processed]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    bow = [dictionary.doc2bow(t) for t in tokenized]
    lda = LdaModel(corpus=bow, id2word=dictionary, num_topics=n_topics,
                   passes=15, random_state=42, alpha="auto", eta="auto")
    dominant = []
    for b in bow:
        dist = lda.get_document_topics(b, minimum_probability=0.0)
        dominant.append(max(dist, key=lambda x: x[1])[0])
    return lda, bow, dominant


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:24px 0 8px 0">
  <h1 style="margin:0;font-size:2rem">🔍 Analyse Thématique NLP</h1>
  <p style="color:#64748b;margin:4px 0 0 0;font-size:0.95rem">
    Politiques publiques françaises · TF-IDF · KMeans · LDA · 300 documents · 6 thèmes
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    n_clusters = st.slider("Nombre de clusters KMeans", 2, 10, 6)
    n_topics   = st.slider("Nombre de topics LDA", 2, 10, 6)
    st.divider()
    st.markdown("### 📖 À propos")
    st.caption("Pipeline NLP non supervisé pour la classification automatique de textes de politiques publiques.")
    st.caption("**Stack :** scikit-learn · Gensim · NLTK · Plotly")

# ── Chargement ────────────────────────────────────────────────────────────────

with st.spinner("Traitement du corpus..."):
    data = get_pipeline_data(n_clusters, n_topics)

corpus        = data["corpus"]
km_labels     = data["km_labels"]
top_terms     = data["top_terms"]
coords        = data["coords"]
lda_topics    = data["lda_topics"]
lda_labels    = data["lda_labels"]
labels_true   = data["labels_true"]
labels_display = data["labels_display"]
sil_scores    = data["sil_scores"]

# ── KPIs ──────────────────────────────────────────────────────────────────────

from sklearn.metrics import adjusted_rand_score
ari_km  = round(adjusted_rand_score(labels_true, km_labels), 3)
ari_lda = round(adjusted_rand_score(labels_true, lda_labels), 3)
best_sil_k = max(sil_scores, key=sil_scores.get)
best_sil_v = sil_scores[best_sil_k]

c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Documents", len(corpus))
c2.metric("🏷️ Thèmes", 6)
c3.metric("📊 ARI KMeans", ari_km)
c4.metric("📊 ARI LDA", ari_lda)

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Carte des documents", "📦 Clusters KMeans", "🧩 Topics LDA", "📈 Évaluation"])

# ── Tab 1 : Scatter plot 2D ───────────────────────────────────────────────────

with tab1:
    st.subheader("Projection 2D du corpus (SVD)")
    df_viz = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "theme": labels_display,
        "cluster": [f"Cluster {k}" for k in km_labels],
        "preview": [d["text"][:80] + "..." for d in corpus],
    })
    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.scatter(
            df_viz, x="x", y="y", color="theme",
            hover_data={"x": False, "y": False, "preview": True},
            color_discrete_sequence=COLORS,
            title="Coloration par thème réel",
        )
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#141624",
            font_color="#cbd5e1", legend_title_text="Thème",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        fig.update_traces(marker=dict(size=6, opacity=0.75))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = px.scatter(
            df_viz, x="x", y="y", color="cluster",
            hover_data={"x": False, "y": False, "preview": True},
            color_discrete_sequence=COLORS,
            title=f"Coloration par cluster KMeans (k={n_clusters})",
        )
        fig2.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#141624",
            font_color="#cbd5e1", legend_title_text="Cluster",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        fig2.update_traces(marker=dict(size=6, opacity=0.75))
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2 : KMeans top termes ─────────────────────────────────────────────────

with tab2:
    st.subheader(f"Top termes par cluster KMeans (k={n_clusters})")

    df_counts = pd.DataFrame({
        "Cluster": [f"Cluster {k}" for k in range(n_clusters)],
        "Documents": [int((km_labels == k).sum()) for k in range(n_clusters)],
    })

    col1, col2 = st.columns([1, 2])
    with col1:
        fig_bar = px.bar(
            df_counts, x="Documents", y="Cluster", orientation="h",
            color="Documents", color_continuous_scale="Blues",
            title="Taille des clusters",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#141624",
            font_color="#cbd5e1", showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        n_cols = min(3, n_clusters)
        cluster_cols = st.columns(n_cols)
        for k in range(n_clusters):
            col = cluster_cols[k % n_cols]
            terms = top_terms.get(k, [])
            col.markdown(f"""
<div class="info-card">
  <p style="color:#60a5fa;font-weight:700;margin:0 0 8px 0;font-size:0.9rem">Cluster {k}</p>
  <p style="color:#94a3b8;font-size:0.75rem;margin:0 0 8px 0">{int((km_labels == k).sum())} docs</p>
  {"".join(f'<span class="badge" style="background:#1e3a5f;color:#60a5fa">{t}</span>' for t in terms[:6])}
</div>
""", unsafe_allow_html=True)

# ── Tab 3 : LDA topics ────────────────────────────────────────────────────────

with tab3:
    st.subheader(f"Topics LDA ({n_topics} topics · 15 passes)")

    n_cols = min(3, n_topics)
    topic_cols = st.columns(n_cols)

    for tid in range(n_topics):
        col = topic_cols[tid % n_cols]
        words = lda_topics.get(tid, [])
        words_data = words if isinstance(words[0], (list, tuple)) else [(w["word"], w["score"]) for w in words]

        fig = go.Figure(go.Bar(
            x=[s for _, s in words_data[:8]],
            y=[w for w, _ in words_data[:8]],
            orientation="h",
            marker_color=COLORS[tid % len(COLORS)],
        ))
        fig.update_layout(
            title=f"Topic {tid}",
            paper_bgcolor="#141624", plot_bgcolor="#1a1f30",
            font_color="#cbd5e1", height=250,
            margin=dict(t=35, b=5, l=5, r=5),
            yaxis=dict(autorange="reversed"),
        )
        col.plotly_chart(fig, use_container_width=True)

# ── Tab 4 : Évaluation ────────────────────────────────────────────────────────

with tab4:
    st.subheader("Métriques d'évaluation")

    col_a, col_b = st.columns(2)

    with col_a:
        df_sil = pd.DataFrame({
            "k": list(sil_scores.keys()),
            "Silhouette": list(sil_scores.values()),
        })
        fig_sil = px.line(
            df_sil, x="k", y="Silhouette", markers=True,
            title="Score de silhouette par nombre de clusters",
            color_discrete_sequence=["#60a5fa"],
        )
        fig_sil.add_vline(x=best_sil_k, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"k optimal = {best_sil_k}")
        fig_sil.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#141624",
            font_color="#cbd5e1", margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with col_b:
        st.markdown("""
<div class="info-card">
  <p style="color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px 0">
    Adjusted Rand Index (ARI)
  </p>
  <p style="color:#64748b;font-size:0.8rem;margin:0 0 16px 0">
    Mesure la cohérence entre les clusters prédits et les vraies thématiques.
    0 = aléatoire · 1 = parfait
  </p>
""", unsafe_allow_html=True)
        for label, val, color in [
            ("KMeans", ari_km, "#60a5fa"),
            ("LDA", ari_lda, "#34d399"),
        ]:
            pct = max(0, min(1, val))
            st.markdown(f"""
  <div style="margin-bottom:14px">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
      <span style="color:#cbd5e1;font-size:0.85rem">{label}</span>
      <span style="color:{color};font-weight:700">{val}</span>
    </div>
    <div style="background:#1e2130;border-radius:999px;height:8px">
      <div style="background:{color};width:{pct*100:.0f}%;height:8px;border-radius:999px"></div>
    </div>
  </div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
