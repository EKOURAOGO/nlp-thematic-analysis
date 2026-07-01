"""
app.py — Analyse Thématique NLP
Dashboard pro · Design inspiré Linear/Vercel
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocessing import preprocess_corpus
from modeling import (
    build_tfidf_matrix, run_kmeans, get_cluster_top_terms,
    reduce_dimensions, run_lda, get_document_topics, find_optimal_k,
)
from generate_corpus import generate_corpus

st.set_page_config(page_title="Analyse Thématique · NLP", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0A0B0E !important;
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px; }
[data-testid="stSidebar"] {
    background-color: #111318 !important;
    border-right: 1px solid #1C1F28 !important;
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; font-family: 'Inter', sans-serif !important; }
.app-header { padding: 0 0 2rem 0; border-bottom: 1px solid #1C1F28; margin-bottom: 2rem; }
.app-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #6366F1; margin-bottom: 0.5rem; }
.app-title { font-size: 2rem; font-weight: 700; color: #F1F5F9;
    line-height: 1.15; letter-spacing: -0.02em; }
.app-subtitle { font-size: 0.88rem; color: #64748B; margin-top: 0.35rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin-bottom: 2rem; }
.kpi-card {
    background: #111318; border: 1px solid #1C1F28;
    border-top: 2px solid #6366F1; border-radius: 10px;
    padding: 1.1rem 1.4rem;
}
.kpi-label { font-size: 0.68rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #F1F5F9;
    font-variant-numeric: tabular-nums; line-height: 1; }
.kpi-sub { font-size: 0.72rem; color: #475569; margin-top: 0.3rem; }
.section-title {
    font-size: 0.72rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 1.5rem 0 0.9rem 0; padding-bottom: 0.5rem;
    border-bottom: 1px solid #1C1F28;
}
.cluster-card {
    background: #111318; border: 1px solid #1C1F28;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
}
.cluster-num { font-size: 0.65rem; font-weight: 700; color: #6366F1;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.cluster-count { font-size: 0.7rem; color: #475569; margin-bottom: 0.55rem; }
.tag-wrap { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.tag { background: #1C1F28; color: #94A3B8; border: 1px solid #2D3142;
    border-radius: 5px; padding: 2px 7px; font-size: 0.68rem; font-weight: 500; }
.eval-row { margin-bottom: 1.1rem; }
.eval-label { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
.eval-name { font-size: 0.82rem; color: #CBD5E1; font-weight: 500; }
.eval-score { font-size: 0.82rem; color: #818CF8; font-weight: 700; }
.bar-track { background: #1C1F28; border-radius: 999px; height: 5px; overflow: hidden; }
.bar-fill { height: 5px; border-radius: 999px; }
[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1C1F28 !important; }
[data-baseweb="tab"] { background: transparent !important; color: #475569 !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.55rem 1.1rem !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #E2E8F0 !important; border-bottom: 2px solid #6366F1 !important;
    background: transparent !important; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#6366F1","#34D399","#F59E0B","#F87171","#A78BFA","#FB923C"]
ACCENT  = "#6366F1"
LAYOUT  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111318",
               font=dict(family="Inter", color="#94A3B8", size=11),
               margin=dict(t=32, b=12, l=12, r=12),
               xaxis=dict(gridcolor="#1C1F28", linecolor="#1C1F28"),
               yaxis=dict(gridcolor="#1C1F28", linecolor="#1C1F28"))
THEME_LABELS = {
    "insertion_professionnelle":"Insertion pro.","sante_publique":"Santé publique",
    "transition_ecologique":"Transition éco.","inclusion_numerique":"Inclusion num.",
    "logement_social":"Logement social","education_jeunesse":"Éducation",
}

@st.cache_data
def load_corpus():
    p = Path(__file__).parent / "data" / "corpus.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return generate_corpus(n_per_theme=50)

@st.cache_data
def run_pipeline(n_clusters, n_topics):
    corpus = load_corpus()
    texts  = [d["text"] for d in corpus]
    labels_true    = [d["theme"] for d in corpus]
    labels_display = [THEME_LABELS.get(d["theme"], d["theme"]) for d in corpus]
    processed = preprocess_corpus(texts)
    matrix, vectorizer = build_tfidf_matrix(processed, max_features=5000)
    km_labels, km_model = run_kmeans(matrix, n_clusters=n_clusters)
    top_terms = get_cluster_top_terms(km_model, vectorizer, n_terms=10)
    coords    = reduce_dimensions(matrix, n_components=2)
    from gensim import corpora as gcorp
    from gensim.models import LdaModel
    tokenized  = [t.split() for t in processed]
    dictionary = gcorp.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    bow = [dictionary.doc2bow(t) for t in tokenized]
    lda = LdaModel(corpus=bow, id2word=dictionary, num_topics=n_topics,
                   passes=15, random_state=42, alpha="auto", eta="auto")
    lda_topics = {i: lda.show_topic(i, topn=8) for i in range(n_topics)}
    lda_dom    = [max(lda.get_document_topics(b, minimum_probability=0.0),
                      key=lambda x: x[1])[0] for b in bow]
    sil_scores = find_optimal_k(matrix, k_range=range(2, 10))
    return dict(corpus=corpus, km_labels=km_labels, top_terms=top_terms,
                coords=coords, lda_topics=lda_topics, lda_dom=lda_dom,
                labels_true=labels_true, labels_display=labels_display,
                sil_scores=sil_scores)

with st.sidebar:
    st.markdown("### Paramètres")
    n_clusters = st.slider("Clusters KMeans", 2, 10, 6)
    n_topics   = st.slider("Topics LDA", 2, 10, 6)

with st.spinner("Traitement…"):
    D = run_pipeline(n_clusters, n_topics)

corpus=D["corpus"]; km_labels=D["km_labels"]; top_terms=D["top_terms"]
coords=D["coords"]; lda_topics=D["lda_topics"]; lda_dom=D["lda_dom"]
labels_true=D["labels_true"]; labels_display=D["labels_display"]; sil_scores=D["sil_scores"]
ari_km  = round(adjusted_rand_score(labels_true, km_labels), 3)
ari_lda = round(adjusted_rand_score(labels_true, lda_dom),   3)
best_k  = max(sil_scores, key=sil_scores.get)

st.markdown(f"""
<div class="app-header">
  <div class="app-eyebrow">NLP · Apprentissage non supervisé</div>
  <div class="app-title">Analyse Thématique<br>Politiques Publiques</div>
  <div class="app-subtitle">TF-IDF · KMeans · LDA · 300 documents · 6 thèmes</div>
</div>
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">Documents</div><div class="kpi-value">{len(corpus)}</div><div class="kpi-sub">6 thèmes équilibrés</div></div>
  <div class="kpi-card"><div class="kpi-label">ARI KMeans</div><div class="kpi-value">{ari_km}</div><div class="kpi-sub">Adjusted Rand Index</div></div>
  <div class="kpi-card"><div class="kpi-label">ARI LDA</div><div class="kpi-value">{ari_lda}</div><div class="kpi-sub">Topic modeling</div></div>
  <div class="kpi-card"><div class="kpi-label">k optimal</div><div class="kpi-value">{best_k}</div><div class="kpi-sub">Silhouette {sil_scores[best_k]:.3f}</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Projection 2D","Clusters KMeans","Topics LDA","Évaluation"])

with tab1:
    df = pd.DataFrame(dict(x=coords[:,0], y=coords[:,1], theme=labels_display,
                           cluster=[f"Cluster {k}" for k in km_labels],
                           preview=[d["text"][:90]+"…" for d in corpus]))
    cl, cr = st.columns(2, gap="large")
    for col, color_col, title, leg in [
        (cl, "theme",   "Thèmes réels",              "Thème"),
        (cr, "cluster", f"Clusters KMeans (k={n_clusters})", "Cluster"),
    ]:
        fig = px.scatter(df, x="x", y="y", color=color_col,
                         hover_data={"x":False,"y":False,"preview":True},
                         color_discrete_sequence=PALETTE, title=title)
        fig.update_layout(**LAYOUT, title_font_size=12, title_font_color="#94A3B8",
                          legend_title_text=leg, legend_font_color="#94A3B8")
        fig.update_traces(marker=dict(size=6, opacity=0.8,
                                      line=dict(width=0.4, color="#0A0B0E")))
        col.plotly_chart(fig, use_container_width=True)

with tab2:
    ca, cb = st.columns([1,2], gap="large")
    with ca:
        st.markdown('<div class="section-title">Taille des clusters</div>', unsafe_allow_html=True)
        df_sz = pd.DataFrame(dict(C=[f"C{k}" for k in range(n_clusters)],
                                   n=[int((km_labels==k).sum()) for k in range(n_clusters)]))
        fig = go.Figure(go.Bar(x=df_sz["n"], y=df_sz["C"], orientation="h",
                               marker=dict(color=ACCENT, opacity=0.85, cornerradius=4),
                               text=df_sz["n"], textposition="outside",
                               textfont=dict(color="#94A3B8", size=10)))
        fig.update_layout(**LAYOUT, height=260, showlegend=False,
                          xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        st.markdown('<div class="section-title">Top termes par cluster</div>', unsafe_allow_html=True)
        cols = st.columns(min(3, n_clusters))
        for k in range(n_clusters):
            terms = top_terms.get(k, [])[:8]
            tags  = "".join(f'<span class="tag">{t}</span>' for t in terms)
            cols[k % len(cols)].markdown(f"""
<div class="cluster-card">
  <div class="cluster-num">Cluster {k}</div>
  <div class="cluster-count">{int((km_labels==k).sum())} documents</div>
  <div class="tag-wrap">{tags}</div>
</div>""", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">Distribution des mots par topic</div>', unsafe_allow_html=True)
    cols = st.columns(min(3, n_topics))
    for tid in range(n_topics):
        ws = [(w, float(s)) for w, s in lda_topics.get(tid, [])[:7]]
        fig = go.Figure(go.Bar(
            x=[s for _,s in ws], y=[w for w,_ in ws], orientation="h",
            marker=dict(color=PALETTE[tid%len(PALETTE)], opacity=0.85, cornerradius=3)))
        fig.update_layout(**LAYOUT, height=210, showlegend=False, title=f"Topic {tid}",
                          title_font_size=12, title_font_color="#94A3B8",
                          yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                          xaxis_showgrid=False)
        cols[tid%len(cols)].plotly_chart(fig, use_container_width=True)

with tab4:
    ce1, ce2 = st.columns(2, gap="large")
    with ce1:
        st.markdown('<div class="section-title">Score silhouette par k</div>', unsafe_allow_html=True)
        df_sil = pd.DataFrame(dict(k=list(sil_scores.keys()), s=list(sil_scores.values())))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sil["k"], y=df_sil["s"], mode="lines+markers",
                                 line=dict(color=ACCENT, width=2),
                                 marker=dict(size=7, color=ACCENT,
                                             line=dict(color="#0A0B0E", width=1.5))))
        fig.add_vline(x=best_k, line=dict(color="#F59E0B", width=1.5, dash="dash"))
        fig.update_layout(**LAYOUT, height=250, xaxis_title="k", yaxis_title="Silhouette")
        st.plotly_chart(fig, use_container_width=True)
    with ce2:
        st.markdown('<div class="section-title">Adjusted Rand Index</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem;color:#475569;margin-bottom:1.2rem;line-height:1.6">Cohérence entre clusters prédits et vraies thématiques. 0 = aléatoire · 1 = parfait</p>', unsafe_allow_html=True)
        for name, val, color in [("KMeans", ari_km,"#6366F1"),("LDA", ari_lda,"#34D399")]:
            pct = max(0,min(1,val))*100
            st.markdown(f"""
<div class="eval-row">
  <div class="eval-label"><span class="eval-name">{name}</span><span class="eval-score">{val:.3f}</span></div>
  <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
</div>""", unsafe_allow_html=True)
