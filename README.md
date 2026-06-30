# NLP — Analyse Thématique de Politiques Publiques

Pipeline NLP complet appliqué à un corpus de descriptions de projets de politiques publiques françaises (300 documents, 6 thématiques). Combine **TF-IDF**, **clustering KMeans**, **Topic Modeling LDA** et **réduction dimensionnelle (SVD)** pour extraire et visualiser automatiquement les grandes thématiques d'un corpus textuel. Architecture modulaire, typée, avec 23 tests unitaires.

---

## Cas d'usage

Analyser automatiquement un corpus de rapports ou de descriptions de projets pour :
- Identifier les grandes thématiques sans labels prédéfinis (approche non supervisée)
- Visualiser la répartition thématique dans l'espace sémantique
- Comparer les méthodes de clustering (KMeans) et de topic modeling (LDA)
- Évaluer la cohérence des clusters par rapport aux vrais thèmes (ARI)

---

## Thématiques couvertes

| Thème | Description |
|-------|-------------|
| Insertion professionnelle | Formation, emploi, reconversion, alternance |
| Santé publique | Prévention, accès aux soins, épidémiologie |
| Transition écologique | Énergie renouvelable, biodiversité, mobilité durable |
| Inclusion numérique | Fracture numérique, e-administration, compétences digitales |
| Logement social | HLM, hébergement, rénovation urbaine |
| Éducation et jeunesse | Décrochage scolaire, réussite éducative, service civique |

---

## Structure du projet

```
nlp-thematic-analysis/
├── src/
│   ├── generate_corpus.py   # Générateur de corpus synthétique (300 docs, 6 thèmes)
│   ├── preprocessing.py     # Nettoyage, tokenisation, stopwords, stemming
│   ├── modeling.py          # TF-IDF, KMeans, LDA, SVD, évaluation
│   └── run_analysis.py      # Pipeline principal (point d'entrée)
├── data/
│   └── corpus.json          # Corpus généré (300 documents)
├── outputs/
│   ├── analysis_results.json  # Métriques, top termes par cluster, topics LDA
│   └── viz_data.json          # Coordonnées 2D pour visualisation
├── tests/
│   └── test_nlp_pipeline.py  # 23 tests unitaires (preprocessing + modeling)
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Corpus brut (JSON)
     │
     ▼
Prétraitement (preprocessing.py)
  clean_text() → tokenize() → remove_stopwords() → (stemming optionnel)
     │
     ▼
Vectorisation TF-IDF (max 5000 features, n-grammes 1-2)
     │
     ├──► Sélection k optimal (score de silhouette sur k=2..9)
     │
     ├──► KMeans clustering (k=6) → top termes par cluster
     │
     ├──► SVD 2D → coordonnées pour visualisation scatter plot
     │
     └──► LDA Topic Modeling (6 topics, 15 passes) → distribution des mots
          │
          ▼
     Évaluation (ARI, silhouette)
          │
          ▼
     Sauvegarde JSON (outputs/)
```

---

## Résultats

| Méthode | ARI | Interprétation |
|---------|-----|----------------|
| KMeans (k=6) | 0.256 | Modéré — cohérence thématique partielle |
| LDA (6 topics) | 0.240 | Modéré — topics sémantiquement distincts |

**Score de silhouette (k=6) : 0.066**

> Un ARI modéré est attendu sur un corpus synthétique dont les thèmes partagent un vocabulaire administratif commun. Sur un corpus réel (rapports DREES, publications INSEE), les thèmes seraient plus nettement séparés et les scores plus élevés.

---

## Installation & lancement

```bash
git clone https://github.com/EKOURAOGO/nlp-thematic-analysis.git
cd nlp-thematic-analysis

pip install -r requirements.txt

# Générer le corpus
python3 src/generate_corpus.py

# Lancer le pipeline complet
python3 src/run_analysis.py
```

---

## Tests

```bash
python3 -m pytest tests/ -v
```

Sortie attendue : `23 passed`

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-LDA-blue?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-stopwords%20·%20stemming-green?style=flat-square)
![pytest](https://img.shields.io/badge/pytest-23%20tests-red?style=flat-square)

---

## Auteur

**Emmanuel KOURAOGO** — M2 IMSD · Paris-Saclay
[GitHub](https://github.com/EKOURAOGO) · [Email](mailto:ekouraogo73@gmail.com)
