"""
generate_corpus.py
------------------
Génère un corpus synthétique réaliste de 300 descriptions
de projets / rapports de politiques publiques françaises,
répartis sur 6 thématiques distinctes.
Seed fixe -> reproductibilité garantie.
"""

import random
import json
from pathlib import Path
from typing import List, Dict


random.seed(42)

THEMES: Dict[str, Dict] = {
    "insertion_professionnelle": {
        "label": "Insertion professionnelle",
        "keywords": [
            "formation professionnelle", "retour à l emploi", "demandeurs d emploi",
            "reconversion professionnelle", "apprentissage", "alternance", "compétences",
            "qualification", "chômage longue durée", "insertion", "employabilité",
            "missions locales", "accompagnement emploi", "contrat aidé",
            "bilan de compétences", "certification professionnelle", "dispositif emploi",
        ]
    },
    "sante_publique": {
        "label": "Santé publique",
        "keywords": [
            "prévention sanitaire", "accès aux soins", "déserts médicaux",
            "santé mentale", "maladies chroniques", "vaccination", "dépistage",
            "médecine de proximité", "hôpital public", "sécurité sociale",
            "remboursement des soins", "épidémiologie", "bien-être",
            "promotion de la santé", "urgences médicales", "maternité",
            "diabète", "cancer", "obésité", "pathologies",
        ]
    },
    "transition_ecologique": {
        "label": "Transition écologique",
        "keywords": [
            "énergie renouvelable", "réduction des émissions", "biodiversité",
            "économie circulaire", "recyclage", "mobilité durable", "véhicules électriques",
            "isolation thermique", "rénovation énergétique", "empreinte carbone",
            "zones naturelles protégées", "agriculture biologique", "pesticides",
            "eau potable", "qualité de l air", "solaire", "éolien",
            "décarbonation", "développement durable", "écologie",
        ]
    },
    "inclusion_numerique": {
        "label": "Inclusion numérique",
        "keywords": [
            "fracture numérique", "accès internet", "compétences numériques",
            "alphabétisation numérique", "e-administration", "dématérialisation",
            "services en ligne", "seniors numériques", "zones blanches",
            "médiation numérique", "cybersécurité", "données personnelles",
            "intelligence artificielle", "automatisation", "télétravail",
            "formation digitale", "usages numériques", "outils informatiques",
        ]
    },
    "logement_social": {
        "label": "Logement social",
        "keywords": [
            "logement social", "HLM", "accès au logement", "mal logement",
            "sans-abri", "hébergement urgence", "rénovation urbaine",
            "mixité sociale", "loyers abordables", "aide au logement",
            "allocation logement", "expulsion locative", "bail solidaire",
            "quartiers prioritaires", "requalification urbaine", "habitat dégradé",
            "relogement", "politique de la ville", "résidence sociale",
        ]
    },
    "education_jeunesse": {
        "label": "Éducation et jeunesse",
        "keywords": [
            "décrochage scolaire", "réussite éducative", "école inclusive",
            "inégalités scolaires", "soutien scolaire", "jeunes en difficulté",
            "orientation scolaire", "lycée professionnel", "bourses étudiantes",
            "vie associative", "citoyenneté", "service civique", "sport jeunesse",
            "protection enfance", "parentalité", "périscolaire",
            "méthodes pédagogiques", "numérique éducatif", "égalité des chances",
        ]
    },
}

STRUCTURES = [
    (
        "Le projet vise à développer {kw0} en faveur de {population} dans le cadre de {cadre}. "
        "Les actions prévues incluent {kw1} et {kw2}. "
        "L objectif est d améliorer {kw3} sur le territoire {territoire}. "
        "Ce dispositif mobilise également {kw4} pour renforcer {kw5}."
    ),
    (
        "Cette initiative a pour but de renforcer {kw0} afin de réduire les inégalités. "
        "Elle s appuie sur {kw1} et cible principalement {population}. "
        "Le dispositif s articule autour de {kw2} et de {kw3}. "
        "Des actions de {kw4} complètent l intervention sur {territoire}."
    ),
    (
        "Dans le cadre de {cadre}, ce programme développe des actions de {kw0}. "
        "Il s adresse à {population} et propose {kw1}. "
        "Les partenaires locaux s engagent à mettre en oeuvre {kw2} et {kw3}. "
        "Le suivi intègre des indicateurs sur {kw4} et {kw5}."
    ),
    (
        "L action porte sur {kw0} en mobilisant {kw1}. "
        "Elle bénéficiera à {population} grâce à {kw2} et {kw3}. "
        "Le programme inclut également {kw4} pour atteindre les objectifs. "
        "Sur le territoire {territoire}, la priorité est donnée à {kw5}."
    ),
]

POPULATIONS = [
    "les jeunes de 16 à 25 ans", "les seniors", "les familles monoparentales",
    "les personnes en situation de handicap", "les habitants des zones rurales",
    "les bénéficiaires du RSA", "les primo-arrivants", "les personnes âgées",
    "les femmes en situation de précarité", "les travailleurs peu qualifiés",
    "les adultes en reconversion", "les enfants issus de milieux défavorisés",
]
TERRITOIRES = [
    "urbain", "rural", "périurbain", "métropolitain",
    "intercommunal", "régional", "départemental",
]
CADRES = [
    "la politique régionale", "le plan national", "le programme européen",
    "la convention territoriale", "le contrat de ville", "le schéma départemental",
    "la stratégie nationale", "le projet de territoire",
]


def generate_document(theme_key: str) -> Dict:
    """Génère un document synthétique pour un thème donné."""
    theme = THEMES[theme_key]
    kw = random.sample(theme["keywords"], k=min(6, len(theme["keywords"])))
    structure = random.choice(STRUCTURES)

    text = structure.format(
        kw0=kw[0], kw1=kw[1], kw2=kw[2],
        kw3=kw[3], kw4=kw[4], kw5=kw[5] if len(kw) > 5 else kw[0],
        population=random.choice(POPULATIONS),
        cadre=random.choice(CADRES),
        territoire=random.choice(TERRITOIRES),
    )
    return {"id": None, "theme": theme_key, "label": theme["label"], "text": text}


def generate_corpus(n_per_theme: int = 50) -> List[Dict]:
    """Génère un corpus équilibré de n_per_theme documents par thème."""
    docs = []
    doc_id = 1
    for theme_key in THEMES:
        for _ in range(n_per_theme):
            doc = generate_document(theme_key)
            doc["id"] = doc_id
            docs.append(doc)
            doc_id += 1
    random.shuffle(docs)
    return docs


if __name__ == "__main__":
    corpus = generate_corpus(n_per_theme=50)
    out = Path(__file__).parent.parent / "data" / "corpus.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Corpus généré : {len(corpus)} documents, {len(THEMES)} thèmes")
    for k, v in THEMES.items():
        print(f"  - {v['label']}")
