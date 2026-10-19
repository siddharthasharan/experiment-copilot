"""Keyword retriever over the methodology corpus.

Deterministic, dependency-free, and portable: the same scoring logic is
mirrored in the showroom's chat UI (vanilla JS), so retrieval behavior
tested here is the behavior users get.
"""

import json
import re
from pathlib import Path

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "per",
    "are", "was", "has", "have", "will", "would", "should", "could",
    "such", "than", "then", "when", "what", "which", "your", "you",
    "our", "its", "they", "them", "their", "about", "under", "over",
    "between", "both", "each", "more", "most", "also", "only", "just",
    "like", "how", "why", "not", "but", "all", "any", "can", "may",
    "does", "did", "were", "been", "being", "who", "whom", "whose",
    "there", "here", "had", "his", "her", "him", "she", "his",
}

CORPUS_PATH = Path(__file__).parent.parent / "rag" / "corpus.json"


def tokenize(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 2}


def load_corpus(path=CORPUS_PATH):
    with open(path) as f:
        return json.load(f)["notes"]


def _note_fields(note):
    """Pre-tokenized searchable fields with weights: keywords 3, title 2, body 1."""
    return {
        "keywords": {t for kw in note["keywords"] for t in tokenize(kw)},
        "title": tokenize(note["title"]),
        "body": tokenize(
            note["summary"] + " " + " ".join(note["key_points"]) + " " + note["interview_angle"]
        ),
    }


def score_note(fields, query_tokens):
    return (
        3 * len(query_tokens & fields["keywords"])
        + 2 * len(query_tokens & fields["title"])
        + 1 * len(query_tokens & fields["body"])
    )


def retrieve(query, notes, k=3):
    """Return the top-k (note, score) pairs for a query string."""
    qtokens = tokenize(query)
    scored = [(note, score_note(_note_fields(note), qtokens)) for note in notes]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]
