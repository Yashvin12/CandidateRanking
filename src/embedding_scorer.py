"""
src/embedding_scorer.py
=======================
Batch-computes semantic similarity between a JD anchor text and all
candidates' career text using sentence-transformers (all-MiniLM-L6-v2).

Returns a list of scores (0.0–15.0), one per candidate, same order as input.

If sentence-transformers is not installed, returns a neutral mid-score (7.5)
for every candidate with a printed warning — the pipeline never crashes.

Usage
-----
    from src.embedding_scorer import compute_embedding_scores

    scores = compute_embedding_scores(candidates)  # list[float]
"""

from __future__ import annotations

from src.config import JD_TEXT_FOR_EMBEDDING

_MODEL_NAME = "all-MiniLM-L6-v2"
_MAX_TEXT_LEN = 512


def _build_text(candidate: dict) -> str:
    """Concatenate summary + all career descriptions, truncated to 512 chars."""
    profile = candidate.get("profile") or {}
    text = profile.get("summary") or ""

    for role in candidate.get("career_history") or []:
        desc = role.get("description") or ""
        if desc:
            text += " " + desc

    if len(text) > _MAX_TEXT_LEN:
        text = text[:_MAX_TEXT_LEN]

    return text


def compute_embedding_scores(candidates: list[dict]) -> list[float]:
    """Compute semantic similarity scores for all candidates against the JD.

    Parameters
    ----------
    candidates:
        List of candidate dicts.

    Returns
    -------
    list[float]
        Scores in [0.0, 15.0], same order and length as input.
    """
    n = len(candidates)
    if n == 0:
        return []

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "[WARNING] sentence-transformers not installed. "
            "Returning neutral mid-score (7.5) for all candidates."
        )
        return [7.5] * n

    print(f"Loading embedding model: {_MODEL_NAME}")
    model = SentenceTransformer(_MODEL_NAME)

    all_texts = [_build_text(c) for c in candidates]

    print(f"Encoding {n} candidate texts...")
    candidate_embeddings = model.encode(
        all_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    jd_embedding = model.encode(
        JD_TEXT_FOR_EMBEDDING,
        normalize_embeddings=True,
    )

    # Normalised embeddings → cosine similarity = dot product.
    similarities = candidate_embeddings @ jd_embedding

    scores = [
        round(max(0.0, min(15.0, float(sim) * 15.0)), 4)
        for sim in similarities
    ]

    print(f"Computed embedding scores for {n} candidates")
    return scores
