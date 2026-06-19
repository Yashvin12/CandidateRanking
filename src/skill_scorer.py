"""
src/skill_scorer.py
===================
Computes a skill-match score (0–40) for a candidate against the JD's
must-have skill groups, nice-to-have skills, credibility checks, and
skill-assessment scores.

Scoring pipeline (in order):

1. **Must-have group matching** — 10 pts per group × 4 groups = 40 max.
2. **Nice-to-have bonus** — up to +5 (capped at 40 total).
3. **Credibility penalty** — -5 or -10 for many "expert" claims with
   near-zero duration.
4. **Assessment score bonus** — up to +5 for high assessment marks on
   must-have skills (capped at 40 total).
5. **Domain ratio recording** — wrong-domain and core-NLP/IR counts are
   stored in the breakdown for downstream use (alignment_scorer.py) but
   do NOT affect the score here.

Usage
-----
    from src.skill_scorer import compute_skill_score

    score, breakdown = compute_skill_score(candidate)
"""

from __future__ import annotations

from src.config import (
    MUST_HAVE_SKILL_GROUPS,
    NICE_TO_HAVE_SKILLS,
    WRONG_DOMAIN_SKILLS,
)

# Pre-compute lowered keyword lists once at import time so we're not
# calling .lower() inside hot loops over 100K candidates.
_MUST_HAVE_LOWER: dict[str, list[str]] = {
    group: [kw.lower() for kw in keywords]
    for group, keywords in MUST_HAVE_SKILL_GROUPS.items()
}

_NICE_TO_HAVE_LOWER: list[str] = [kw.lower() for kw in NICE_TO_HAVE_SKILLS]
_WRONG_DOMAIN_LOWER: list[str] = [kw.lower() for kw in WRONG_DOMAIN_SKILLS]

# Flatten all must-have keywords into one set for assessment matching.
_ALL_MUST_HAVE_LOWER: set[str] = set()
for _kws in _MUST_HAVE_LOWER.values():
    _ALL_MUST_HAVE_LOWER.update(_kws)

# ── Career description fallback patterns (Step 4 — plain-language rescue) ──
# These keywords are checked against combined career descriptions when a
# must-have skill group scores 0 from the skill list alone.  This rescues
# Tier 5 candidates who describe systems in plain English without using
# exact product names like "Pinecone" or "FAISS".
_VECTOR_DB_FALLBACK: list[str] = [
    "vector database", "vector store", "vector search",
    "semantic search", "approximate nearest neighbor", "ann index",
    "embedding index", "faiss", "pinecone", "weaviate", "milvus",
    "opensearch", "elasticsearch", "qdrant", "pgvector", "chroma",
]

_PRODUCTION_RETRIEVAL_FALLBACK: list[str] = [
    "recommendation system", "search system", "retrieval system",
    "ranking system", "information retrieval", "hybrid search",
    "dense retrieval", "sparse retrieval", "bm25", "rag",
    "retrieval augmented", "reranking", "re-ranking",
]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_skill_score(candidate: dict) -> tuple[float, dict]:
    """Score a candidate's skills against the JD requirements.

    Parameters
    ----------
    candidate:
        Full candidate dict.  Expected keys used:
        - ``skills`` — list of dicts with ``name``, ``proficiency``,
          ``endorsements``, ``duration_months``.
        - ``redrob_signals.skill_assessment_scores`` — dict of
          ``{skill_name: score_0_to_100}``.

    Returns
    -------
    tuple[float, dict]
        ``(score, breakdown)`` where score ∈ [0.0, 40.0].
    """
    skills = candidate.get("skills") or []

    # Fast path: nothing to score.
    if not skills:
        return 0.0, _empty_breakdown()

    # Normalise skill names once; carry along the full skill dict.
    normed: list[tuple[str, dict]] = []
    for s in skills:
        raw_name = s.get("name")
        if not raw_name or not isinstance(raw_name, str):
            continue
        normed.append((raw_name.lower(), s))

    # ── STEP 1: Must-have group matching (max 40) ────────────────────────
    group_scores: dict[str, float] = {}
    for group_name, keywords in _MUST_HAVE_LOWER.items():
        group_scores[group_name] = _score_group(normed, keywords)

    # ── STEP 1b: Career description fallback for skill groups ─────────────
    # A Tier 5 candidate may describe building systems without using exact
    # skill keywords.  If a must-have group scored 0, scan career descriptions
    # as a fallback.  (Python fallback existed already; this extends to
    # vector_db and production_retrieval per the JD's plain-language trap.)
    career_history = candidate.get("career_history") or []
    desc_combined = " ".join(
        (r.get("description") or "") for r in career_history
    ).lower()

    if group_scores.get("python", 0.0) == 0.0:
        if "python" in desc_combined:
            group_scores["python"] = 7.0  # base score (no duration/endorsement bonuses)

    if group_scores.get("vector_db", 0.0) == 0.0:
        if any(kw in desc_combined for kw in _VECTOR_DB_FALLBACK):
            group_scores["vector_db"] = 7.5  # boosted: rewards Tier 5 plain-language builders

    if group_scores.get("production_retrieval", 0.0) == 0.0:
        if any(kw in desc_combined for kw in _PRODUCTION_RETRIEVAL_FALLBACK):
            group_scores["production_retrieval"] = 7.5  # boosted: rewards Tier 5 plain-language builders

    step1_total = sum(group_scores.values())


    # ── STEP 2: Nice-to-have bonus (max +5, total capped at 40) ──────────
    nth_count = _count_keyword_hits(normed, _NICE_TO_HAVE_LOWER)
    nth_bonus = min(nth_count * 1.5, 5.0)
    running = min(step1_total + nth_bonus, 40.0)

    # ── STEP 3: Credibility penalty ─────────────────────────────────────
    cred_penalty = _credibility_penalty(normed, desc_combined)
    running = max(0.0, running + cred_penalty)

    # ── STEP 4: Assessment score bonus (max +5, total capped at 40) ──────
    assess_bonus = _assessment_bonus(candidate)
    final = min(40.0, running + assess_bonus)

    # ── STEP 5: Domain ratio counts (recorded, not applied) ──────────────
    wrong_domain_count = _count_keyword_hits(normed, _WRONG_DOMAIN_LOWER)
    core_nlp_ir_count = _count_must_have_hits(normed)

    breakdown = {
        **group_scores,
        "nice_to_have_bonus": nth_bonus,
        "credibility_penalty": cred_penalty,
        "assessment_bonus": assess_bonus,
        "wrong_domain_count": wrong_domain_count,
        "core_nlp_ir_count": core_nlp_ir_count,
        "total": round(final, 2),
    }
    return round(final, 2), breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _score_group(
    normed_skills: list[tuple[str, dict]],
    group_keywords: list[str],
) -> float:
    """Score a single must-have skill group (0–10).

    Finds all candidate skills whose name contains any keyword from the
    group (case-insensitive substring match).  The *best* matching skill
    (by duration) drives the bonus points.
    """
    best_duration: int = -1
    best_endorsements: int = 0
    best_proficiency: str = ""
    matched = False

    for skill_lower, skill_dict in normed_skills:
        for kw in group_keywords:
            if kw in skill_lower:
                matched = True
                dur = skill_dict.get("duration_months") or 0
                if dur > best_duration:
                    best_duration = dur
                    best_endorsements = skill_dict.get("endorsements") or 0
                    best_proficiency = (skill_dict.get("proficiency") or "").lower()
                elif dur == best_duration:
                    # Tiebreak: prefer higher endorsements
                    end = skill_dict.get("endorsements") or 0
                    if end > best_endorsements:
                        best_endorsements = end
                        best_proficiency = (skill_dict.get("proficiency") or "").lower()
                break  # one keyword match is enough per skill

    if not matched:
        return 0.0

    score = 7.0
    if best_duration >= 12:
        score += 1.5
    if best_endorsements >= 10:
        score += 1.0
    if best_proficiency in ("advanced", "expert"):
        score += 0.5

    return min(score, 10.0)


def _count_keyword_hits(
    normed_skills: list[tuple[str, dict]],
    keyword_list: list[str],
) -> int:
    """Count how many keywords from a flat list match any candidate skill."""
    count = 0
    for kw in keyword_list:
        for skill_lower, _ in normed_skills:
            if kw in skill_lower:
                count += 1
                break  # count each keyword at most once
    return count


def _count_must_have_hits(normed_skills: list[tuple[str, dict]]) -> int:
    """Count candidate skills that match ANY must-have keyword."""
    count = 0
    for skill_lower, _ in normed_skills:
        for kw in _ALL_MUST_HAVE_LOWER:
            if kw in skill_lower:
                count += 1
                break
    return count


def _credibility_penalty(
    normed_skills: list[tuple[str, dict]],
    desc_combined: str = "",
) -> float:
    """Penalise candidates who claim expert/advanced with <6 months duration.

    Part B of Step 4: if keyword stuffing is detected AND career descriptions
    don't mention the stuffed skills, apply a stronger penalty (-15 instead
    of -10).  This separates genuine experts from keyword stuffers.
    """
    suspect_count = 0
    suspect_names: list[str] = []
    for skill_lower, skill_dict in normed_skills:
        prof = (skill_dict.get("proficiency") or "").lower()
        dur = skill_dict.get("duration_months") or 0
        if prof in ("expert", "advanced") and dur < 6:
            suspect_count += 1
            suspect_names.append(skill_lower)

    if suspect_count >= 5:
        # Check if career descriptions corroborate the stuffed skills.
        # If none of the suspect skill names appear in descriptions,
        # this is a pure keyword stuffer → harsher penalty.
        if desc_combined and not any(sn in desc_combined for sn in suspect_names):
            return -15.0
        return -10.0
    if suspect_count >= 3:
        return -5.0
    return 0.0


def _assessment_bonus(candidate: dict) -> float:
    """Bonus for high assessment scores on must-have skills (max +5)."""
    try:
        assessments = candidate["redrob_signals"]["skill_assessment_scores"]
    except (KeyError, TypeError):
        return 0.0

    if not assessments or not isinstance(assessments, dict):
        return 0.0

    bonus = 0.0
    for skill_name, score_val in assessments.items():
        if not isinstance(skill_name, str):
            continue
        skill_lower = skill_name.lower()

        # Check if this assessment skill matches any must-have keyword
        hit = False
        for kw in _ALL_MUST_HAVE_LOWER:
            if kw in skill_lower:
                hit = True
                break
        if not hit:
            continue

        try:
            score_val = float(score_val)
        except (ValueError, TypeError):
            continue

        if score_val >= 70:
            bonus += 2.0
        elif score_val >= 50:
            bonus += 1.0

    return min(bonus, 5.0)


def _empty_breakdown() -> dict:
    """Return a zeroed-out breakdown dict for candidates with no skills."""
    bd: dict = {}
    for group_name in MUST_HAVE_SKILL_GROUPS:
        bd[group_name] = 0.0
    bd["nice_to_have_bonus"] = 0.0
    bd["credibility_penalty"] = 0.0
    bd["assessment_bonus"] = 0.0
    bd["wrong_domain_count"] = 0
    bd["core_nlp_ir_count"] = 0
    bd["total"] = 0.0
    return bd
