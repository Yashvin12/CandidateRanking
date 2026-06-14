"""
src/career_scorer.py
====================
Scores the career quality of a candidate (0–30) across four dimensions:

A. **Company quality** (0–12) — product > research > unknown > consulting/non_tech.
B. **Title relevance** (0–10) — ML/AI titles score highest; non-tech titles score 0.
C. **Experience depth** (0–8) — 5–9 years is the sweet spot per the JD.
D. **Description analysis** (adjustments: +3 to −8) — production evidence
   bonuses, non-tech penalties, and title-description mismatch detection.

Usage
-----
    from src.career_scorer import compute_career_score

    score, breakdown = compute_career_score(candidate, company_map)
"""

from __future__ import annotations

from src.config import (
    CONSULTING_FIRMS,
    PRODUCT_COMPANIES,
    ML_AI_TITLES,
    ADJACENT_TECH_TITLES,
    NON_TECH_TITLES,
    PRODUCTION_EVIDENCE_KEYWORDS,
    CODE_WRITING_EVIDENCE,
    NON_TECH_DESCRIPTION_KEYWORDS,
)

# ── Pre-compute lowered title sets for fuzzy matching ────────────────
_ML_AI_LOWER:       list[str] = [t.lower() for t in ML_AI_TITLES]
_ADJACENT_LOWER:    list[str] = [t.lower() for t in ADJACENT_TECH_TITLES]
_NON_TECH_LOWER:    list[str] = [t.lower() for t in NON_TECH_TITLES]

_CONSULTING_LOWER:  set[str]  = {c.strip().lower() for c in CONSULTING_FIRMS}
_PRODUCT_LOWER:     set[str]  = {c.strip().lower() for c in PRODUCT_COMPANIES}

_PROD_KW_LOWER:     list[str] = [kw.lower() for kw in PRODUCTION_EVIDENCE_KEYWORDS]
_CODE_KW_LOWER:     list[str] = [kw.lower() for kw in CODE_WRITING_EVIDENCE]
_NONTECH_KW_LOWER:  list[str] = [kw.lower() for kw in NON_TECH_DESCRIPTION_KEYWORDS]

# Points per company classification
_COMPANY_POINTS: dict[str, int] = {
    "product": 4,
    "consulting": 0,
    "research": 1,
    "non_tech": 0,
    "unknown": 2,
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_career_score(
    candidate: dict,
    company_map: dict[str, str],
) -> tuple[float, dict]:
    """Compute career quality score for a candidate.

    Parameters
    ----------
    candidate:
        Full candidate dict with ``profile`` and ``career_history``.
    company_map:
        Pre-computed mapping of company name → classification string,
        as produced by ``company_classifier.py``.

    Returns
    -------
    tuple[float, dict]
        ``(score, breakdown)`` where score ∈ [0.0, 30.0].
    """
    career_history = candidate.get("career_history") or []
    profile = candidate.get("profile") or {}

    # ── A: Company quality (0-12) ────────────────────────────────────────
    company_score, consulting_only = _dimension_company(career_history, company_map)

    # ── B: Title relevance (0-10) ────────────────────────────────────────
    current_title = profile.get("current_title") or ""
    title_score, ml_history_count = _dimension_title(current_title, career_history)

    # ── C: Experience depth (0-8) ────────────────────────────────────────
    years = profile.get("years_of_experience")
    exp_score = _dimension_experience(years)

    # ── D: Description analysis (bonus/penalty) ──────────────────────────
    prod_bonus, nontech_penalty, mismatch_penalty, title_desc_mismatch = (
        _dimension_description(current_title, career_history)
    )

    raw = company_score + title_score + exp_score + prod_bonus + nontech_penalty + mismatch_penalty
    total = max(0.0, min(30.0, raw))

    breakdown = {
        "company_score": company_score,
        "title_score": title_score,
        "exp_score": exp_score,
        "production_bonus": prod_bonus,
        "nontech_penalty": nontech_penalty,
        "mismatch_penalty": mismatch_penalty,
        "consulting_only": consulting_only,
        "title_description_mismatch": title_desc_mismatch,
        "total": round(total, 2),
    }
    return round(total, 2), breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Dimension A — Company quality
# ═══════════════════════════════════════════════════════════════════════════

def _dimension_company(
    career_history: list[dict],
    company_map: dict[str, str],
) -> tuple[float, bool]:
    if not career_history:
        return 0.0, False

    # Build a case-insensitive lookup from the company_map.
    map_lower: dict[str, str] = {k.strip().lower(): v for k, v in company_map.items()}

    points = 0
    all_consulting = True

    for role in career_history:
        company_raw = role.get("company") or ""
        company_key = company_raw.strip().lower()
        if not company_key:
            all_consulting = False
            continue

        # Look up in company_map first.
        classification = map_lower.get(company_key)

        # Fallback to seed lists if not in the map.
        if classification is None:
            if company_key in _CONSULTING_LOWER:
                classification = "consulting"
            elif company_key in _PRODUCT_LOWER:
                classification = "product"
            else:
                classification = "unknown"

        points += _COMPANY_POINTS.get(classification, 2)

        if classification != "consulting":
            all_consulting = False

    return min(float(points), 12.0), all_consulting


# ═══════════════════════════════════════════════════════════════════════════
# Dimension B — Title relevance
# ═══════════════════════════════════════════════════════════════════════════

def _fuzzy_match_title(title: str, title_set: list[str]) -> bool:
    """Check bidirectional containment: either the title contains a known
    title or a known title contains the candidate's title."""
    t = title.lower()
    if not t:
        return False
    for known in title_set:
        if known in t or t in known:
            return True
    return False


def _dimension_title(
    current_title: str,
    career_history: list[dict],
) -> tuple[float, int]:
    if _fuzzy_match_title(current_title, _ML_AI_LOWER):
        base = 8.0
    elif _fuzzy_match_title(current_title, _ADJACENT_LOWER):
        base = 5.0
    elif _fuzzy_match_title(current_title, _NON_TECH_LOWER):
        base = 0.0
    else:
        base = 3.0

    ml_history_count = 0
    for role in career_history:
        role_title = role.get("title") or ""
        if _fuzzy_match_title(role_title, _ML_AI_LOWER):
            ml_history_count += 1

    return min(base + ml_history_count * 1.0, 10.0), ml_history_count


# ═══════════════════════════════════════════════════════════════════════════
# Dimension C — Experience depth
# ═══════════════════════════════════════════════════════════════════════════

def _dimension_experience(years: float | int | None) -> float:
    if years is None:
        return 2.0
    try:
        y = float(years)
    except (ValueError, TypeError):
        return 2.0

    if 5 <= y <= 9:
        return 8.0
    if 4 <= y < 5:
        return 6.0
    if 9 < y <= 12:
        return 6.0
    if 3 <= y < 4:
        return 4.0
    if 12 < y <= 15:
        return 4.0
    if y > 15:
        return 3.0
    # y < 3
    return 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Dimension D — Description analysis
# ═══════════════════════════════════════════════════════════════════════════

def _count_kw_hits(text: str, keywords: list[str]) -> int:
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


def _dimension_description(
    current_title: str,
    career_history: list[dict],
) -> tuple[float, float, float, bool]:
    """Returns (production_bonus, nontech_penalty, mismatch_penalty,
    title_description_mismatch)."""
    if not career_history:
        return 0.0, 0.0, 0.0, False

    combined = " ".join(
        role.get("description") or "" for role in career_history
    ).lower()

    production_count = _count_kw_hits(combined, _PROD_KW_LOWER)
    code_count = _count_kw_hits(combined, _CODE_KW_LOWER)       # tracked, not used in score
    nontech_count = _count_kw_hits(combined, _NONTECH_KW_LOWER)

    # Production bonus
    if production_count >= 3:
        bonus = 3.0
    elif production_count >= 1:
        bonus = 1.0
    else:
        bonus = 0.0

    # Non-tech penalty (higher threshold overrides lower)
    penalty = 0.0
    if nontech_count >= 8 and production_count == 0:
        penalty = -8.0
    elif nontech_count >= 5 and production_count == 0:
        penalty = -5.0

    # Title-description mismatch
    mismatch_penalty = 0.0
    title_desc_mismatch = False
    if (
        _fuzzy_match_title(current_title, _ML_AI_LOWER)
        and nontech_count >= 3
        and production_count == 0
    ):
        mismatch_penalty = -8.0
        title_desc_mismatch = True

    return bonus, penalty, mismatch_penalty, title_desc_mismatch
