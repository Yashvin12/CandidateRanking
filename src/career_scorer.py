"""
src/career_scorer.py
====================
Scores the career quality of a candidate (0–30) across five dimensions:

A. **Company quality** (0–12) — product > research > unknown > consulting/non_tech.
B. **Title relevance** (0–10) — ML/AI titles score highest; non-tech titles score 0.
C. **Experience depth** (0–8) — 5–9 years is the sweet spot per the JD.
D. **Description analysis** (adjustments: +3 to −8) — production evidence
   bonuses, non-tech penalties, and title-description mismatch detection.
E. **LLM features** (optional, adjustments: −15 to +12) — signals extracted
   offline by ``src/llm_extractor.py`` and passed in as a pre-loaded dict.
   If the dict is absent the function falls back to heuristic-only scoring.
F. **Title-chaser penalty** (0 to −8) — candidates who switch companies every
   <18 months 3+ times are explicitly disqualified by the JD. Applied as a
   hard deduction so they cannot reach top-100 purely on skill/embedding score.

Usage
-----
    from src.career_scorer import compute_career_score

    # Heuristic only (no LLM features file)
    score, breakdown = compute_career_score(candidate, company_map)

    # With offline LLM features
    score, breakdown = compute_career_score(candidate, company_map, llm_features)
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

# ── Cache for lowercase company map (avoid rebuilding 100K times) ────
_map_cache_id: int | None = None
_map_cache_lower: dict[str, str] = {}


def _get_map_lower(company_map: dict[str, str]) -> dict[str, str]:
    """Return a lowercased version of company_map, cached across calls."""
    global _map_cache_id, _map_cache_lower
    cid = id(company_map)
    if cid != _map_cache_id:
        _map_cache_lower = {k.strip().lower(): v for k, v in company_map.items()}
        _map_cache_id = cid
    return _map_cache_lower


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_career_score(
    candidate: dict,
    company_map: dict[str, str],
    llm_features: dict[str, dict] | None = None,
) -> tuple[float, dict]:
    """Compute career quality score for a candidate.

    Parameters
    ----------
    candidate:
        Full candidate dict with ``profile`` and ``career_history``.
    company_map:
        Pre-computed mapping of company name → classification string,
        as produced by ``company_classifier.py``.
    llm_features:
        Optional mapping of ``candidate_id → feature dict`` pre-loaded from
        ``data/llm_features.jsonl`` by ``rank.py``.  If ``None`` or the id is
        absent the function falls back to heuristic-only scoring silently.

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

    # ── E: LLM feature adjustments (optional) ────────────────────────────
    llm_adj, llm_bd = _dimension_llm(
        candidate_id  = candidate.get("candidate_id", ""),
        llm_features  = llm_features,
        heuristic_consulting_only   = consulting_only,
        heuristic_title_mismatch    = title_desc_mismatch,
        heuristic_prod_bonus        = prod_bonus,
    )

    # ── F: Title-chaser penalty (0 to −8) ────────────────────────────────
    # JD explicitly disqualifies candidates who switch companies every 1.5yr.
    # 3+ short stints (<18mo) = hard disqualifier penalty.
    # 2 short stints = warning-level penalty.
    title_chaser_penalty, short_stint_count = _dimension_title_chaser(career_history)

    raw = (
        company_score + title_score + exp_score
        + prod_bonus + nontech_penalty + mismatch_penalty
        + llm_adj + title_chaser_penalty
    )
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
        "llm_adjustment": llm_adj,
        "llm_breakdown": llm_bd,
        "title_chaser_penalty": title_chaser_penalty,
        "short_stint_count": short_stint_count,
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

    # Use cached lowercase lookup (built once, reused for all 100K candidates).
    map_lower = _get_map_lower(company_map)

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
# Dimension F — Title-chaser penalty
# ═══════════════════════════════════════════════════════════════════════════

# JD states: candidates who switch every 1.5 years = not a fit, 3+ year
# commitment expected. 18 months (1.5yr) is the per-role threshold.
_TITLE_CHASER_THRESHOLD_MONTHS: int = 18
_TITLE_CHASER_HARD_PENALTY: float = -8.0   # 3+ short stints → disqualifier
_TITLE_CHASER_WARN_PENALTY: float = -3.0   # 2 short stints → warning
_TITLE_CHASER_MIN_HARD: int = 3             # stints below threshold to trigger hard
_TITLE_CHASER_MIN_WARN: int = 2             # stints below threshold to trigger warn


def _dimension_title_chaser(
    career_history: list[dict],
) -> tuple[float, int]:
    """Penalise title-chasers — candidates with many short stints < 18 months.

    Returns
    -------
    tuple[float, int]
        ``(penalty, short_stint_count)`` — penalty is 0.0, -3.0 or -8.0.
    """
    if not career_history:
        return 0.0, 0

    short_count = 0
    for role in career_history:
        dur = role.get("duration_months")
        try:
            if int(dur) < _TITLE_CHASER_THRESHOLD_MONTHS:
                short_count += 1
        except (TypeError, ValueError):
            continue

    if short_count >= _TITLE_CHASER_MIN_HARD:
        return _TITLE_CHASER_HARD_PENALTY, short_count
    if short_count >= _TITLE_CHASER_MIN_WARN:
        return _TITLE_CHASER_WARN_PENALTY, short_count
    return 0.0, short_count


# ═══════════════════════════════════════════════════════════════════════════
# Dimension D — Description analysis
# ═══════════════════════════════════════════════════════════════════════════

def _count_kw_hits(text: str, keywords: list[str]) -> int:
    """Count unique keywords present in text (each keyword counted at most once)."""
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Dimension E — LLM feature adjustments
# ═══════════════════════════════════════════════════════════════════════════

# Point values for LLM-extracted signals.
_LLM_PRODUCTION_RETRIEVAL_BONUS: float =  10.0   # confirmed prod RAG/vector DB
_LLM_PURE_CONSULTING_PENALTY:    float = -15.0   # entire career at body-shops
_LLM_TITLE_MISMATCH_PENALTY:     float = -12.0   # ML title + non-tech work
_LLM_ML_YEARS_BONUS_PER_YEAR:    float =   0.5   # incremental bonus per year of ML
_LLM_ML_YEARS_BONUS_CAP:         float =   5.0   # cap on the years bonus


def _dimension_llm(
    candidate_id: str,
    llm_features: dict[str, dict] | None,
    *,
    heuristic_consulting_only: bool,
    heuristic_title_mismatch:  bool,
    heuristic_prod_bonus:      float,
) -> tuple[float, dict]:
    """Apply LLM-extracted signal adjustments.

    Returns (adjustment_points, breakdown_dict).
    Returns (0.0, {}) when llm_features is None or the candidate has no entry,
    so the caller degrades silently to pure heuristic scoring.

    Design notes
    ------------
    * ``has_production_retrieval`` only adds the LLM bonus when the *heuristic*
      production bonus was **not already awarded** (>0), avoiding double-dipping.
    * ``is_pure_consulting`` only applies when the heuristic ``consulting_only``
      flag is False — i.e. the LLM caught something the company-map missed.
    * ``title_desc_mismatch`` applies only when the heuristic did *not* already
      fire, preventing double-stacking of the -8 heuristic and -12 LLM penalty.
    """
    if not llm_features or not candidate_id:
        return 0.0, {"source": "none"}

    feats = llm_features.get(candidate_id)
    if feats is None:
        return 0.0, {"source": "missing"}

    adjustment = 0.0
    breakdown: dict = {"source": "llm"}

    # ── Production retrieval bonus ────────────────────────────────────────
    has_prod = feats.get("has_production_retrieval", False)
    prod_bonus_applied = 0.0
    if has_prod and heuristic_prod_bonus == 0.0:
        # Heuristic missed it; trust the LLM.
        prod_bonus_applied = _LLM_PRODUCTION_RETRIEVAL_BONUS
        adjustment += prod_bonus_applied
    breakdown["has_production_retrieval"] = has_prod
    breakdown["production_retrieval_bonus"] = prod_bonus_applied

    # ── Pure consulting penalty ───────────────────────────────────────────
    is_pure_consulting = feats.get("is_pure_consulting", False)
    consulting_penalty_applied = 0.0
    if is_pure_consulting and not heuristic_consulting_only:
        consulting_penalty_applied = _LLM_PURE_CONSULTING_PENALTY
        adjustment += consulting_penalty_applied
    breakdown["is_pure_consulting"] = is_pure_consulting
    breakdown["consulting_penalty"] = consulting_penalty_applied

    # ── Title–description mismatch penalty ───────────────────────────────
    title_mismatch = feats.get("title_desc_mismatch", False)
    mismatch_penalty_applied = 0.0
    if title_mismatch and not heuristic_title_mismatch:
        mismatch_penalty_applied = _LLM_TITLE_MISMATCH_PENALTY
        adjustment += mismatch_penalty_applied
    breakdown["title_desc_mismatch"] = title_mismatch
    breakdown["mismatch_penalty"] = mismatch_penalty_applied

    # ── Years of applied ML bonus ─────────────────────────────────────────
    years_ml = max(0, int(feats.get("years_applied_ml", 0)))
    ml_years_bonus = min(years_ml * _LLM_ML_YEARS_BONUS_PER_YEAR, _LLM_ML_YEARS_BONUS_CAP)
    adjustment += ml_years_bonus
    breakdown["years_applied_ml"] = years_ml
    breakdown["ml_years_bonus"] = ml_years_bonus

    breakdown["total_adjustment"] = round(adjustment, 2)
    return round(adjustment, 2), breakdown


def _count_kw_hits_deduped(roles: list[dict], keywords: list[str]) -> int:
    """Count keywords that appear in AT LEAST ONE role description.

    This prevents recycled (copy-pasted) descriptions from inflating counts.
    Each keyword is counted at most once regardless of how many roles contain it.
    """
    found: set[str] = set()
    for role in roles:
        desc = (role.get("description") or "").lower()
        if not desc:
            continue
        for kw in keywords:
            if kw not in found and kw in desc:
                found.add(kw)
    return len(found)


def _dimension_description(
    current_title: str,
    career_history: list[dict],
) -> tuple[float, float, float, bool]:
    """Returns (production_bonus, nontech_penalty, mismatch_penalty,
    title_description_mismatch).

    Uses per-role deduplication so that recycled / copy-pasted descriptions
    don't multiply keyword counts.  Each keyword is counted at most once
    regardless of how many roles mention it (fixes 3A and 3B).
    """
    if not career_history:
        return 0.0, 0.0, 0.0, False

    # Deduplicated counts: a keyword found in 5 identical descriptions still
    # counts as 1.  Only the *presence* of a keyword across any role matters.
    production_count = _count_kw_hits_deduped(career_history, _PROD_KW_LOWER)
    nontech_count    = _count_kw_hits_deduped(career_history, _NONTECH_KW_LOWER)
    # code_count tracked for potential future use, not applied to score.
    # code_count = _count_kw_hits_deduped(career_history, _CODE_KW_LOWER)

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
