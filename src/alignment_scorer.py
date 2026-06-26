"""
src/alignment_scorer.py
=======================
Scores secondary alignment signals (0–15) across three dimensions plus a
domain-fit penalty:

A. **Location** (0–8) — Noida/Pune preferred, India tier-1 OK, abroad penalised.
B. **Education** (0–4) — CS/ML fields and top-tier institutions score highest.
C. **Notice period** (0–3) — shorter notice + open-to-work flags score higher.
D. **Domain fit penalty** (−6 to 0) — applied when wrong-domain skills
   (CV/speech) dominate over core NLP/IR skills.

Usage
-----
    from src.alignment_scorer import compute_alignment_score

    score, breakdown = compute_alignment_score(candidate, skill_breakdown)
"""

from __future__ import annotations

from src.config import PREFERRED_LOCATIONS, TIER1_INDIA_CITIES, NON_TECH_TITLES

_PREFERRED_LOWER: list[str] = [loc.lower() for loc in PREFERRED_LOCATIONS]
_TIER1_LOWER:     list[str] = [loc.lower() for loc in TIER1_INDIA_CITIES]
_NON_TECH_LOWER:  list[str] = [t.lower() for t in NON_TECH_TITLES]

# Education field tiers (checked in order — first match wins)
_FIELD_TIER1: list[str] = [
    "computer science", "machine learning", "artificial intelligence",
    "data science", "information technology", "statistics", "mathematics",
]
_FIELD_TIER2: list[str] = [
    "electronics", "electrical", "computer engineering",
]
_FIELD_TIER3: list[str] = [
    "physics", "engineering",
]

_TIER_SCORES: dict[str, float] = {
    "tier_1": 1.5,
    "tier_2": 1.0,
    "tier_3": 0.5,
    "tier_4": 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_alignment_score(
    candidate: dict,
    skill_breakdown: dict,
) -> tuple[float, dict]:
    """Compute alignment score for a candidate.

    Parameters
    ----------
    candidate:
        Full candidate dict.
    skill_breakdown:
        Breakdown dict from ``skill_scorer.compute_skill_score``, expected
        to contain ``wrong_domain_count`` and ``core_nlp_ir_count``.

    Returns
    -------
    tuple[float, dict]
        ``(score, breakdown)`` where score ∈ [0.0, 15.0].
    """
    location_score = _dimension_location(candidate)
    education_score = _dimension_education(candidate)
    notice_score = _dimension_notice(candidate)
    domain_penalty = _domain_fit_penalty(skill_breakdown)

    raw = location_score + education_score + notice_score + domain_penalty
    total = max(0.0, min(15.0, raw))

    # ── Non-tech title gate (JD: CV/robotics/non-tech without NLP/IR = disqualifier) ──
    # If the candidate's current title is clearly non-technical AND they have
    # zero core NLP/IR skills, location+education alone should not float them
    # into top rankings. Cap alignment at 2.0 as a soft gate.
    profile = candidate.get("profile") or {}
    current_title = (profile.get("current_title") or "").lower()
    core_nlp_ir = skill_breakdown.get("core_nlp_ir_count", 0)
    is_non_tech_title = any(nt in current_title for nt in _NON_TECH_LOWER)
    non_tech_gated = False
    if is_non_tech_title and core_nlp_ir == 0:
        total = min(total, 2.0)
        non_tech_gated = True

    breakdown = {
        "location_score": location_score,
        "education_score": education_score,
        "notice_score": notice_score,
        "domain_penalty": domain_penalty,
        "non_tech_gated": non_tech_gated,
        "total": round(total, 2),
    }
    return round(total, 2), breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Dimension A — Location
# ═══════════════════════════════════════════════════════════════════════════

def _dimension_location(candidate: dict) -> float:
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").strip()
    willing = signals.get("willing_to_relocate", False)

    # Preferred locations (Noida, Pune)
    for pref in _PREFERRED_LOWER:
        if pref in location:
            return 8.0

    is_india = country.lower() == "india" if country else False

    if is_india:
        # Check tier-1 cities
        for city in _TIER1_LOWER:
            if city in location:
                return 7.0 if willing else 5.0
        # Other Indian cities
        return 5.0 if willing else 3.0

    # Outside India
    return 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Dimension B — Education
# ═══════════════════════════════════════════════════════════════════════════

def _field_score(field: str) -> float:
    fl = field.lower()
    for kw in _FIELD_TIER1:
        if kw in fl:
            return 2.0
    for kw in _FIELD_TIER2:
        if kw in fl:
            return 1.5
    for kw in _FIELD_TIER3:
        if kw in fl:
            return 1.0
    return 0.5


def _dimension_education(candidate: dict) -> float:
    education = candidate.get("education") or []
    if not education:
        return 0.5  # no education data → minimal baseline

    best = 0.0
    for entry in education:
        if not isinstance(entry, dict):
            continue

        field = entry.get("field_of_study") or ""
        tier = entry.get("tier", "tier_4") or "tier_4"

        fs = _field_score(field)
        ts = _TIER_SCORES.get(tier, 0.0)
        entry_score = fs + ts

        if entry_score > best:
            best = entry_score

    return min(best, 4.0)


# ═══════════════════════════════════════════════════════════════════════════
# Dimension C — Notice period
# ═══════════════════════════════════════════════════════════════════════════

def _dimension_notice(candidate: dict) -> float:
    signals = candidate.get("redrob_signals") or {}
    notice = signals.get("notice_period_days")
    open_to_work = signals.get("open_to_work_flag", False)

    if notice is None:
        return 1.0  # no data → conservative default

    try:
        notice = int(notice)
    except (ValueError, TypeError):
        return 1.0

    if notice <= 30:
        return 3.0
    if notice <= 60:
        # JD: "bar gets higher" past 30 days — steeper cliff from 3.0 → 1.5
        return 1.5
    if notice < 90:
        return 1.0 if open_to_work else 0.5
    # notice >= 90 — consistent with behavioral.py >=90 heavy penalty
    return 0.5 if open_to_work else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Domain fit penalty
# ═══════════════════════════════════════════════════════════════════════════

def _domain_fit_penalty(skill_breakdown: dict) -> float:
    wrong = skill_breakdown.get("wrong_domain_count", 0)
    core = skill_breakdown.get("core_nlp_ir_count", 0)

    if wrong <= 0:
        return 0.0
    if core == 0:
        # No NLP/IR skills at all — pure wrong-domain candidate
        return -6.0
    if wrong > core * 2:
        # More than 2:1 wrong-to-core ratio → significant domain mismatch
        # Tightened from *3 to *2: catches mixed profiles like speech+NLP earlier
        return -3.0
    if wrong >= 2 and wrong >= core:
        # Roughly equal or more wrong-domain than core — gentle signal
        return -1.5
    return 0.0
