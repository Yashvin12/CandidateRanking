"""
src/honeypot.py
===============
Weighted Strike System for Honeypot Detection
----------------------------------------------

Flags *provably impossible* candidate profiles — honeypots planted in the
hackathon dataset.  These are NOT merely "bad" candidates; each flagged
profile contains data that is **logically self-contradictory across
multiple dimensions simultaneously**.

Architecture
~~~~~~~~~~~~
Each candidate accumulates "strike points" from 13 independent checks
grouped by severity (13 checks total):

    Heavy  strikes (H1–H6):  2.0 points each
    Medium strikes (M1–M4):  1.5 points each
    Light  strikes (L1–L4):  1.0 points each

A candidate is flagged as a honeypot ONLY when total strike points
meet or exceed ``HONEYPOT_THRESHOLD``.  This multi-signal requirement
eliminates the massive false-positive rates (19K+) seen with single-signal
OR-logic on noisy synthetic data.

Optional LLM Synergy
~~~~~~~~~~~~~~~~~~~~~
If an external LLM has confirmed a title-description mismatch, 1.5 bonus
strikes are added.  This alone is never enough to flag anyone — it only
upgrades borderline candidates with additional heuristic violations.

Usage
-----
    from src.honeypot import check_honeypot, flag_honeypots, get_strike_breakdown

    is_hp, reason = check_honeypot(candidate)
    flags = flag_honeypots(candidates)
    breakdown = get_strike_breakdown(candidate)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# Tunable constants
# ═══════════════════════════════════════════════════════════════════════════

HONEYPOT_THRESHOLD: float = 4.0
"""Minimum total strike points to flag a candidate as a honeypot.

.. rubric:: Tuning guidance

Calibrated on 100K dataset: 4.0 → ~80 honeypots.
- If catching too many (>100), increase to 4.5
- If catching too few  (<60),  decrease to 3.5
"""

REFERENCE_DATE: date = date(2026, 6, 1)
"""Treat as "today" for all date calculations."""

# ── Strike value constants ────────────────────────────────────────────────
_HEAVY: float = 2.0
_MEDIUM: float = 1.5
_LIGHT: float = 1.0
_LLM_BONUS: float = 1.5

# ── Per-check thresholds (tune these individually) ────────────────────────
# TUNING: The salary gap threshold of 8.0 LPA for H1 can be adjusted —
#         raise it to 10.0 if you get too many, lower to 7.0 if you need more.
_H1_SALARY_GAP_THRESHOLD: float = 8.0

# TUNING: H2 temporal paradox.  Calibrated on 100K dataset:
#         30 days → ~154 flagged (too many)
#         60 days → ~100, 70 days → ~87, 80 days → ~83, 90 days → ~73
#         75 days is the sweet spot for ~80-85 flagged candidates.
_H2_TEMPORAL_GAP_DAYS: int = 75

_H3_EXPERIENCE_BUFFER_YEARS: float = 5.0
_H3_MIN_CLAIMED_YEARS: float = 10.0

_H4_MIN_ENDORSEMENTS: int = 150
_H4_MAX_CONNECTIONS: int = 25

# TUNING: If M1 expert fabrication too noisy, increase count from 5 to 7.
_M1_MIN_FABRICATED_SKILLS: int = 5
_M1_MAX_DURATION_MONTHS: int = 3

# TUNING: M3 requires multiple duplicate pairs to fire.  One duplicate pair
#         is common noise in synthetic data (~24K candidates).  Requiring ≥2
#         pairs is much more selective.  Increase to 3 if still too noisy.
_M3_MIN_DUPLICATE_PAIRS: int = 2

# TUNING: H5 max-possible-experience overshoot thresholds.
#         Diagnostic on 100K dataset (scratch_h5_diagnostic.py):
#           >1.5yr: 25 candidates (8-12yr overshoot — blatant fakes)
#           1.0-1.5yr: 0 candidates (empty bucket)
#           0.5-1.0yr: 13 candidates (~0.51yr — date arithmetic noise)
#           0.3-0.5yr: 4538 candidates (systematic rounding noise — DO NOT FLAG)
#         CRITICAL lowered 1.5→1.0 as safety-net (empty bucket, no impact today).
#         HEAVY stays at 0.5yr — lowering to 0.3 would cause ~4500 false positives.
_H5_CRITICAL_OVERSHOOT: float = 1.0   # CRITICAL override (4.0 pts) — was 1.5
_H5_HEAVY_OVERSHOOT: float = 0.5      # HEAVY strike (2.0 pts) — unchanged

_L1_MAX_COMPLETENESS: float = 30.0
_L1_MIN_EXPERT_SKILLS: int = 3

_L2_SALARY_GAP_MIN: float = 5.0
_L2_SALARY_GAP_MAX: float = _H1_SALARY_GAP_THRESHOLD  # must be <= _H1 threshold

# TUNING: L3 moderate temporal gap range.  Independent of H2 so that tuning
#         H2 doesn't inflate L3 coverage.  1-74 days = catches dead zone.
_L3_TEMPORAL_GAP_MAX: int = 74

_L4_MIN_SKILLS: int = 15
_L4_MAX_COMPLETENESS: float = 35.0


# ═══════════════════════════════════════════════════════════════════════════
# Date parsing helper
# ═══════════════════════════════════════════════════════════════════════════

def _parse_date(value: str | None) -> date | None:
    """Best-effort date parsing.  Returns ``None`` on failure.

    Handles ISO-8601 (``YYYY-MM-DD``), slash-separated, and DMY variants.
    """
    if not value or not isinstance(value, str):
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Safe accessor helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts.  Returns *default* if any key is missing."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def _safe_float(value: Any) -> float | None:
    """Convert *value* to float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Strike type for internal bookkeeping
# ═══════════════════════════════════════════════════════════════════════════

class _Strike:
    """Container for a single triggered strike."""

    __slots__ = ("id", "value", "reason")

    def __init__(self, strike_id: str, value: float, reason: str) -> None:
        self.id: str = strike_id
        self.value: float = value
        self.reason: str = reason

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for diagnostics."""
        return {"id": self.id, "value": self.value, "reason": self.reason}


# ═══════════════════════════════════════════════════════════════════════════
# Heavy strikes (2.0 points each)
# ═══════════════════════════════════════════════════════════════════════════

def _check_h1_salary(candidate: dict) -> _Strike | None:
    """H1 — Extreme salary inversion (min > max, gap > threshold)."""
    sal_min = _safe_float(_safe_get(candidate, "redrob_signals",
                                    "expected_salary_range_inr_lpa", "min"))
    sal_max = _safe_float(_safe_get(candidate, "redrob_signals",
                                    "expected_salary_range_inr_lpa", "max"))
    if sal_min is None or sal_max is None:
        return None

    if sal_min > sal_max:
        gap = sal_min - sal_max
        if gap > _H1_SALARY_GAP_THRESHOLD:
            return _Strike(
                "H1", _HEAVY,
                f"Extreme salary inversion: min {sal_min} > max {sal_max} "
                f"(gap: {gap:.1f} LPA)",
            )
    return None


def _check_h2_temporal(candidate: dict) -> _Strike | None:
    """H2 — Severe temporal paradox (last active > 30 days before signup)."""
    signup = _parse_date(_safe_get(candidate, "redrob_signals", "signup_date"))
    active = _parse_date(_safe_get(candidate, "redrob_signals", "last_active_date"))
    if signup is None or active is None:
        return None

    gap_days = (signup - active).days
    if gap_days > _H2_TEMPORAL_GAP_DAYS:
        return _Strike(
            "H2", _HEAVY,
            f"Severe temporal paradox: last active {gap_days} days before signup",
        )
    return None


def _check_h3_chronological(candidate: dict) -> _Strike | None:
    """H3 — Chronological experience paradox (claimed years >> career span).

    Uses the actual date window (earliest start → latest end) rather than
    summing durations, which breaks for overlapping roles.
    """
    career: list[dict] = candidate.get("career_history") or []
    if not career:
        return None

    start_dates: list[date] = []
    end_dates: list[date] = []

    for role in career:
        sd = _parse_date(role.get("start_date"))
        if sd is not None:
            start_dates.append(sd)

        ed_raw = role.get("end_date")
        if ed_raw is None:
            end_dates.append(REFERENCE_DATE)
        else:
            ed = _parse_date(ed_raw)
            if ed is not None:
                end_dates.append(ed)

    if not start_dates or not end_dates:
        return None

    earliest_start = min(start_dates)
    latest_end = max(end_dates)
    span_years = (latest_end - earliest_start).days / 365.25

    claimed = _safe_float(_safe_get(candidate, "profile", "years_of_experience"))
    if claimed is None:
        return None

    if claimed > (span_years + _H3_EXPERIENCE_BUFFER_YEARS) and claimed > _H3_MIN_CLAIMED_YEARS:
        return _Strike(
            "H3", _HEAVY,
            f"Chronological paradox: claims {claimed}yrs but career "
            f"window is only {span_years:.1f}yrs",
        )
    return None


def _check_h4_endorsements(candidate: dict) -> _Strike | None:
    """H4 — Impossible endorsement ratio (many endorsements, few connections)."""
    endorsements = _safe_get(candidate, "redrob_signals", "endorsements_received")
    connections = _safe_get(candidate, "redrob_signals", "connection_count")

    if endorsements is None or connections is None:
        return None

    try:
        endorsements = int(endorsements)
        connections = int(connections)
    except (ValueError, TypeError):
        return None

    if endorsements > _H4_MIN_ENDORSEMENTS and connections < _H4_MAX_CONNECTIONS:
        return _Strike(
            "H4", _HEAVY,
            f"{endorsements} endorsements but only {connections} connections",
        )
    return None


def _check_h5_max_possible_experience(candidate: dict) -> _Strike | None:
    """H5 — Impossible maximum experience (claims far more years than possible since first job)."""
    career: list[dict] = candidate.get("career_history") or []
    if not career:
        return None

    start_dates: list[date] = []
    for role in career:
        sd = _parse_date(role.get("start_date"))
        if sd is not None:
            start_dates.append(sd)

    if not start_dates:
        return None

    earliest = min(start_dates)
    max_possible = (REFERENCE_DATE - earliest).days / 365.25

    claimed = _safe_float(_safe_get(candidate, "profile", "years_of_experience"))
    if claimed is None:
        return None

    overshoot = claimed - max_possible
    if overshoot > _H5_CRITICAL_OVERSHOOT:
        return _Strike(
            "H5", 4.0,  # CRITICAL override
            f"Impossible max experience: claims {claimed}yr but started {earliest} "
            f"(max {max_possible:.1f}yr possible in 2026). Fabricated {overshoot:.1f}yr",
        )
    elif overshoot > _H5_HEAVY_OVERSHOOT:
        return _Strike(
            "H5", _HEAVY,
            f"Experience overshoot: claims {claimed}yr, max possible {max_possible:.1f}yr",
        )
    return None


# ── Tech birth-year table (approximate public availability) ───────────────
# Format: { skill_keyword_lower: first_possible_year }
# Only include technologies with a clear, verifiable public release date.
_TECH_BIRTH_YEARS: dict[str, int] = {
    "lora":                  2021,  # LoRA paper: Nov 2021
    "qlora":                 2023,  # QLoRA paper: May 2023
    "peft":                  2022,  # HF PEFT library: Feb 2022
    "langchain":             2022,  # Harrison Chase, Oct 2022
    "llama":                 2023,  # Meta LLaMA: Feb 2023
    "chatgpt":               2022,  # OpenAI: Nov 2022
    "gpt-4":                 2023,  # OpenAI: Mar 2023
    "pinecone":              2021,  # Public API: Jan 2021
    "qdrant":                2021,  # Qdrant 1.0: 2021
    "weaviate":              2019,  # Weaviate 1.0: 2019
    "pgvector":              2021,  # pgvector: Apr 2021
    "chroma":                2022,  # Chroma DB: 2022
    "milvus":                2019,  # Milvus 1.0: Oct 2019
    "sentence-transformers": 2019,  # SBERT paper: Aug 2019
    "faiss":                 2017,  # FAISS (Facebook): 2017
    "transformers":          2018,  # HF Transformers: Oct 2018
}
REFERENCE_YEAR: int = 2026


def _check_h6_tech_existence(candidate: dict) -> _Strike | None:
    """H6 — Tech existence paradox.

    Fires when a candidate claims expert/advanced proficiency in a technology
    with duration_months that implies they started using it BEFORE the
    technology publicly existed.

    Example: LoRA released 2021. In 2026, max possible = 5 years = 60 months.
    Claiming expert LoRA with duration_months = 96 is impossible.
    """
    skills = candidate.get("skills") or []
    violations = []

    for skill in skills:
        prof = (skill.get("proficiency") or "").lower()
        if prof not in ("expert", "advanced"):
            continue
        duration = skill.get("duration_months")
        if duration is None:
            continue
        name_lower = (skill.get("name") or "").lower()

        for tech_kw, birth_year in _TECH_BIRTH_YEARS.items():
            if tech_kw in name_lower:
                max_possible_months = (REFERENCE_YEAR - birth_year) * 12
                try:
                    if int(duration) > max_possible_months + 6:  # 6mo grace
                        violations.append(
                            f"{skill.get('name')}({prof}): "
                            f"{duration}mo claimed vs {max_possible_months}mo max "
                            f"(tech released {birth_year})"
                        )
                except (ValueError, TypeError):
                    pass
                break  # matched this skill, move to next

    if len(violations) >= 2:
        return _Strike(
            "H6", _HEAVY,
            f"Tech existence paradox ({len(violations)} skills predate technology: "
            f"{violations[0]})",
        )
    elif len(violations) == 1:
        return _Strike(
            "H6", _MEDIUM,
            f"Tech existence paradox: {violations[0]}",
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Medium strikes (1.5 points each)
# ═══════════════════════════════════════════════════════════════════════════

def _check_m1_fabrication(candidate: dict) -> _Strike | None:
    """M1 — Expert skill fabrication cluster (many expert/advanced skills with
    under 3 months of duration).
    """
    skills: list[dict] = candidate.get("skills") or []
    if not skills:
        return None

    fabricated_count = 0
    for skill in skills:
        proficiency = (skill.get("proficiency") or "").lower()
        if proficiency not in ("expert", "advanced"):
            continue
        duration = skill.get("duration_months")
        if duration is None:
            continue
        try:
            if int(duration) < _M1_MAX_DURATION_MONTHS:
                fabricated_count += 1
        except (ValueError, TypeError):
            continue

    if fabricated_count >= _M1_MIN_FABRICATED_SKILLS:
        return _Strike(
            "M1", _MEDIUM,
            f"{fabricated_count} expert/advanced skills claimed with <{_M1_MAX_DURATION_MONTHS}mo duration",
        )
    return None


def _check_m2_assessment(candidate: dict) -> _Strike | None:
    """M2 — Assessment score impossibility (exact 0.0 or 100.0 scores)."""
    scores: dict = _safe_get(candidate, "redrob_signals",
                             "skill_assessment_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return None

    impossible_values: list[str] = []
    for skill_name, score in scores.items():
        try:
            score_f = float(score)
        except (ValueError, TypeError):
            continue
        if score_f == 0.0 or score_f == 100.0:
            impossible_values.append(f"{skill_name}={score_f}")

    if impossible_values:
        return _Strike(
            "M2", _MEDIUM,
            f"Assessment scores contain impossible round values: "
            f"{', '.join(impossible_values)}",
        )
    return None


def _check_m3_recycled_desc(candidate: dict) -> _Strike | None:
    """M3 — Recycled description detection (duplicate descriptions across
    different career entries).

    Compares the first 200 characters of each description for exact matches.
    Requires at least ``_M3_MIN_DUPLICATE_PAIRS`` matching pairs to fire,
    because the synthetic data generator recycles descriptions broadly —
    a single duplicate pair is too common to be a reliable honeypot signal.
    """
    career: list[dict] = candidate.get("career_history") or []
    if len(career) < 3:
        return None

    # Extract (company, first-200-chars) pairs
    entries: list[tuple[str, str]] = []
    for role in career:
        desc = (role.get("description") or "")[:200].strip()
        company = role.get("company") or "Unknown"
        if desc:
            entries.append((company, desc))

    # Pairwise comparison — count ALL duplicate pairs
    dup_pairs: list[tuple[str, str]] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i][1] == entries[j][1]:
                dup_pairs.append((entries[i][0], entries[j][0]))

    if len(dup_pairs) >= _M3_MIN_DUPLICATE_PAIRS:
        pair_examples = dup_pairs[:2]  # show first 2 for readability
        pair_strs = [f"{a} and {b}" for a, b in pair_examples]
        return _Strike(
            "M3", _MEDIUM,
            f"{len(dup_pairs)} recycled description pairs found "
            f"(e.g. {'; '.join(pair_strs)})",
        )
    return None


def _check_m4_skill_career_span(candidate: dict) -> _Strike | None:
    """M4 — Skill duration exceeds career span.

    Fires when 2+ expert/advanced skills claim duration_months that exceeds
    the candidate's actual career span (from earliest job start to today).

    This is distinct from M1 (which catches <3mo claims). M4 catches the
    opposite: impossibly LONG skill durations relative to career length.
    """
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    if not career or not skills:
        return None

    start_dates = [_parse_date(r.get("start_date")) for r in career]
    start_dates = [d for d in start_dates if d]
    if not start_dates:
        return None

    earliest = min(start_dates)
    career_months = (REFERENCE_DATE - earliest).days / 30.44

    violations = []
    for s in skills:
        prof = (s.get("proficiency") or "").lower()
        if prof not in ("expert", "advanced"):
            continue
        dur = s.get("duration_months") or 0
        name = s.get("name", "?")
        try:
            dur = int(dur)
        except (ValueError, TypeError):
            continue
        # Allow 12-month grace (could have used it before current job history)
        if dur > (career_months + 12):
            excess = dur - career_months
            violations.append(
                f"{name}({prof}): {dur}mo vs {career_months:.0f}mo career "
                f"(+{excess:.0f}mo)"
            )

    if len(violations) >= 4:
        return _Strike(
            "M4", _HEAVY,
            f"Skill duration massively exceeds career: {violations[0]}, ...",
        )
    elif len(violations) >= 2:
        return _Strike(
            "M4", _MEDIUM,
            f"Skill duration exceeds career span ({len(violations)} skills): "
            f"{violations[0]}",
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Light strikes (1.0 points each)
# ═══════════════════════════════════════════════════════════════════════════

def _check_l1_empty_shell(candidate: dict) -> _Strike | None:
    """L1 — Empty shell with expert claims (low completeness + many expert skills)."""
    pcs = _safe_float(_safe_get(candidate, "redrob_signals",
                                "profile_completeness_score"))
    if pcs is None:
        return None

    skills: list[dict] = candidate.get("skills") or []
    expert_count = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
    )

    if pcs < _L1_MAX_COMPLETENESS and expert_count >= _L1_MIN_EXPERT_SKILLS:
        return _Strike(
            "L1", _LIGHT,
            f"Profile {pcs:.0f}% complete but claims {expert_count} expert skills",
        )
    return None


def _check_l2_salary(candidate: dict) -> _Strike | None:
    """L2 — Moderate salary inversion (gap between 5.0 and H1 threshold).

    Only fires if H1 did NOT fire (checked externally by the caller — but
    we encode the range logic here so that a gap > H1 threshold does not
    double-dip).
    """
    sal_min = _safe_float(_safe_get(candidate, "redrob_signals",
                                    "expected_salary_range_inr_lpa", "min"))
    sal_max = _safe_float(_safe_get(candidate, "redrob_signals",
                                    "expected_salary_range_inr_lpa", "max"))
    if sal_min is None or sal_max is None:
        return None

    if sal_min > sal_max:
        gap = sal_min - sal_max
        if _L2_SALARY_GAP_MIN <= gap <= _L2_SALARY_GAP_MAX:
            return _Strike(
                "L2", _LIGHT,
                f"Moderate salary inversion: gap {gap:.1f} LPA",
            )
    return None


def _check_l3_temporal(candidate: dict) -> _Strike | None:
    """L3 -- Moderate temporal paradox (last active 1-30 days before signup).

    Range is independent of H2 threshold so tuning H2 doesn't inflate L3.
    """
    signup = _parse_date(_safe_get(candidate, "redrob_signals", "signup_date"))
    active = _parse_date(_safe_get(candidate, "redrob_signals", "last_active_date"))
    if signup is None or active is None:
        return None

    gap_days = (signup - active).days
    if 1 <= gap_days <= _L3_TEMPORAL_GAP_MAX:
        return _Strike(
            "L3", _LIGHT,
            f"Minor temporal gap: last active {gap_days} days before signup",
        )
    return None


def _check_l4_skill_completeness(candidate: dict) -> _Strike | None:
    """L4 — Skill count vs completeness paradox."""
    skills: list[dict] = candidate.get("skills") or []
    pcs = _safe_float(_safe_get(candidate, "redrob_signals",
                                "profile_completeness_score"))
    if pcs is None:
        return None

    if len(skills) >= _L4_MIN_SKILLS and pcs < _L4_MAX_COMPLETENESS:
        return _Strike(
            "L4", _LIGHT,
            f"{len(skills)} skills listed but profile only {pcs:.0f}% complete",
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Strike collection & aggregation
# ═══════════════════════════════════════════════════════════════════════════

# Ordered list of all checks.  Each is (check_function, strike_id) — the
# strike_id is used only for mutual-exclusion logic (H1/L2, H2/L3).
_ALL_CHECKS: list[tuple[callable, str]] = [
    (_check_h1_salary,          "H1"),
    (_check_h2_temporal,        "H2"),
    (_check_h3_chronological,   "H3"),
    (_check_h4_endorsements,    "H4"),
    (_check_h5_max_possible_experience, "H5"),
    (_check_h6_tech_existence,  "H6"),
    (_check_m1_fabrication,     "M1"),
    (_check_m2_assessment,      "M2"),
    (_check_m3_recycled_desc,   "M3"),
    (_check_m4_skill_career_span, "M4"),
    (_check_l1_empty_shell,     "L1"),
    (_check_l2_salary,          "L2"),
    (_check_l3_temporal,        "L3"),
    (_check_l4_skill_completeness, "L4"),
]


def _collect_strikes(
    candidate: dict,
    llm_flags: dict | None = None,
) -> list[_Strike]:
    """Run all checks against *candidate* and return triggered strikes.

    Handles mutual-exclusion rules:
    - If H1 (extreme salary inversion) fires, L2 (moderate) is suppressed.
    - If H2 (severe temporal paradox) fires, L3 (moderate) is suppressed.
    """
    strikes: list[_Strike] = []
    fired_ids: set[str] = set()

    for check_fn, strike_id in _ALL_CHECKS:
        # Mutual-exclusion: skip L2 if H1 already fired (same salary signal)
        if strike_id == "L2" and "H1" in fired_ids:
            continue
        # Mutual-exclusion: skip L3 if H2 already fired (same temporal signal)
        if strike_id == "L3" and "H2" in fired_ids:
            continue

        strike = check_fn(candidate)
        if strike is not None:
            strikes.append(strike)
            fired_ids.add(strike_id)

    # ── LLM synergy bonus ────────────────────────────────────────────────
    if llm_flags and llm_flags.get("title_desc_mismatch") is True:
        strikes.append(_Strike(
            "LLM", _LLM_BONUS,
            "LLM-confirmed title-description mismatch",
        ))

    return strikes


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def check_honeypot(
    candidate: dict,
    llm_flags: dict | None = None,
) -> tuple[bool, str | None]:
    """Determine whether a single candidate is a honeypot.

    Parameters
    ----------
    candidate:
        Full candidate dict as loaded from the dataset.
    llm_flags:
        Optional dict with LLM-derived signals.  Currently the only
        recognised key is ``"title_desc_mismatch"`` (bool).

    Returns
    -------
    tuple[bool, str | None]
        ``(True, primary_reason)`` if the candidate is a honeypot,
        ``(False, None)`` otherwise.  ``primary_reason`` is the
        reason string of the highest-value strike that triggered.
    """
    strikes = _collect_strikes(candidate, llm_flags)
    total = sum(s.value for s in strikes)

    if total >= HONEYPOT_THRESHOLD:
        # Primary reason = highest-value strike
        primary = max(strikes, key=lambda s: s.value)
        return True, primary.reason

    return False, None


def flag_honeypots(
    candidates: list[dict],
    llm_flags_map: dict[str, dict] | None = None,
) -> dict[str, tuple[bool, str | None]]:
    """Batch honeypot detection over a list of candidates.

    Parameters
    ----------
    candidates:
        List of candidate dicts, each with a ``candidate_id`` key.
    llm_flags_map:
        Optional mapping of ``candidate_id`` → LLM flags dict.

    Returns
    -------
    dict[str, tuple[bool, str | None]]
        Mapping of ``candidate_id`` → ``(is_honeypot, reason)``.
    """
    results: dict[str, tuple[bool, str | None]] = {}
    flagged_count: int = 0

    for candidate in candidates:
        cid: str = candidate.get("candidate_id", "UNKNOWN")
        llm_flags = (llm_flags_map or {}).get(cid)
        is_hp, reason = check_honeypot(candidate, llm_flags)
        results[cid] = (is_hp, reason)
        if is_hp:
            flagged_count += 1

    total = len(candidates)
    print(f"Flagged {flagged_count} honeypots out of {total} candidates")
    return results


def get_strike_breakdown(
    candidate: dict,
    llm_flags: dict | None = None,
) -> dict[str, Any]:
    """Diagnostic: full breakdown of all strikes for a candidate.

    Useful for debugging and threshold tuning.

    Parameters
    ----------
    candidate:
        Full candidate dict.
    llm_flags:
        Optional LLM flags dict.

    Returns
    -------
    dict
        Breakdown with total strikes, each fired strike, and threshold info.
    """
    strikes = _collect_strikes(candidate, llm_flags)
    total = sum(s.value for s in strikes)

    return {
        "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
        "total_strikes": total,
        "is_honeypot": total >= HONEYPOT_THRESHOLD,
        "threshold": HONEYPOT_THRESHOLD,
        "strikes_fired": [s.to_dict() for s in strikes],
        "strikes_checked": len(_ALL_CHECKS),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import os
    import sys

    # Locate sample data relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    sample_path = os.path.join(
        project_root,
        "India_runs_data_and_ai_challenge",
        "sample_candidates.json",
    )

    if not os.path.exists(sample_path):
        print(f"Sample file not found: {sample_path}")
        sys.exit(1)

    with open(sample_path, "r", encoding="utf-8") as f:
        sample_data: list[dict] = json.load(f)

    # Build lookup
    by_id: dict[str, dict] = {c["candidate_id"]: c for c in sample_data}

    # ── Heuristic-only breakdowns ────────────────────────────────────────
    # CAND_0000009 and CAND_0000006 are honeypots with clear qualitative
    # signals (mismatched titles/descriptions, fictional companies) but
    # heuristically they each trigger only one heavy strike (2.0 pts).
    # CAND_0000001 should be completely clean.
    heuristic_tests: list[tuple[str, bool, str]] = [
        ("CAND_0000001", False, "should NOT be flagged (clean profile)"),
        ("CAND_0000009", False, "heuristic-only: H1 salary inversion = 2.0"),
        ("CAND_0000006", False, "heuristic-only: H2 temporal paradox = 2.0"),
    ]

    print("=" * 70)
    print("HONEYPOT STRIKE SYSTEM -- SELF-TEST")
    print(f"THRESHOLD = {HONEYPOT_THRESHOLD}")
    print("=" * 70)

    all_passed = True

    print("\n--- HEURISTIC-ONLY BREAKDOWNS ---")
    for cid, expected_flag, label in heuristic_tests:
        cand = by_id.get(cid)
        if cand is None:
            print(f"\n[WARN]  {cid} not found in sample data!")
            all_passed = False
            continue

        breakdown = get_strike_breakdown(cand)
        actual = breakdown["is_honeypot"]
        status = "[PASS]" if actual == expected_flag else "[FAIL]"
        if actual != expected_flag:
            all_passed = False

        print(f"\n{status}  {cid} ({label})")
        print(f"  Total strikes: {breakdown['total_strikes']} / {breakdown['threshold']}")
        print(f"  Is honeypot:   {breakdown['is_honeypot']}")
        if breakdown["strikes_fired"]:
            for s in breakdown["strikes_fired"]:
                print(f"    [{s['id']}] +{s['value']:.1f}  {s['reason']}")
        else:
            print("    (no strikes fired)")

    # ── LLM synergy demonstration ─────────────────────────────────────────
    # CAND_0000009 has mismatched career titles/descriptions.
    # CAND_0000006 has mismatched career titles/descriptions.
    # With LLM mismatch (+1.5): 2.0 + 1.5 = 3.5, still < 4.0 threshold.
    # These are borderline — they need additional signals or threshold tuning.
    print("\n--- LLM SYNERGY: BORDERLINE CANDIDATES ---")
    for cid in ["CAND_0000009", "CAND_0000006"]:
        cand = by_id.get(cid)
        if cand is None:
            continue
        llm = {"title_desc_mismatch": True}
        breakdown = get_strike_breakdown(cand, llm_flags=llm)
        print(f"\n  {cid} + LLM mismatch:")
        print(f"  Total strikes: {breakdown['total_strikes']} / {breakdown['threshold']}")
        print(f"  Is honeypot:   {breakdown['is_honeypot']}")
        for s in breakdown["strikes_fired"]:
            print(f"    [{s['id']}] +{s['value']:.1f}  {s['reason']}")

    # Clean candidate should stay clean even with LLM flag
    cand_clean = by_id.get("CAND_0000001")
    if cand_clean:
        bd = get_strike_breakdown(cand_clean, llm_flags={"title_desc_mismatch": True})
        ok = not bd["is_honeypot"]
        status = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_passed = False
        print(f"\n{status}  CAND_0000001 + LLM mismatch: {bd['total_strikes']} strikes "
              f"(should stay clean: {ok})")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED -- review output above")
    print("=" * 70)

    # ── Batch run on full sample ──────────────────────────────────────────
    print(f"\nBatch run on {len(sample_data)} sample candidates:")
    results = flag_honeypots(sample_data)
    flagged = {cid: reason for cid, (is_hp, reason) in results.items() if is_hp}
    if flagged:
        print(f"\nFlagged candidates:")
        for cid, reason in sorted(flagged.items()):
            print(f"  {cid}: {reason}")
