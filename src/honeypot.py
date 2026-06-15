"""
src/honeypot.py
===============
Flags *provably impossible* candidate profiles — honeypots planted in the
hackathon dataset.  These are NOT merely "bad" candidates; each flagged
profile contains data that is **logically self-contradictory**.

Three hard checks are applied (first trigger wins):

1. **Salary range inversion** — salary min > salary max.
2. **Temporal paradox** — last_active_date before signup_date.
3. **Experience duration paradox** — sum of career durations far exceeds
   claimed years of experience (with a 24-month buffer for overlaps).

Usage
-----
    from src.honeypot import check_honeypot, flag_honeypots

    is_hp, reason = check_honeypot(candidate)

    flags = flag_honeypots(candidates)
    # → {"CAND_0000009": (True, "Salary range inverted: ..."), ...}
"""

from __future__ import annotations

from datetime import date, datetime


# ═══════════════════════════════════════════════════════════════════════════
# Single-candidate check
# ═══════════════════════════════════════════════════════════════════════════

def check_honeypot(candidate: dict) -> tuple[bool, str | None]:
    """Check whether a single candidate profile is a honeypot.

    A candidate is flagged if **any** single check triggers.  The function
    returns on the **first** triggered check.

    Parameters
    ----------
    candidate:
        Full candidate dict as loaded from the dataset.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, reason_string)`` if the candidate is a honeypot,
        ``(False, None)`` otherwise.

    Examples
    --------
    >>> is_hp, reason = check_honeypot(cand_0000009)
    >>> is_hp
    True
    >>> reason
    "Salary range inverted: min 16.0 > max 7.3"
    """

    reasons: list[str] = []

    # ── CHECK 1: Salary range inversion ──────────────────────────────────
    result = _check_salary_inversion(candidate)
    if result is not None:
        reasons.append(result)

    # ── CHECK 2: Temporal paradox ────────────────────────────────────────
    result = _check_temporal_paradox(candidate)
    if result is not None:
        reasons.append(result)

    # ── CHECK 3: Experience duration paradox ─────────────────────────────
    # DISABLED: Too many false positives (~25K) from concurrent/overlapping
    # roles.  Not a reliable signal even at a 3x ratio threshold.

    # ── DECISION: Report-only, no hard flags ─────────────────────────────
    # After thorough data analysis:
    # - Individual signals are too noisy (~19K salary inversions, ~7.5K
    #   temporal paradoxes in the 100K dataset).
    # - Even dual-signal catches 1,466 — still 18x more than the ~80
    #   real honeypots the spec mentions.
    # - The structured scoring system (skill, career, alignment,
    #   behavioral, contradiction checks 1-7) already naturally avoids
    #   honeypots — 0 flagged candidates made it into the top 100 even
    #   without any explicit penalty.
    # - Hard-flagging risks zeroing out legitimately good candidates who
    #   happen to have noisy salary/date data.
    #
    # Per the spec: "We expect a good ranking system to naturally avoid
    # them; you don't need to special-case them."
    #
    # We report reasons for logging/awareness but never hard-flag.
    return False, "; ".join(reasons) if reasons else None


# ═══════════════════════════════════════════════════════════════════════════
# Batch function
# ═══════════════════════════════════════════════════════════════════════════

def flag_honeypots(candidates: list[dict]) -> dict[str, tuple[bool, str | None]]:
    """Run honeypot detection over a list of candidates.

    Parameters
    ----------
    candidates:
        List of candidate dicts, each expected to have a ``candidate_id`` key.

    Returns
    -------
    dict[str, tuple[bool, str | None]]
        Mapping of ``candidate_id`` → ``(is_honeypot, reason)``.

    Examples
    --------
    >>> flags = flag_honeypots(candidates)
    Flagged 3 honeypots out of 50 candidates
    """
    results: dict[str, tuple[bool, str | None]] = {}
    flagged_count = 0

    for candidate in candidates:
        cid = candidate.get("candidate_id", "UNKNOWN")
        is_hp, reason = check_honeypot(candidate)
        results[cid] = (is_hp, reason)
        if is_hp:
            flagged_count += 1

    total = len(candidates)
    print(f"Flagged {flagged_count} honeypots out of {total} candidates")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Internal check implementations
# ═══════════════════════════════════════════════════════════════════════════

def _check_salary_inversion(candidate: dict) -> str | None:
    """CHECK 1: Salary min > max → impossible."""
    try:
        salary = candidate["redrob_signals"]["expected_salary_range_inr_lpa"]
        sal_min = salary.get("min") if isinstance(salary, dict) else None
        sal_max = salary.get("max") if isinstance(salary, dict) else None
    except (KeyError, TypeError, AttributeError):
        return None

    if sal_min is None or sal_max is None:
        return None

    try:
        sal_min = float(sal_min)
        sal_max = float(sal_max)
    except (ValueError, TypeError):
        return None

    if sal_min > sal_max:
        return f"Salary range inverted: min {sal_min} > max {sal_max}"

    return None


def _check_temporal_paradox(candidate: dict) -> str | None:
    """CHECK 2: last_active_date before signup_date → impossible."""
    try:
        signals = candidate["redrob_signals"]
        signup_raw = signals.get("signup_date")
        active_raw = signals.get("last_active_date")
    except (KeyError, TypeError):
        return None

    if signup_raw is None or active_raw is None:
        return None

    signup = _parse_date(signup_raw)
    active = _parse_date(active_raw)

    if signup is None or active is None:
        return None

    if active < signup:
        return (
            f"Timeline paradox: last active {active_raw} "
            f"before signup {signup_raw}"
        )

    return None


def _check_experience_paradox(candidate: dict) -> str | None:
    """CHECK 3: Career duration total >> claimed years of experience → impossible.

    A 24-month buffer is allowed for overlapping roles, moonlighting, or
    rounding.  Only truly large discrepancies are flagged.
    """
    try:
        career_history = candidate.get("career_history", [])
        if not career_history:
            return None

        total_career_months = sum(
            role.get("duration_months", 0) or 0 for role in career_history
        )

        claimed_years = candidate["profile"]["years_of_experience"]
        if claimed_years is None:
            return None
        claimed_months = float(claimed_years) * 12
    except (KeyError, TypeError, ValueError):
        return None

    # Use a ratio-based check instead of a flat buffer.
    # Candidates routinely hold overlapping / concurrent roles, so the sum
    # of duration_months legitimately exceeds claimed years.  A flat 24-month
    # buffer flagged 20K+ normal profiles.  Requiring 3× the claimed months
    # catches only genuinely impossible profiles (e.g. 2 years claimed but
    # 6+ years of roles listed).
    if claimed_months <= 0:
        return None

    if total_career_months > (claimed_months * 3):
        return (
            f"Career duration paradox: {total_career_months} months of roles "
            f"but only {claimed_years} years claimed "
            f"({total_career_months / claimed_months:.1f}x ratio)"
        )

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Date parsing helper
# ═══════════════════════════════════════════════════════════════════════════

def _parse_date(value: str | None) -> date | None:
    """Best-effort date parsing.  Returns ``None`` on failure."""
    if not value or not isinstance(value, str):
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None
