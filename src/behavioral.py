"""
src/behavioral.py
=================
Computes a **multiplicative** behavioral modifier (0.1–1.15) based on
platform engagement signals.  A value of 0.2 means an otherwise perfect
90-point candidate drops to 18 — this is intentional per the JD's
explicit instruction to down-weight unavailable candidates.

Five signals are combined via geometric mean:

1. Activity recency (days since last active)
2. Recruiter response rate
3. Availability composite (open-to-work + notice period)
4. Interview completion rate
5. Offer acceptance rate

Usage
-----
    from src.behavioral import compute_behavioral_multiplier

    multiplier, breakdown = compute_behavioral_multiplier(candidate["redrob_signals"])
"""

from __future__ import annotations

from datetime import date, datetime

from src.config import BEHAVIORAL, REFERENCE_DATE

_REF_DATE: date = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()


def compute_behavioral_multiplier(signals: dict) -> tuple[float, dict]:
    """Compute the behavioral multiplier for a candidate.

    Parameters
    ----------
    signals:
        The ``candidate["redrob_signals"]`` dict.

    Returns
    -------
    tuple[float, dict]
        ``(multiplier, breakdown)`` where multiplier ∈ [0.1, 1.15].
    """
    s1 = _signal_recency(signals)
    s2 = _signal_response_rate(signals)
    s3 = _signal_availability(signals)
    s4 = _signal_interview_completion(signals)
    s5 = _signal_offer_acceptance(signals)

    raw = (s1 * s2 * s3 * s4 * s5) ** (1.0 / 5.0)
    
    extra_bonus = _signal_extra_bonus(signals)
    multiplier = max(0.1, min(1.15, raw * extra_bonus))

    breakdown = {
        "recency": s1,
        "response_rate": s2,
        "availability": s3,
        "interview_completion": s4,
        "offer_acceptance": s5,
        "extra_bonus": extra_bonus,
        "combined": round(multiplier, 4),
    }
    return round(multiplier, 4), breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Signal implementations
# ═══════════════════════════════════════════════════════════════════════════

def _signal_recency(signals: dict) -> float:
    """SIGNAL 1: Days since last active → recency multiplier."""
    raw = signals.get("last_active_date")
    if raw is None:
        return 0.2  # worst case

    parsed = _parse_date(raw)
    if parsed is None:
        return 0.2

    days_since = (_REF_DATE - parsed).days
    if days_since < 0:
        days_since = 0  # future date = very active

    for max_days, mult in BEHAVIORAL["recency_tiers"]:
        if days_since < max_days:
            return mult

    return 0.2  # fallback (shouldn't reach here given 9999 sentinel)


def _signal_response_rate(signals: dict) -> float:
    """SIGNAL 2: Recruiter response rate."""
    rr = signals.get("recruiter_response_rate")
    if rr is None:
        return 0.35  # worst case

    try:
        rr = float(rr)
    except (ValueError, TypeError):
        return 0.35

    for min_rate, mult in BEHAVIORAL["response_rate_tiers"]:
        if rr >= min_rate:
            return mult

    return 0.35


def _signal_availability(signals: dict) -> float:
    """SIGNAL 3: Open-to-work flag + notice period composite."""
    notice = signals.get("notice_period_days")
    try:
        notice = int(notice) if notice is not None else 180
    except (ValueError, TypeError):
        notice = 180

    open_to_work = signals.get("open_to_work_flag", False)

    # 90-day notice is the JD's outer limit ("30+ day notice, bar gets higher").
    # Treat >=90 days as functionally hard to hire on short timeline.
    if notice >= 90:
        return 0.2 if open_to_work else 0.1

    if notice <= 30:
        base = 0.9
    elif notice <= 60:
        base = 0.7
    else:
        base = 0.5

    return min(1.0, base + 0.2) if open_to_work else base


def _signal_interview_completion(signals: dict) -> float:
    """SIGNAL 4: Interview completion rate."""
    icr = signals.get("interview_completion_rate")
    if icr is None:
        return 0.3  # worst case

    try:
        icr = float(icr)
    except (ValueError, TypeError):
        return 0.3

    for min_rate, mult in BEHAVIORAL["interview_rate_tiers"]:
        if icr >= min_rate:
            return mult

    return 0.3


def _signal_offer_acceptance(signals: dict) -> float:
    """SIGNAL 5: Offer acceptance rate."""
    oar = signals.get("offer_acceptance_rate")
    if oar is None:
        return 0.7  # treat like no-data sentinel

    try:
        oar = float(oar)
    except (ValueError, TypeError):
        return 0.7

    # Sentinel values from config
    if oar in BEHAVIORAL["offer_rate_map"]:
        return BEHAVIORAL["offer_rate_map"][oar]

    # oar > 0
    if oar > BEHAVIORAL["offer_rate_low_threshold"]:
        return BEHAVIORAL["offer_rate_default_above_zero"]

    # 0 < oar <= 0.5
    return BEHAVIORAL["offer_rate_low_value"]


# ═══════════════════════════════════════════════════════════════════════════
# Date helper
# ═══════════════════════════════════════════════════════════════════════════

def _parse_date(value: str | None) -> date | None:
    if not value or not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None

def _signal_extra_bonus(signals: dict) -> float:
    """Upgraded bonus: range now 1.0–1.15 to meaningfully separate twins.

    High-weight signals (GitHub, recruiter saves, applications, response time)
    use tiered bonuses to create real differentiation between otherwise-identical
    candidates.  Low-weight identity/verification signals stay small.

    A dead-lead gate prevents bonus accumulation for candidates with very long
    notice periods who aren't open to work.
    """
    bonus = 1.0

    # ── HIGH WEIGHT signals (separators between otherwise-identical candidates)

    # GitHub activity: strong real-world evidence of coding
    gh = signals.get("github_activity_score", -1)
    if isinstance(gh, (int, float)) and gh > 70:
        bonus += 0.04
    elif isinstance(gh, (int, float)) and gh > 40:
        bonus += 0.02

    # Market demand: other recruiters saving this profile
    saves = signals.get("saved_by_recruiters_30d", 0)
    if isinstance(saves, int) and saves >= 5:
        bonus += 0.03
    elif isinstance(saves, int) and saves >= 2:
        bonus += 0.015

    # Active job seeker: submitted applications recently
    apps = signals.get("applications_submitted_30d", 0)
    if isinstance(apps, int) and apps >= 5:
        bonus += 0.02

    # Response speed: fast responders are easier to hire
    resp_time = signals.get("avg_response_time_hours")
    if isinstance(resp_time, (int, float)) and resp_time < 12:
        bonus += 0.02
    elif isinstance(resp_time, (int, float)) and resp_time < 24:
        bonus += 0.005

    # Profile views: passive market signal
    views = signals.get("profile_views_received_30d")
    if isinstance(views, int) and views >= 20:
        bonus += 0.01

    # Search appearances: recruiter demand signal
    searches = signals.get("search_appearance_30d")
    if isinstance(searches, int) and searches >= 100:
        bonus += 0.01

    # Preferred work mode: flexibility signal
    work_mode = signals.get("preferred_work_mode")
    if isinstance(work_mode, str) and work_mode.lower() in ("hybrid", "flexible"):
        bonus += 0.005

    # ── LOW WEIGHT signals (identity/verification — keep small)
    if signals.get("verified_email"):
        bonus += 0.005
    if signals.get("verified_phone"):
        bonus += 0.005
    if signals.get("linkedin_connected"):
        bonus += 0.005

    # ── Gate: unavailable candidates get no bonus
    notice = signals.get("notice_period_days", 0)
    open_to_work = signals.get("open_to_work_flag", False)
    try:
        if int(notice) > 90 and not open_to_work:
            return 1.0  # no bonus for dead-lead candidates
    except (TypeError, ValueError):
        pass

    return min(1.15, bonus)  # increased cap from 1.05 to 1.15
