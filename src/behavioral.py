"""
src/behavioral.py
=================
Computes a **multiplicative** behavioral modifier (0.1–1.0) based on
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
        ``(multiplier, breakdown)`` where multiplier ∈ [0.1, 1.0].
    """
    s1 = _signal_recency(signals)
    s2 = _signal_response_rate(signals)
    s3 = _signal_availability(signals)
    s4 = _signal_interview_completion(signals)
    s5 = _signal_offer_acceptance(signals)

    raw = (s1 * s2 * s3 * s4 * s5) ** (1.0 / 5.0)
    multiplier = max(0.1, min(1.0, raw))

    breakdown = {
        "recency": s1,
        "response_rate": s2,
        "availability": s3,
        "interview_completion": s4,
        "offer_acceptance": s5,
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
    open_to_work = signals.get("open_to_work_flag")
    if open_to_work is None:
        open_to_work = False

    if open_to_work:
        return 1.0

    notice = signals.get("notice_period_days")
    if notice is None:
        return 0.3  # worst case

    try:
        notice = int(notice)
    except (ValueError, TypeError):
        return 0.3

    if notice <= 60:
        return 0.8
    if notice <= 90:
        return 0.6
    return 0.3


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
