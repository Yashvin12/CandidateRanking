"""
src/reasoning.py
================
Generates a specific, data-grounded reasoning string (max 200 chars) for
each ranked candidate.  The reasoning is built from concrete profile data —
never generic filler.

Usage
-----
    from src.reasoning import generate_reasoning

    text = generate_reasoning(candidate, subscores)
"""

from __future__ import annotations

from src.config import MUST_HAVE_SKILL_GROUPS

# Pre-build a flat set of must-have keywords (lowered) for skill extraction.
_MUST_HAVE_KEYWORDS: dict[str, list[str]] = {
    group: [kw.lower() for kw in keywords]
    for group, keywords in MUST_HAVE_SKILL_GROUPS.items()
}


def generate_reasoning(candidate: dict, subscores: dict) -> str:
    """Generate a concise reasoning string for a candidate's ranking.

    Parameters
    ----------
    candidate:
        Full candidate dict.
    subscores:
        Dict containing breakdown dicts from all scoring modules.
        Expected keys: ``skill_score``, ``career_score``, ``embedding_score``,
        ``behavioral``, ``skill_breakdown``.

    Returns
    -------
    str
        Reasoning string, max 200 characters, safe for CSV embedding.
    """
    parts: list[str] = []

    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    # ── Part 1: Career headline ──────────────────────────────────────────
    headline = _build_headline(profile)
    if headline:
        parts.append(headline)

    # ── Part 2: Key strength ─────────────────────────────────────────────
    strength = _pick_strength(subscores)
    if strength:
        parts.append(strength)

    # ── Part 3: Top matched skills (up to 3) ─────────────────────────────
    matched = _extract_matched_skills(candidate)
    if matched:
        parts.append("Skills: " + " ".join(matched))

    # ── Part 4: Location + notice ────────────────────────────────────────
    loc_notice = _build_location_notice(profile, signals)
    if loc_notice:
        parts.append(loc_notice)

    # ── Part 5: Behavioral signal ────────────────────────────────────────
    rr = signals.get("recruiter_response_rate")
    if rr is not None:
        try:
            pct = int(float(rr) * 100)
            parts.append(f"{pct}% response rate")
        except (ValueError, TypeError):
            pass

    result = "; ".join(parts)

    # Truncate to 200 chars, clean up for CSV safety.
    if len(result) > 200:
        result = result[:197] + "..."

    # Strip any newlines or bare problematic chars.
    result = result.replace("\n", " ").replace("\r", " ")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_headline(profile: dict) -> str | None:
    years = profile.get("years_of_experience")
    title = profile.get("current_title")
    company = profile.get("current_company")

    if not title:
        return None

    # Format years as int if whole, else 1 decimal.
    yr_str = ""
    if years is not None:
        try:
            y = float(years)
            yr_str = str(int(y)) if y == int(y) else f"{y:.1f}"
        except (ValueError, TypeError):
            pass

    parts = []
    if yr_str:
        parts.append(f"{yr_str}yr")
    parts.append(title)
    if company:
        parts.append(f"at {company}")

    return " ".join(parts)


def _pick_strength(subscores: dict) -> str | None:
    skill = subscores.get("skill_score", 0)
    career = subscores.get("career_score", 0)
    embedding = subscores.get("embedding_score", 0)
    behavioral = subscores.get("behavioral", 0)

    if skill >= 30:
        return "strong retrieval/ranking skill match"
    if career >= 25:
        return "deep product-company ML track record"
    if embedding >= 12:
        return "semantically aligned career trajectory"
    if behavioral >= 0.85:
        return "highly active and responsive"
    return "relevant technical background"


def _extract_matched_skills(candidate: dict) -> list[str]:
    """Find up to 3 candidate skill names that matched must-have groups."""
    skills = candidate.get("skills") or []
    if not skills:
        return []

    matched: list[str] = []
    seen_groups: set[str] = set()

    for s in skills:
        raw_name = s.get("name")
        if not raw_name or not isinstance(raw_name, str):
            continue
        name_lower = raw_name.lower()

        for group_name, keywords in _MUST_HAVE_KEYWORDS.items():
            if group_name in seen_groups:
                continue
            for kw in keywords:
                if kw in name_lower:
                    # Use a short, clean version of the skill name.
                    display = raw_name.strip()
                    if len(display) > 20:
                        display = display[:17] + "..."
                    matched.append(display)
                    seen_groups.add(group_name)
                    break
            if len(matched) >= 3:
                break
        if len(matched) >= 3:
            break

    return matched


def _build_location_notice(profile: dict, signals: dict) -> str | None:
    location = profile.get("location")
    notice = signals.get("notice_period_days")

    parts = []
    if location:
        # Take just the city part if it's long.
        loc = location.strip()
        if len(loc) > 25:
            loc = loc[:22] + "..."
        parts.append(loc)
    if notice is not None:
        try:
            parts.append(f"{int(notice)}d notice")
        except (ValueError, TypeError):
            pass

    return " ".join(parts) if parts else None
