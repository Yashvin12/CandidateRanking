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

from src.config import MUST_HAVE_SKILL_GROUPS, TIER1_INDIA_CITIES, PREFERRED_LOCATIONS

# Pre-build a flat set of must-have keywords (lowered) for skill extraction.
_MUST_HAVE_KEYWORDS: dict[str, list[str]] = {
    group: [kw.lower() for kw in keywords]
    for group, keywords in MUST_HAVE_SKILL_GROUPS.items()
}


def generate_reasoning(
    candidate: dict,
    subscores: dict,
    *,
    rank: int = 0,
    career_breakdown: dict | None = None,
) -> str:
    """Generate a concise reasoning string for a candidate's ranking.

    Parameters
    ----------
    candidate:
        Full candidate dict.
    subscores:
        Dict containing breakdown dicts from all scoring modules.
        Expected keys: ``skill_score``, ``career_score``, ``embedding_score``,
        ``behavioral``, ``skill_breakdown``.
    rank:
        1-based rank position. Concerns are suppressed for ranks 1-5.
    career_breakdown:
        Career scorer breakdown dict (contains ``consulting_only``,
        ``title_chaser_penalty``, etc.).  Optional.

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

    # ── Concern clause (ranks 6+) ─────────────────────────────────────────
    if rank > 5:
        concern = _pick_concern(candidate, subscores, career_breakdown)
        if concern:
            result += f"; Note: {concern}"

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
    skill_bd = subscores.get("skill_breakdown", {})

    # Order matters: most specific/differentiating first.
    if skill >= 30 and career >= 25:
        return "elite skill-and-career fit"
    if skill >= 30:
        return "strong retrieval/ranking skill match"
    if career >= 25:
        return "deep product-company ML track record"
    if behavioral >= 0.90:
        return "highly active and responsive"
    if embedding >= 12:
        return "semantically aligned career trajectory"
    # Notice period <=30 is explicitly preferred by the JD
    # (checked via skill_breakdown proxy — can't access signals directly here,
    # so use career score as tie-breaker)
    if career >= 20 and skill >= 25:
        return "strong system-builder with verified production track record"
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
                    # Use a readable version — truncate only if genuinely long.
                    display = raw_name.strip()
                    if len(display) > 22:
                        display = display[:19] + "..."
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


# Pre-built lowercase set of all recognised Indian locations.
_INDIA_LOCATIONS_LOWER: set[str] = {
    loc.lower() for loc in (TIER1_INDIA_CITIES | PREFERRED_LOCATIONS)
}

# Indian state / territory names — fallback for Tier 2/3 cities not in the
# explicit city sets.  Catches "Indore, Madhya Pradesh", "Trivandrum, Kerala",
# "Coimbatore, Tamil Nadu", etc.
_INDIA_STATES_LOWER: set[str] = {
    s.lower() for s in {
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
        "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
        "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
        "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal",
        # Union territories
        "Chandigarh", "Delhi", "Jammu and Kashmir", "Ladakh",
        "Lakshadweep", "Puducherry", "Andaman and Nicobar",
    }
}


def _pick_concern(
    candidate: dict,
    subscores: dict,
    career_breakdown: dict | None,
) -> str | None:
    """Return the single highest-priority concern string, or *None*.

    Priority order (1 = highest):
      1. Limited experience (<4 yr)
      2. Long notice (>90 d)
      3. Low platform engagement (behavioral < 0.6)
      4. Consulting-only background
      5. Frequent job-switching (title_chaser_penalty <= -8)
      6. Based outside India
    """
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    cb = career_breakdown or {}

    # 1. Limited experience
    yoe = profile.get("years_of_experience")
    if yoe is not None:
        try:
            y = float(yoe)
            if y < 4:
                yr_display = str(int(y)) if y == int(y) else f"{y:.1f}"
                return f"Limited experience ({yr_display}yr)"
        except (ValueError, TypeError):
            pass

    # 2. Long notice period
    notice = signals.get("notice_period_days")
    if notice is not None:
        try:
            n = int(notice)
            if n > 90:
                return f"Long notice ({n}d)"
        except (ValueError, TypeError):
            pass

    # 3. Low platform engagement
    behav = subscores.get("behavioral", 1.0)
    try:
        if float(behav) < 0.6:
            return "Low platform engagement"
    except (ValueError, TypeError):
        pass

    # 4. Consulting-only background
    if cb.get("consulting_only") is True:
        return "Consulting-only background"

    # 5. Frequent job-switching
    tcp = cb.get("title_chaser_penalty")
    if tcp is not None:
        try:
            if float(tcp) <= -8.0:
                return "Frequent job-switching"
        except (ValueError, TypeError):
            pass

    # 6. Based outside India
    # Three-layer check: (a) country field, (b) known city names,
    # (c) Indian state names in location string.
    country = (profile.get("country") or "").strip().lower()
    location = profile.get("location")
    if location:
        loc_lower = location.strip().lower()
        in_india = (
            country == "india"
            or any(city in loc_lower for city in _INDIA_LOCATIONS_LOWER)
            or any(state in loc_lower for state in _INDIA_STATES_LOWER)
            or "india" in loc_lower
        )
        if not in_india:
            return "Based outside India"

    return None

