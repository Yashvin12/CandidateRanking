"""
src/contradiction.py
====================
Catches "soft honeypots" — profiles whose individual data points are
plausible but whose *combination* is contradictory.  Returns a penalty
multiplier (0.5–1.0) applied to the candidate's composite score.

Seven checks are run; each increments a floating-point contradiction
counter.  The counter maps to the multiplier via fixed bands.

Usage
-----
    from src.contradiction import compute_contradiction_penalty

    multiplier, reasons = compute_contradiction_penalty(candidate, career_flags)
"""

from __future__ import annotations


# Roles that, when found in the summary but not the title, signal a
# summary-title mismatch.  Each entry is (summary_keyword, anti_title_kw).
# If summary contains the keyword AND title contains "engineer" or
# "scientist" (but NOT the keyword's domain word), it's a contradiction.
_ROLE_MISMATCHES: list[tuple[str, str]] = [
    ("marketing manager", "marketing"),
    ("accountant", "account"),
    ("sales executive", "sales"),
    ("hr manager", "hr"),
    ("civil engineer", "civil"),
    ("mechanical engineer", "mechanical"),
    ("graphic designer", "design"),
    ("content writer", "content"),
]


def compute_contradiction_penalty(
    candidate: dict,
    career_flags: dict,
) -> tuple[float, list[str]]:
    """Compute a contradiction-based penalty multiplier.

    Parameters
    ----------
    candidate:
        Full candidate dict.
    career_flags:
        Breakdown dict from ``career_scorer.compute_career_score``.
        Expected key: ``title_description_mismatch`` (bool).

    Returns
    -------
    tuple[float, list[str]]
        ``(multiplier, reasons)`` where multiplier ∈ [0.5, 1.0].
    """
    count: float = 0.0
    reasons: list[str] = []

    skills = candidate.get("skills") or []
    signals = candidate.get("redrob_signals") or {}
    profile = candidate.get("profile") or {}
    career_history = candidate.get("career_history") or []

    has_skills = len(skills) > 0

    # ── CHECK 1: Expert claims + zero duration ───────────────────────────
    if has_skills:
        suspect = 0
        for s in skills:
            prof = (s.get("proficiency") or "").lower()
            dur = s.get("duration_months") or 0
            if prof in ("expert", "advanced") and dur < 6:
                suspect += 1

        if suspect >= 6:
            count += 2
            reasons.append(f"{suspect} skills claimed advanced/expert with <6mo duration")
        elif suspect >= 3:
            count += 1
            reasons.append(f"{suspect} skills claimed advanced/expert with <6mo duration")

    # ── CHECK 2: Many skills + zero assessments ──────────────────────────
    if has_skills and len(skills) >= 10:
        assessments = signals.get("skill_assessment_scores")
        if not assessments:  # None, {}, or missing
            count += 1
            reasons.append(f"{len(skills)} skills claimed but zero assessment scores")

    # ── CHECK 3: Low profile completeness + expert claims ────────────────
    if has_skills:
        completeness = signals.get("profile_completeness_score")
        if completeness is not None:
            try:
                completeness = float(completeness)
            except (ValueError, TypeError):
                completeness = None

            if completeness is not None and completeness < 35:
                expert_count = sum(
                    1 for s in skills
                    if (s.get("proficiency") or "").lower() in ("expert", "advanced")
                )
                if expert_count >= 3:
                    count += 1
                    reasons.append(
                        f"Profile {completeness:.0f}% complete but claims "
                        f"{expert_count} advanced/expert skills"
                    )

    # ── CHECK 4: Title-description mismatch ──────────────────────────────
    if career_flags.get("title_description_mismatch", False):
        count += 1
        reasons.append("Title does not match career description content")

    # ── CHECK 5: Career history gap ──────────────────────────────────────
    if career_history:
        total_career_months = sum(
            (role.get("duration_months") or 0) for role in career_history
        )
        claimed_years = profile.get("years_of_experience")
        if claimed_years is not None:
            try:
                claimed_months = float(claimed_years) * 12
            except (ValueError, TypeError):
                claimed_months = None

            if claimed_months and claimed_months > 0:
                if total_career_months < claimed_months * 0.4:
                    pct = (total_career_months / claimed_months) * 100
                    count += 1
                    reasons.append(
                        f"Career history only accounts for {pct:.0f}% of claimed experience"
                    )

    # ── CHECK 6: High endorsements + low connections ─────────────────────
    endorsements = signals.get("endorsements_received")
    connections = signals.get("connection_count")
    if endorsements is not None and connections is not None:
        try:
            endorsements = int(endorsements)
            connections = int(connections)
        except (ValueError, TypeError):
            endorsements = None
            connections = None

        if endorsements is not None and connections is not None:
            if endorsements > 40 and connections < 30:
                count += 1
                reasons.append(
                    f"{endorsements} endorsements but only {connections} connections"
                )

    # ── CHECK 7: Summary mentions different role than title ──────────────
    summary = (profile.get("summary") or "").lower()
    title = (profile.get("current_title") or "").lower()

    if summary and title:
        for role_kw, domain_word in _ROLE_MISMATCHES:
            if role_kw in summary and domain_word not in title:
                # Hard contradiction: summary says one role, title says engineer/scientist
                if "engineer" in title or "scientist" in title:
                    count += 1
                    reasons.append(
                        f"Summary mentions '{role_kw}' but title is '{title}'"
                    )
                else:
                    count += 0.5
                break  # one mismatch is enough

    # ── Convert count → multiplier ───────────────────────────────────────
    if count == 0:
        multiplier = 1.0
    elif count <= 1:
        multiplier = 0.85
    elif count <= 2:
        multiplier = 0.7
    elif count <= 3:
        multiplier = 0.55
    else:
        multiplier = 0.5

    return multiplier, reasons
