"""
audit_top100.py
===============
FORENSIC HONEYPOT AUDIT — ALL 100 SUBMISSION CANDIDATES
Reference date: 2026-06-16

Checks that honeypot.py DOES NOT cover:
  - TIMELINE: started 2020, claims 10 years experience (impossible in 2026 = only 6 yrs)
  - CAREER SPAN vs CLAIMED YEARS (tight check, no 5yr buffer)
  - EDUCATION TIMELINE: graduated 2022, but started career 2019 (impossible)
  - DURATION SUM vs CLAIMED YEARS (padded roles)
  - SKILL DURATION > CAREER DURATION (claimed 10yr skill, 5yr career)

Run: python audit_top100.py
"""

import json
import csv
from datetime import date, datetime
from pathlib import Path

CANDIDATES_PATH = Path("India_runs_data_and_ai_challenge/candidates.jsonl")
SUBMISSION_PATH = Path("submission.csv")
REFERENCE_DATE  = date(2026, 6, 16)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(s):
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

def years_ago(d):
    """How many years before REFERENCE_DATE is date d?"""
    if d is None:
        return None
    return (REFERENCE_DATE - d).days / 365.25

# ─────────────────────────────────────────────────────────────────────────────
# NEW CHECK A — Career-Start vs Claimed Years (The 2020 → 10yr example)
# "Person started work 2020, claims 10 years experience in 2026 = IMPOSSIBLE"
# Max possible experience = years since earliest career start date
# ─────────────────────────────────────────────────────────────────────────────

def check_A_timeline_experience(c):
    """
    Checks if claimed years_of_experience > actual years since first job start.
    Tolerance: 1.0 year (account for data lag).
    """
    profile   = c.get("profile") or {}
    career    = c.get("career_history") or []
    claimed   = profile.get("years_of_experience")
    if claimed is None or not career:
        return None

    start_dates = []
    for role in career:
        sd = parse_date(role.get("start_date"))
        if sd:
            start_dates.append(sd)

    if not start_dates:
        return None

    earliest = min(start_dates)
    max_possible = years_ago(earliest)   # years since first job in 2026

    overshoot = float(claimed) - max_possible
    if overshoot > 1.0:   # allow 1yr tolerance
        return {
            "check": "A_timeline_experience",
            "severity": "CRITICAL" if overshoot > 3 else "HIGH",
            "detail": (
                f"Claims {claimed}yr but first job started {earliest} "
                f"= max possible {max_possible:.1f}yr in 2026. "
                f"Overshoot: {overshoot:.1f}yr"
            )
        }
    return None

# ─────────────────────────────────────────────────────────────────────────────
# NEW CHECK B — Education Graduation vs Career Start
# "Graduated 2022 but career history shows job started 2019 = IMPOSSIBLE"
# ─────────────────────────────────────────────────────────────────────────────

def check_B_education_career_paradox(c):
    """
    Checks if any career role started BEFORE the candidate finished their
    highest education degree (with a 1-year gap allowance for internships).
    """
    education = c.get("education") or []
    career    = c.get("career_history") or []

    if not education or not career:
        return None

    # Find the latest graduation year
    grad_years = [e.get("end_year") for e in education if e.get("end_year")]
    if not grad_years:
        return None
    latest_grad = max(grad_years)

    # Find the earliest career start
    start_dates = []
    for role in career:
        sd = parse_date(role.get("start_date"))
        if sd:
            start_dates.append(sd)
    if not start_dates:
        return None

    earliest_start = min(start_dates)
    # Allow 1 year gap (internship / gap year before final graduation)
    if earliest_start.year < (latest_grad - 1):
        paradox_years = (latest_grad - 1) - earliest_start.year
        return {
            "check": "B_education_career_paradox",
            "severity": "HIGH" if paradox_years > 2 else "MEDIUM",
            "detail": (
                f"Latest degree ended {latest_grad} but career started "
                f"{earliest_start} — {paradox_years}yr paradox"
            )
        }
    return None

# ─────────────────────────────────────────────────────────────────────────────
# NEW CHECK C — Duration Sum vs Claimed Years (Padded Roles)
# Sum of all role duration_months >> claimed years_of_experience × 12
# (impossible if roles are claimed sequentially, not overlapping)
# ─────────────────────────────────────────────────────────────────────────────

def check_C_duration_inflation(c):
    """
    Checks if total role months >> claimed experience.
    Ratio > 2.5x is impossible for sequential career.
    """
    profile  = c.get("profile") or {}
    career   = c.get("career_history") or []
    claimed  = profile.get("years_of_experience")
    if claimed is None or not career:
        return None

    total_months = sum((r.get("duration_months") or 0) for r in career)
    claimed_months = float(claimed) * 12

    if claimed_months < 6:
        return None  # avoid division by zero edge cases

    ratio = total_months / claimed_months
    if ratio > 2.5:
        return {
            "check": "C_duration_inflation",
            "severity": "CRITICAL" if ratio > 4 else "HIGH",
            "detail": (
                f"Total role months={total_months} vs claimed "
                f"{claimed}yr={claimed_months:.0f}mo. "
                f"Ratio: {ratio:.1f}x (impossible for sequential career)"
            )
        }
    return None

# ─────────────────────────────────────────────────────────────────────────────
# NEW CHECK D — Skill Duration > Entire Career Duration
# "10yr skill duration but candidate only has 5yr career" = impossible
# ─────────────────────────────────────────────────────────────────────────────

def check_D_skill_duration_paradox(c):
    """
    Checks if any skill claims duration_months > the candidate's actual
    career span in months (derived from start dates, not claimed).
    """
    career = c.get("career_history") or []
    skills = c.get("skills") or []
    if not career or not skills:
        return None

    start_dates = [parse_date(r.get("start_date")) for r in career]
    start_dates = [d for d in start_dates if d]
    if not start_dates:
        return None

    earliest = min(start_dates)
    career_months = (REFERENCE_DATE - earliest).days / 30.44

    violations = []
    for s in skills:
        dur = s.get("duration_months") or 0
        name = s.get("name", "?")
        prof = s.get("proficiency", "?")
        # Only flag expert/advanced with duration > career span + 12mo tolerance
        if prof in ("expert", "advanced") and dur > (career_months + 12):
            violations.append(
                f"{name} ({prof}): {dur}mo skill vs {career_months:.0f}mo career"
            )

    if violations:
        return {
            "check": "D_skill_duration_paradox",
            "severity": "HIGH",
            "detail": f"Skill duration exceeds career span: {'; '.join(violations[:3])}"
        }
    return None

# ─────────────────────────────────────────────────────────────────────────────
# NEW CHECK E — Future or Impossible Dates
# ─────────────────────────────────────────────────────────────────────────────

def check_E_future_dates(c):
    """
    Checks for roles with start_date or end_date in the future (after 2026-06-16).
    Also checks for end_date < start_date (negative tenure).
    """
    career = c.get("career_history") or []
    issues = []

    for role in career:
        company = role.get("company", "?")
        title   = role.get("title", "?")
        sd = parse_date(role.get("start_date"))
        ed = parse_date(role.get("end_date"))
        is_current = role.get("is_current", False)

        if sd and sd > REFERENCE_DATE:
            issues.append(f"{company}/{title}: start_date {sd} is in the FUTURE")

        if ed and not is_current:
            if ed > REFERENCE_DATE:
                issues.append(f"{company}/{title}: end_date {ed} is in the FUTURE")
            if sd and ed < sd:
                issues.append(f"{company}/{title}: end_date {ed} < start_date {sd}")

    if issues:
        return {
            "check": "E_future_dates",
            "severity": "CRITICAL",
            "detail": "; ".join(issues[:3])
        }
    return None

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING SYSTEM CHECKS (re-implemented for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def check_existing_salary_inversion(c):
    sig = (c.get("redrob_signals") or {})
    sal = sig.get("expected_salary_range_inr_lpa") or {}
    mn, mx = sal.get("min"), sal.get("max")
    if mn is None or mx is None:
        return None
    try:
        mn, mx = float(mn), float(mx)
    except:
        return None
    if mn > mx:
        gap = mn - mx
        sev = "CRITICAL" if gap > 8 else "HIGH" if gap > 5 else "MEDIUM"
        return {"check": "EXISTING_salary_inversion", "severity": sev,
                "detail": f"Salary min={mn} > max={mx}, gap={gap:.1f} LPA"}
    return None

def check_existing_temporal_paradox(c):
    sig = (c.get("redrob_signals") or {})
    signup = parse_date(sig.get("signup_date"))
    active = parse_date(sig.get("last_active_date"))
    if not signup or not active:
        return None
    gap = (signup - active).days
    if gap > 0:
        sev = "CRITICAL" if gap > 75 else "HIGH" if gap > 30 else "MEDIUM"
        return {"check": "EXISTING_temporal_paradox", "severity": sev,
                "detail": f"last_active {gap} days BEFORE signup (dead zone 31-74d = currently uncaught by H2/L3)"}
    return None

def check_existing_h3_experience(c):
    """H3 from honeypot.py: claimed > span + 5.0 AND claimed > 10"""
    career = c.get("career_history") or []
    profile = c.get("profile") or {}
    claimed = profile.get("years_of_experience")
    if not claimed or not career:
        return None

    start_dates = [parse_date(r.get("start_date")) for r in career]
    end_dates   = []
    for r in career:
        ed = r.get("end_date")
        if ed is None:
            end_dates.append(REFERENCE_DATE)
        else:
            d = parse_date(ed)
            if d:
                end_dates.append(d)

    start_dates = [d for d in start_dates if d]
    if not start_dates or not end_dates:
        return None

    span = (max(end_dates) - min(start_dates)).days / 365.25
    if float(claimed) > (span + 5.0) and float(claimed) > 10.0:
        return {"check": "EXISTING_H3_chronological", "severity": "CRITICAL",
                "detail": f"H3: claims {claimed}yr, span={span:.1f}yr, buffer=5yr, min=10yr → WOULD FLAG"}
    return None

def check_existing_expert_fabrication(c):
    """M1: 5+ expert/advanced skills with duration < 3 months"""
    skills = c.get("skills") or []
    count = sum(1 for s in skills
                if (s.get("proficiency") or "").lower() in ("expert","advanced")
                and (s.get("duration_months") or 0) < 3)
    if count >= 5:
        return {"check": "EXISTING_M1_fabrication", "severity": "HIGH",
                "detail": f"M1: {count} expert/advanced skills with <3mo duration"}
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MASTER AUDIT
# ─────────────────────────────────────────────────────────────────────────────

ALL_CHECKS = [
    check_A_timeline_experience,
    check_B_education_career_paradox,
    check_C_duration_inflation,
    check_D_skill_duration_paradox,
    check_E_future_dates,
    check_existing_salary_inversion,
    check_existing_temporal_paradox,
    check_existing_h3_experience,
    check_existing_expert_fabrication,
]

def audit_candidate(c):
    results = []
    for fn in ALL_CHECKS:
        r = fn(c)
        if r:
            results.append(r)
    return results


def main():
    # Load submission rankings
    submission = {}
    with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            submission[row["candidate_id"]] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
            }

    print(f"Submission has {len(submission)} candidates.\n")
    print("Scanning full dataset for top-100 candidates...\n")

    findings = {}  # cand_id → list of issues

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = c.get("candidate_id", "")
            if cid not in submission:
                continue  # only audit top-100

            issues = audit_candidate(c)
            if issues:
                findings[cid] = {
                    "rank": submission[cid]["rank"],
                    "score": submission[cid]["score"],
                    "issues": issues,
                }

    # ── Print Results ──────────────────────────────────────────────────────
    print("=" * 80)
    print("FORENSIC AUDIT RESULTS — ALL 100 SUBMISSION CANDIDATES")
    print(f"Reference date: {REFERENCE_DATE}")
    print("=" * 80)

    if not findings:
        print("\n✅  NO ISSUES FOUND — All 100 candidates passed all checks.")
    else:
        # Sort by rank
        sorted_findings = sorted(findings.items(), key=lambda x: x[1]["rank"])

        critical_count = 0
        high_count = 0
        medium_count = 0

        for cid, data in sorted_findings:
            rank  = data["rank"]
            score = data["score"]
            issues = data["issues"]

            severities = [i["severity"] for i in issues]
            worst = ("CRITICAL" if "CRITICAL" in severities
                     else "HIGH" if "HIGH" in severities
                     else "MEDIUM")

            if worst == "CRITICAL": critical_count += 1
            elif worst == "HIGH":   high_count += 1
            else:                   medium_count += 1

            flag = "[CRITICAL]" if worst == "CRITICAL" else "[HIGH]    " if worst == "HIGH" else "[MEDIUM]  "
            print(f"\n{flag} Rank {rank:>3} | {cid} | score={score:.4f} | worst={worst}")
            for iss in issues:
                sev_icon = "[CRIT]" if iss["severity"] == "CRITICAL" else "[HIGH]" if iss["severity"] == "HIGH" else "[MED] "
                print(f"     {sev_icon} [{iss['check']}] {iss['detail']}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Total candidates with issues : {len(findings)} / {len(submission)}")
        print(f"  CRITICAL issues              : {critical_count}")
        print(f"  HIGH issues                  : {high_count}")
        print(f"  MEDIUM issues                : {medium_count}")

        # Check if any critical ones are in Top 10
        top10_issues = {k: v for k, v in findings.items() if v["rank"] <= 10}
        if top10_issues:
            print(f"\n  ⚠️  TOP-10 CANDIDATES WITH ISSUES: {len(top10_issues)}")
            for cid, data in sorted(top10_issues.items(), key=lambda x: x[1]["rank"]):
                sev_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}
                worst_sev = max((i["severity"] for i in data["issues"]),
                                key=lambda s: sev_order.get(s, 0))
                print(f"     Rank {data['rank']}: {cid} — {worst_sev}")
        else:
            print("\n  ✅  No issues in Top 10.")

        # Disqualification risk
        print("\n  DISQUALIFICATION RISK:")
        print("  The hackathon spec says ~10% of 100K data = ~10,000 honeypots.")
        print("  If ANY honeypot appears in your top-100 submission,")
        print("  you may be disqualified or severely penalized.")
        if critical_count > 0:
            print(f"  🚨 {critical_count} CRITICAL candidates need immediate review.")
        elif high_count > 0:
            print(f"  ⚠️  {high_count} HIGH-severity candidates need review before submission.")
        else:
            print("  ✅  No critical risks detected. Review MEDIUM cases manually.")

    print("\nDone.")


if __name__ == "__main__":
    main()
