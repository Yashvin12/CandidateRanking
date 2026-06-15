"""Quick validation that all bug fixes are in effect."""
import sys
sys.path.insert(0, 'c:/dev/CandidateRanking')

from src.config import REFERENCE_DATE, NICE_TO_HAVE_SKILLS
from src.career_scorer import _count_kw_hits_deduped, _dimension_description
from src.contradiction import compute_contradiction_penalty
from src.skill_scorer import compute_skill_score
from src.embedding_scorer import _MAX_TEXT_LEN
from src.config import PRODUCTION_EVIDENCE_KEYWORDS

print("=== Config ===")
print(f"REFERENCE_DATE: {REFERENCE_DATE}")
assert REFERENCE_DATE == "2026-06-15", f"FAIL: expected 2026-06-15 got {REFERENCE_DATE}"
print("PASS: REFERENCE_DATE updated to 2026-06-15")

has_marketplace = "marketplace" in NICE_TO_HAVE_SKILLS
assert not has_marketplace, "FAIL: 'marketplace' still in NICE_TO_HAVE_SKILLS"
print("PASS: 'marketplace' removed from NICE_TO_HAVE_SKILLS")

print()
print("=== Embedding ===")
assert _MAX_TEXT_LEN == 2048, f"FAIL: expected 2048 got {_MAX_TEXT_LEN}"
print(f"PASS: _MAX_TEXT_LEN={_MAX_TEXT_LEN}")

print()
print("=== Career scorer dedup test (3A/3B) ===")
roles = [
    {"description": "owned and deployed production system at scale"},
    {"description": "owned and deployed production system at scale"},  # recycled
    {"description": "owned and deployed production system at scale"},  # recycled
]
prod_kw = [k.lower() for k in PRODUCTION_EVIDENCE_KEYWORDS]
count_3roles = _count_kw_hits_deduped(roles, prod_kw)
count_1role  = _count_kw_hits_deduped(roles[:1], prod_kw)
assert count_3roles == count_1role, f"FAIL: 3 recycled roles gave {count_3roles} != {count_1role}"
print(f"PASS: 3 recycled descriptions give same count ({count_3roles}) as 1 role")

print()
print("=== Contradiction 5C test (count=0.5 should give 0.85) ===")
cand = {
    "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 24}],
    "redrob_signals": {"signup_date": "2024-01-01"},
    "profile": {
        "summary": "marketing manager with 5 years experience",
        "current_title": "senior engineer",
        "years_of_experience": 5,
    },
    "career_history": [{"duration_months": 60}],
}
mult, reasons = compute_contradiction_penalty(cand, {})
print(f"Multiplier: {mult}, Reasons: {reasons}")
assert mult == 0.85, f"FAIL: expected 0.85 for count=0.5, got {mult}"
print("PASS: 0.5 count correctly maps to 0.85 multiplier")

print()
print("=== Contradiction 5B test (2026 signup skips Check 7) ===")
cand_new = {
    "skills": [],
    "redrob_signals": {"signup_date": "2026-05-01"},  # recent signup (< 6 months)
    "profile": {
        "summary": "marketing manager with 5 years experience",
        "current_title": "senior engineer",
        "years_of_experience": 5,
    },
    "career_history": [{"duration_months": 60}],
}
mult2, reasons2 = compute_contradiction_penalty(cand_new, {})
# With 2026 signup, Check 7 should not fire
check7_fired = any("Summary mentions" in r for r in reasons2)
assert not check7_fired, f"FAIL: Check 7 fired for a 2026 signup: {reasons2}"
print(f"PASS: Check 7 skipped for 2026 signup (mult={mult2}, reasons={reasons2})")

print()
print("=== Contradiction 5A test (10 skills threshold now 20) ===")
cand_10skills = {
    "skills": [{"name": f"Skill{i}", "proficiency": "intermediate", "duration_months": 12} for i in range(15)],
    "redrob_signals": {"signup_date": "2023-01-01", "skill_assessment_scores": None},
    "profile": {"summary": "", "current_title": "engineer", "years_of_experience": 5},
    "career_history": [{"duration_months": 60}],
}
mult3, reasons3 = compute_contradiction_penalty(cand_10skills, {})
check2_fired = any("skills claimed but zero assessment" in r for r in reasons3)
assert not check2_fired, f"FAIL: Check 2 fired for only 15 skills: {reasons3}"
print(f"PASS: Check 2 did NOT fire for 15 skills (threshold is now 20): mult={mult3}")

print()
print("=== Skill scorer Python fallback test (2B) ===")
cand_no_python_skill = {
    "skills": [{"name": "TensorFlow", "proficiency": "expert", "duration_months": 24}],
    "redrob_signals": {},
    "career_history": [{"description": "Built ML pipelines using Python and TensorFlow"}],
}
score, bd = compute_skill_score(cand_no_python_skill)
python_group_score = bd.get("python", 0.0)
assert python_group_score == 7.0, f"FAIL: Python fallback gave {python_group_score} (expected 7.0)"
print(f"PASS: Python description fallback gave group score={python_group_score}")

print()
print("=" * 40)
print("ALL BUG FIX VALIDATIONS PASSED")
print("=" * 40)
