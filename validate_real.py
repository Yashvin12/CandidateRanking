import json
import time
from src.loader import load_candidates
from src.honeypot import _check_experience_paradox
from src.contradiction import compute_contradiction_penalty
from src.career_scorer import _dimension_description
from src.embedding_scorer import compute_embedding_scores
from src.config import PRODUCTION_EVIDENCE_KEYWORDS

print("Loading real candidates.jsonl dataset...")
candidates = load_candidates("candidates.jsonl")
total_candidates = len(candidates)

print(f"\n--- 1. Experience Paradox ---")
hp_flagged = 0
examples_hp = []
for c in candidates:
    res = _check_experience_paradox(c)
    if res is not None:
        hp_flagged += 1
        if len(examples_hp) < 20:
            claimed = c.get("profile", {}).get("years_of_experience")
            history_sum = sum(r.get("duration_months") or 0 for r in c.get("career_history", []))
            examples_hp.append((c.get("candidate_id"), claimed, history_sum))

print(f"Number flagged: {hp_flagged}")
if total_candidates > 0:
    print(f"Percentage flagged: {(hp_flagged/total_candidates)*100:.2f}%")
print("20 Example Candidates (ID, Claimed Years, Total Career Months):")
for ex in examples_hp:
    claimed_val = ex[1] if ex[1] is not None else 0
    print(f"  {ex[0]}: Claimed {claimed_val} yrs ({float(claimed_val)*12:.0f} mo), History sums to {ex[2]} mo")

print(f"\n--- 2. Contradiction Rule #2 (>=10 skills, no assessments) ---")
c2_flagged = 0
skill_counts = {}
assess_stats = {"has_assessments": 0, "no_assessments": 0}

for c in candidates:
    num_skills = len(c.get("skills") or [])
    skill_counts[num_skills] = skill_counts.get(num_skills, 0) + 1
    
    signals = c.get("redrob_signals") or {}
    assessments = signals.get("skill_assessment_scores")
    
    if assessments:
        assess_stats["has_assessments"] += 1
    else:
        assess_stats["no_assessments"] += 1
        
    if num_skills >= 10 and not assessments:
        c2_flagged += 1

print(f"Number flagged: {c2_flagged}")
if total_candidates > 0:
    print(f"Percentage flagged: {(c2_flagged/total_candidates)*100:.2f}%")
print(f"Distribution of skills count: {dict(sorted(skill_counts.items()))}")
print(f"Assessment availability statistics: {assess_stats}")

print(f"\n--- 3. Contradiction Rule #5 (Career Gap) ---")
c5_flagged = 0
examples_c5 = []
career_flags = {"title_description_mismatch": False}
for c in candidates:
    mult, reasons = compute_contradiction_penalty(c, career_flags)
    is_c5 = any("accounts for" in r for r in reasons)
    if is_c5:
        c5_flagged += 1
        if len(examples_c5) < 5:
            gap_reason = next(r for r in reasons if "accounts for" in r)
            examples_c5.append((c.get("candidate_id"), gap_reason))

print(f"Number flagged: {c5_flagged}")
if total_candidates > 0:
    print(f"Percentage flagged: {(c5_flagged/total_candidates)*100:.2f}%")
print("Examples:")
for ex in examples_c5:
    print(f"  {ex[0]}: {ex[1]}")

print(f"\n--- 4. Career Keyword Inflation ---")
_PROD_KW_LOWER = [kw.lower() for kw in PRODUCTION_EVIDENCE_KEYWORDS]
beneficiaries = []
for c in candidates:
    career_history = c.get("career_history") or []
    if not career_history:
        continue
        
    combined = " ".join(role.get("description") or "" for role in career_history).lower()
    total_occurrences = sum(combined.count(kw) for kw in _PROD_KW_LOWER)
    
    prof = c.get("profile", {})
    if prof is None:
        prof = {}
    title = prof.get("current_title", "")
    bonus, _, _, _ = _dimension_description(title, career_history)
    
    if total_occurrences > 0:
        beneficiaries.append((c.get("candidate_id"), total_occurrences, bonus))

beneficiaries.sort(key=lambda x: -x[1])
print("Top 20 candidates with highest frequency of production keywords in descriptions:")
for b in beneficiaries[:20]:
    print(f"  {b[0]}: keyword occurrences = {b[1]} -> Production Bonus = +{b[2]}")

print(f"\n--- 5. Embedding Runtime Benchmark ---")
try:
    from sentence_transformers import SentenceTransformer
    has_st = True
except ImportError:
    has_st = False

if not has_st:
    print("sentence-transformers not installed. Environment is missing the package.")
else:
    subset_100 = candidates[:100]
    subset_1000 = candidates[:1000]

    t0 = time.time()
    compute_embedding_scores(subset_100)
    t_100 = time.time() - t0

    t0 = time.time()
    compute_embedding_scores(subset_1000)
    t_1000 = time.time() - t0

    rate = 1000 / t_1000 if t_1000 > 0 else 0
    proj = total_candidates / rate if rate > 0 else 0

    print(f"100 candidates: {t_100:.2f} seconds")
    print(f"1000 candidates: {t_1000:.2f} seconds")
    print(f"candidates/sec: {rate:.1f}")
    print(f"Projected runtime for {total_candidates}: {proj:.1f} seconds ({proj/60:.2f} minutes)")
