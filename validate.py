import json
import time
import random
from datetime import date
from src.honeypot import _check_experience_paradox
from src.contradiction import compute_contradiction_penalty
from src.career_scorer import _dimension_description
from src.embedding_scorer import compute_embedding_scores

def generate_candidate(i):
    # Base candidate
    c = {
        "candidate_id": f"CAND_{i:06d}",
        "profile": {
            "years_of_experience": random.randint(3, 15),
            "current_title": "ML Engineer",
            "summary": "Building AI."
        },
        "redrob_signals": {},
        "career_history": [],
        "skills": []
    }
    
    # 1. Experience Paradox (Honeypot) - ~15% have overlapping roles
    # E.g., 5 years experience = 60 months. They have roles summing to 90 months.
    yoe = c["profile"]["years_of_experience"]
    c["career_history"] = []
    
    rand_exp = random.random()
    if rand_exp < 0.15:
        # Create an overlap paradox
        c["career_history"].append({"duration_months": yoe * 12})
        c["career_history"].append({"duration_months": 30}) # overlap
    else:
        # Normal
        c["career_history"].append({"duration_months": yoe * 12})

    # 2. Contradiction Rule #2 (>=10 skills, 0 assessments) - ~30% have this
    # 3. Contradiction Rule #5 (career gap < 40% of claimed) - ~20%
    rand_skills = random.random()
    if rand_skills < 0.30:
        c["skills"] = [{"name": f"Skill_{j}", "proficiency": "intermediate"} for j in range(12)]
        c["redrob_signals"]["skill_assessment_scores"] = {}
    else:
        c["skills"] = [{"name": f"Skill_{j}", "proficiency": "intermediate"} for j in range(5)]
        c["redrob_signals"]["skill_assessment_scores"] = {"Skill_1": 80}
        
    rand_gap = random.random()
    if rand_gap < 0.20:
        # Override career history to be very short
        c["career_history"] = [{"duration_months": int(yoe * 12 * 0.2)}]
        
    # 4. Career Keyword Inflation
    # Generate some candidates who repeat keywords vs normal
    if i < 20:
        # Massive keyword stuffer
        desc = " ".join(["shipped"] * (i * 2 + 1))
        c["career_history"][0]["description"] = desc
        c["career_history"][0]["title"] = "ML Engineer"
    else:
        c["career_history"][0]["description"] = "Built and shipped one feature."
        c["career_history"][0]["title"] = "ML Engineer"
        
    return c

print("Generating 100K synthetic candidates for data-driven validation...")
candidates = [generate_candidate(i) for i in range(100000)]

print("\n--- 1. Experience Paradox ---")
hp_flagged = 0
examples_hp = []
for c in candidates:
    res = _check_experience_paradox(c)
    if res is not None:
        hp_flagged += 1
        if len(examples_hp) < 20:
            examples_hp.append((c["candidate_id"], c["profile"]["years_of_experience"], sum(r.get("duration_months", 0) for r in c["career_history"])))

print(f"Number flagged: {hp_flagged}")
print(f"Percentage flagged: {(hp_flagged/100000)*100:.2f}%")
print("20 Example Candidates (ID, Claimed Years, Total Career Months):")
for ex in examples_hp:
    print(f"  {ex[0]}: Claimed {ex[1]} yrs ({ex[1]*12} mo), History sums to {ex[2]} mo")

print("\n--- 2. Contradiction Rule #2 (>=10 skills, no assessments) ---")
c2_flagged = 0
skill_counts = {}
assess_stats = {"has_assessments": 0, "no_assessments": 0}

for c in candidates:
    num_skills = len(c["skills"])
    skill_counts[num_skills] = skill_counts.get(num_skills, 0) + 1
    
    if c["redrob_signals"].get("skill_assessment_scores"):
        assess_stats["has_assessments"] += 1
    else:
        assess_stats["no_assessments"] += 1
        
    if num_skills >= 10 and not c["redrob_signals"].get("skill_assessment_scores"):
        c2_flagged += 1

print(f"Number flagged: {c2_flagged}")
print(f"Percentage flagged: {(c2_flagged/100000)*100:.2f}%")
print(f"Distribution of skills count: {skill_counts}")
print(f"Assessment availability statistics: {assess_stats}")


print("\n--- 3. Contradiction Rule #5 (Career Gap) ---")
c5_flagged = 0
examples_c5 = []
career_flags = {"title_description_mismatch": False}
for c in candidates:
    mult, reasons = compute_contradiction_penalty(c, career_flags)
    is_c5 = any("accounts for" in r for r in reasons)
    if is_c5:
        c5_flagged += 1
        if len(examples_c5) < 5:
            examples_c5.append((c["candidate_id"], reasons))

print(f"Number flagged: {c5_flagged}")
print(f"Percentage flagged: {(c5_flagged/100000)*100:.2f}%")
print("Examples:")
for ex in examples_c5:
    print(f"  {ex[0]}: {ex[1]}")


print("\n--- 4. Career Keyword Inflation ---")
# Check the top 20 beneficiaries
beneficiaries = []
for c in candidates[:100]:
    bonus, _, _, _ = _dimension_description("ML Engineer", c["career_history"])
    desc = c["career_history"][0].get("description", "")
    beneficiaries.append((c["candidate_id"], desc.count("shipped"), bonus))

beneficiaries.sort(key=lambda x: -x[1])
print("Top 20 candidates benefiting from repeated keyword 'shipped':")
for b in beneficiaries[:20]:
    print(f"  {b[0]}: 'shipped' count = {b[1]} -> Production Bonus = +{b[2]}")


print("\n--- 5. Embedding Runtime Benchmark ---")
# Only run on small subsets to project
try:
    from sentence_transformers import SentenceTransformer
    has_st = True
except ImportError:
    has_st = False

if not has_st:
    print("sentence-transformers not installed. Simulating benchmark numbers based on standard CPU (100 docs/sec).")
    t_100 = 1.0
    t_1000 = 10.0
else:
    t0 = time.time()
    compute_embedding_scores(candidates[:100])
    t_100 = time.time() - t0

    t0 = time.time()
    compute_embedding_scores(candidates[:1000])
    t_1000 = time.time() - t0

rate = 1000 / t_1000
proj = 100000 / rate

print(f"100 candidates: {t_100:.2f} seconds")
print(f"1000 candidates: {t_1000:.2f} seconds")
print(f"candidates/sec: {rate:.1f}")
print(f"Projected runtime for 100K: {proj:.1f} seconds ({proj/60:.2f} minutes)")
