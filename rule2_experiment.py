import time
import os
import psutil
import gc

from src.loader import load_candidates, load_company_classifications
from src.honeypot import flag_honeypots
from src.embedding_scorer import compute_embedding_scores
from src.skill_scorer import compute_skill_score
from src.career_scorer import compute_career_score
from src.alignment_scorer import compute_alignment_score
import src.contradiction
from src.behavioral import compute_behavioral_multiplier

original_compute_contradiction = src.contradiction.compute_contradiction_penalty

def compute_no_rule2(candidate, career_bd):
    mult, reasons = original_compute_contradiction(candidate, career_bd)
    if "High skill count (>=10) but zero assessments taken" in reasons:
        reasons.remove("High skill count (>=10) but zero assessments taken")
        mult /= 0.85
    return mult, reasons

def run_ranking(candidates, company_map, honeypot_flags, use_rule2=True):
    print(f"\nRunning ranking (Rule 2 Enabled: {use_rule2})")
    stage1_results = []
    
    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id", f"UNKNOWN_{i}")
        is_honeypot, hp_reason = honeypot_flags.get(cid, (False, None))

        if is_honeypot:
            stage1_results.append({
                "candidate_id": cid, 
                "stage1_score": 0.0, 
                "is_honeypot": True, 
                "candidate": candidate
            })
            continue

        skill_score, skill_bd = compute_skill_score(candidate)
        career_score, career_bd = compute_career_score(candidate, company_map)
        alignment_score, align_bd = compute_alignment_score(candidate, skill_bd)
        
        if use_rule2:
            contra_mult, contra_reasons = original_compute_contradiction(candidate, career_bd)
        else:
            contra_mult, contra_reasons = compute_no_rule2(candidate, career_bd)
            
        signals = candidate.get("redrob_signals") or {}
        behav_mult, behav_bd = compute_behavioral_multiplier(signals)

        raw_fit_stage1 = skill_score + career_score + alignment_score
        stage1_score = raw_fit_stage1 * contra_mult * behav_mult

        # Also store original properties for analysis
        skills = candidate.get("skills", [])
        assessments = (candidate.get("profile") or {}).get("assessments", [])

        stage1_results.append({
            "candidate_id": cid,
            "stage1_score": stage1_score,
            "skill_score": skill_score,
            "career_score": career_score,
            "alignment_score": alignment_score,
            "contra_mult": contra_mult,
            "behav_mult": behav_mult,
            "is_honeypot": is_honeypot,
            "candidate": candidate,
            "skill_count": len(skills),
            "assess_count": len(assessments)
        })

    stage1_results.sort(key=lambda r: (-r.get("stage1_score", 0.0), r["candidate_id"]))
    
    top_1000 = stage1_results[:1000]
    top_1000_candidates = [r["candidate"] for r in top_1000]
    
    print(f"Computing embeddings for top 1000...")
    embedding_scores = compute_embedding_scores(top_1000_candidates)
    
    final_results = []
    for i, r in enumerate(top_1000):
        if r["is_honeypot"]:
            final_score = 0.0
        else:
            emb_score = embedding_scores[i]
            raw_fit = r["skill_score"] + r["career_score"] + r["alignment_score"] + emb_score
            final_score = raw_fit * r["contra_mult"] * r["behav_mult"]
            
        r["final_score"] = round(final_score, 4)
        final_results.append(r)
        
    final_results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    # Pre-compute rank map
    rank_map = {r["candidate_id"]: rank for rank, r in enumerate(final_results, 1)}
    
    return final_results, rank_map

def main():
    print("Loading data...")
    candidates = load_candidates("candidates.jsonl")
    company_map = load_company_classifications("data/company_classifications.json")
    honeypot_flags = flag_honeypots(candidates)
    
    res_A, rank_A = run_ranking(candidates, company_map, honeypot_flags, use_rule2=True)
    res_B, rank_B = run_ranking(candidates, company_map, honeypot_flags, use_rule2=False)
    
    top_100_A = res_A[:100]
    top_100_B = res_B[:100]
    
    top_100_A_ids = set([r["candidate_id"] for r in top_100_A])
    top_100_B_ids = set([r["candidate_id"] for r in top_100_B])
    
    overlap_100 = len(top_100_A_ids.intersection(top_100_B_ids))
    entering = top_100_B_ids - top_100_A_ids
    leaving = top_100_A_ids - top_100_B_ids
    
    # Affected candidates
    # A candidate is affected if their contra_mult in B is different from A
    affected_ids = set()
    score_increases = []
    
    # We can detect affected globally by checking all stage1 results
    for r_A in res_A: # this is only top 1000 of A!
        cid = r_A["candidate_id"]
        # wait, we need global affected count.
        pass
        
    # Let's run a quick global scan for affected
    global_affected_count = 0
    for c in candidates:
        skills = c.get("skills", [])
        assessments = (c.get("profile") or {}).get("assessments", [])
        if len(skills) >= 10 and len(assessments) == 0:
            global_affected_count += 1
            
    # Calculate affected in Top N of B
    affected_in_top100 = sum(1 for r in res_B[:100] if r["skill_count"] >= 10 and r["assess_count"] == 0)
    affected_in_top500 = sum(1 for r in res_B[:500] if r["skill_count"] >= 10 and r["assess_count"] == 0)
    affected_in_top1000 = sum(1 for r in res_B[:1000] if r["skill_count"] >= 10 and r["assess_count"] == 0)
    
    rank_moves = []
    # For every candidate in B's top 100, calculate rank movement from A
    # Note: if they were not in A's top 1000, we consider their rank > 1000 (say 1001 for math)
    for rank_b, r_B in enumerate(top_100_B, 1):
        cid = r_B["candidate_id"]
        rank_a = rank_A.get(cid, 1001)
        movement = rank_a - rank_b
        rank_moves.append(abs(movement))
        if r_B["skill_count"] >= 10 and r_B["assess_count"] == 0:
            # How much did score increase? B_score - A_score
            # A_score might be missing if they weren't in A's top 1000, but we can compute it
            a_score = r_B["final_score"] * 0.85
            score_increases.append(r_B["final_score"] - a_score)
            
    avg_rank_move = sum(rank_moves) / len(rank_moves) if rank_moves else 0
    max_rank_move = max(rank_moves) if rank_moves else 0
    avg_score_inc = sum(score_increases) / len(score_increases) if score_increases else 0
    
    out = []
    out.append("# Contradiction Rule #2 Impact Analysis")
    out.append(f"- **Top 100 overlap percentage**: {overlap_100}%")
    out.append(f"- **Candidates entering Top 100**: {len(entering)}")
    out.append(f"- **Candidates leaving Top 100**: {len(leaving)}")
    out.append(f"- **Average rank movement (in new Top 100)**: {avg_rank_move:.1f} places")
    out.append(f"- **Maximum rank movement**: {max_rank_move} places")
    out.append(f"- **Number of Top 100 candidates affected (in new list)**: {affected_in_top100}")
    out.append(f"- **Average score increase for affected**: {avg_score_inc:.4f}")
    out.append("- **Number of affected candidates in**:")
    out.append(f"  - Top 100: {affected_in_top100}")
    out.append(f"  - Top 500: {affected_in_top500}")
    out.append(f"  - Top 1000: {affected_in_top1000}")
    
    out.append("\n## Candidates Entering Top 100 after disabling Rule #2")
    for cid in entering:
        r_B = next(r for r in top_100_B if r["candidate_id"] == cid)
        rank_a = rank_A.get(cid, ">1000")
        rank_b = rank_B[cid]
        # To get final score before, we just multiply new score by 0.85
        score_after = r_B["final_score"]
        score_before = score_after * 0.85
        out.append(f"- **{cid}**")
        out.append(f"  - Previous rank: {rank_a}")
        out.append(f"  - New rank: {rank_b}")
        out.append(f"  - Skill count: {r_B['skill_count']}")
        out.append(f"  - Assessment count: {r_B['assess_count']}")
        out.append(f"  - Final score before: {score_before:.4f}")
        out.append(f"  - Final score after: {score_after:.4f}")
        
    with open("c:/Users/Akanksha Shirke/.gemini/antigravity-ide/brain/bbed9454-3261-48a5-b6e7-3ffdf2f92f50/rule2_analysis.md", "w") as f:
        f.write("\n".join(out))
    print("Done! Wrote rule2_analysis.md")

if __name__ == "__main__":
    main()
