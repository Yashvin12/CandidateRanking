import time
import os
import psutil
import gc
import statistics

from src.loader import load_candidates, load_company_classifications
from src.honeypot import flag_honeypots
from src.embedding_scorer import compute_embedding_scores
from src.skill_scorer import compute_skill_score
from src.career_scorer import compute_career_score
from src.alignment_scorer import compute_alignment_score
from src.contradiction import compute_contradiction_penalty
from src.behavioral import compute_behavioral_multiplier

def run_ranking(candidates, company_map, honeypot_flags, fix_behavioral=False):
    print(f"\nRunning ranking (Fix Behavioral: {fix_behavioral})")
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
        
        contra_mult, contra_reasons = compute_contradiction_penalty(candidate, career_bd)
        
        signals = candidate.get("redrob_signals") or {}
        real_behav_mult, behav_bd = compute_behavioral_multiplier(signals)
        
        if fix_behavioral:
            behav_mult = 1.0
        else:
            behav_mult = real_behav_mult

        raw_fit_stage1 = skill_score + career_score + alignment_score
        stage1_score = raw_fit_stage1 * contra_mult * behav_mult

        stage1_results.append({
            "candidate_id": cid,
            "stage1_score": stage1_score,
            "skill_score": skill_score,
            "career_score": career_score,
            "alignment_score": alignment_score,
            "contra_mult": contra_mult,
            "behav_mult": behav_mult,
            "real_behav_mult": real_behav_mult,
            "is_honeypot": is_honeypot,
            "candidate": candidate
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
    
    rank_map = {r["candidate_id"]: rank for rank, r in enumerate(final_results, 1)}
    
    global_results = {r["candidate_id"]: r for r in stage1_results}
    
    return final_results, rank_map, global_results

def percentile(data, percent):
    data.sort()
    k = (len(data) - 1) * percent
    f = int(k)
    c = f + 1
    if f == c: return data[f]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1

def main():
    print("Loading data...")
    candidates = load_candidates("candidates.jsonl")
    company_map = load_company_classifications("data/company_classifications.json")
    honeypot_flags = flag_honeypots(candidates)
    
    res_A, rank_A, global_A = run_ranking(candidates, company_map, honeypot_flags, fix_behavioral=False)
    res_B, rank_B, global_B = run_ranking(candidates, company_map, honeypot_flags, fix_behavioral=True)
    
    all_behav_mults = []
    for c in candidates:
        signals = c.get("redrob_signals") or {}
        mult, _ = compute_behavioral_multiplier(signals)
        all_behav_mults.append(mult)
        
    p10 = percentile(all_behav_mults, 0.10)
    p25 = percentile(all_behav_mults, 0.25)
    p50 = percentile(all_behav_mults, 0.50)
    p75 = percentile(all_behav_mults, 0.75)
    p90 = percentile(all_behav_mults, 0.90)
    
    top_100_A = res_A[:100]
    top_100_B = res_B[:100]
    top_20_A = res_A[:20]
    top_20_B = res_B[:20]
    
    top_100_A_ids = set([r["candidate_id"] for r in top_100_A])
    top_100_B_ids = set([r["candidate_id"] for r in top_100_B])
    top_20_A_ids = set([r["candidate_id"] for r in top_20_A])
    top_20_B_ids = set([r["candidate_id"] for r in top_20_B])
    
    overlap_100 = len(top_100_A_ids.intersection(top_100_B_ids))
    overlap_20 = len(top_20_A_ids.intersection(top_20_B_ids))
    
    entering = top_100_B_ids - top_100_A_ids
    leaving = top_100_A_ids - top_100_B_ids
    
    rank_moves = []
    score_increases = []
    affected_count = 0
    
    for rank_b, r_B in enumerate(top_100_B, 1):
        cid = r_B["candidate_id"]
        rank_a = rank_A.get(cid, 1001)
        rank_moves.append(abs(rank_a - rank_b))
        
        if r_B["real_behav_mult"] != 1.0:
            affected_count += 1
            a_score = r_B["final_score"] * r_B["real_behav_mult"]
            score_increases.append(r_B["final_score"] - a_score)

    avg_rank_move = sum(rank_moves) / len(rank_moves) if rank_moves else 0
    max_rank_move = max(rank_moves) if rank_moves else 0
    avg_score_inc = sum(score_increases) / len(score_increases) if score_increases else 0
    
    out = []
    out.append("# Behavioral Multiplier Impact Analysis")
    out.append(f"- **Top 20 overlap percentage**: {overlap_20 / 20 * 100:.1f}% ({overlap_20}/20)")
    out.append(f"- **Top 100 overlap percentage**: {overlap_100 / 100 * 100:.1f}% ({overlap_100}/100)")
    out.append(f"- **Candidates entering Top 100**: {len(entering)}")
    out.append(f"- **Candidates leaving Top 100**: {len(leaving)}")
    out.append(f"- **Average rank movement (in new Top 100)**: {avg_rank_move:.1f} places")
    out.append(f"- **Maximum rank movement**: {max_rank_move} places")
    out.append(f"- **Number of Top 100 candidates affected (mult != 1.0)**: {affected_count}")
    out.append(f"- **Average score increase for affected**: {avg_score_inc:.4f}")
    
    out.append("\n### Distribution of Behavioral Multipliers (All 100K)")
    out.append(f"- p10: {p10:.4f}")
    out.append(f"- p25: {p25:.4f}")
    out.append(f"- p50: {p50:.4f}")
    out.append(f"- p75: {p75:.4f}")
    out.append(f"- p90: {p90:.4f}")
    
    out.append("\n## Candidates Entering Top 100 after fixing behavioral to 1.0")
    for cid in entering:
        r_B = next(r for r in top_100_B if r["candidate_id"] == cid)
        rank_a = rank_A.get(cid, ">1000")
        rank_b = rank_B[cid]
        
        score_after = r_B["final_score"]
        behav_mult = r_B["real_behav_mult"]
        score_before = score_after * behav_mult
        
        out.append(f"- **{cid}**")
        out.append(f"  - Previous rank: {rank_a}")
        out.append(f"  - New rank: {rank_b}")
        out.append(f"  - Behavioral multiplier: {behav_mult:.4f}")
        out.append(f"  - Final score before: {score_before:.4f}")
        out.append(f"  - Final score after: {score_after:.4f}")
        
    with open("c:/Users/Akanksha Shirke/.gemini/antigravity-ide/brain/bbed9454-3261-48a5-b6e7-3ffdf2f92f50/behavioral_analysis.md", "w") as f:
        f.write("\n".join(out))
    print("Done! Wrote behavioral_analysis.md")

if __name__ == "__main__":
    main()
