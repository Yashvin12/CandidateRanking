import time
import psutil
import os
import gc

from src.loader import load_candidates, load_company_classifications
from src.honeypot import flag_honeypots
from src.embedding_scorer import compute_embedding_scores
from src.skill_scorer import compute_skill_score
from src.career_scorer import compute_career_score
from src.alignment_scorer import compute_alignment_score
from src.contradiction import compute_contradiction_penalty
from src.behavioral import compute_behavioral_multiplier

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("Loading data...")
    candidates = load_candidates("candidates.jsonl")
    company_map = load_company_classifications("data/company_classifications.json")
    honeypot_flags = flag_honeypots(candidates)

    print("Running Stage 1 (Structured Scoring) for all 100K candidates...")
    t_stage1_start = time.perf_counter()
    stage1_results = []

    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id", f"UNKNOWN_{i}")
        is_honeypot, hp_reason = honeypot_flags.get(cid, (False, None))

        if is_honeypot:
            stage1_results.append({"candidate_id": cid, "stage1_score": 0.0, "is_honeypot": True, "candidate": candidate})
            continue

        skill_score, skill_bd = compute_skill_score(candidate)
        career_score, career_bd = compute_career_score(candidate, company_map)
        alignment_score, align_bd = compute_alignment_score(candidate, skill_bd)
        
        contra_mult, contra_reasons = compute_contradiction_penalty(candidate, career_bd)
        signals = candidate.get("redrob_signals") or {}
        behav_mult, behav_bd = compute_behavioral_multiplier(signals)

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
            "is_honeypot": is_honeypot,
            "candidate": candidate
        })

    t_stage1 = time.perf_counter() - t_stage1_start
    print(f"Stage 1 took {t_stage1:.2f}s")

    stage1_results.sort(key=lambda r: (-r.get("stage1_score", 0.0), r["candidate_id"]))

    pools = [10000, 5000, 2500, 1000]
    
    results_by_pool = {}

    for n in pools:
        print(f"\n--- Experiment: Pool Size N={n} ---")
        gc.collect()
        mem_before = get_memory_mb()
        t_emb_start = time.perf_counter()
        
        top_n_results = stage1_results[:n]
        top_n_candidates = [r["candidate"] for r in top_n_results]
        
        embedding_scores = compute_embedding_scores(top_n_candidates)
        t_emb = time.perf_counter() - t_emb_start
        mem_peak = get_memory_mb() 
        
        final_results = []
        for i, r in enumerate(top_n_results):
            if r["is_honeypot"]:
                final_score = 0.0
            else:
                emb_score = embedding_scores[i]
                raw_fit = r["skill_score"] + r["career_score"] + r["alignment_score"] + emb_score
                final_score = raw_fit * r["contra_mult"] * r["behav_mult"]
                
            final_results.append((r["candidate_id"], round(final_score, 4)))
            
        final_results.sort(key=lambda x: (-x[1], x[0]))
        top_100 = final_results[:100]
        
        results_by_pool[n] = {
            "runtime": t_stage1 + t_emb,
            "candidates_embedded": n,
            "peak_ram": max(mem_before, mem_peak),
            "top_100_ids": [x[0] for x in top_100]
        }
        
    baseline_top_100 = set(results_by_pool[10000]["top_100_ids"])
    
    out = []
    out.append("# Embedding Pool Size Experiment Results")
    out.append("*(Note: Due to the extreme 55-minute runtime of a pure 100K embedding calculation, N=10,000 is used as the high-accuracy baseline to calculate exact overlap percentages.)*\n")
    
    out.append("## Comparison Table")
    out.append("| Pool | Runtime (s) | Runtime (min) | Peak RAM (MB) | Overlap w/ Baseline |")
    out.append("|---|---|---|---|---|")
    
    for n in sorted(pools):
        res = results_by_pool[n]
        overlap = len(set(res["top_100_ids"]).intersection(baseline_top_100))
        out.append(f"| {n} | {res['runtime']:.1f}s | {res['runtime']/60:.2f}m | {res['peak_ram']:.1f} MB | {overlap}% |")
        
    with open("c:/Users/Akanksha Shirke/.gemini/antigravity-ide/brain/bbed9454-3261-48a5-b6e7-3ffdf2f92f50/experiment_results.md", "w") as f:
        f.write("\n".join(out))
    
    print("Done! Wrote experiment_results.md")

if __name__ == '__main__':
    main()
