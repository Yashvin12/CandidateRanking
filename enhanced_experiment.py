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
    
    # Pre-record structured rank mapping
    structured_rank_map = {}
    for rnk, r in enumerate(stage1_results, 1):
        structured_rank_map[r["candidate_id"]] = rnk

    pools = [10000, 5000, 2500, 1000]
    
    results_by_pool = {}

    baseline_data = None
    baseline_top_100_ids = set()

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
                
            final_results.append({
                "candidate_id": r["candidate_id"],
                "final_score": round(final_score, 4),
                "embedding_score": emb_score if not r["is_honeypot"] else 0.0
            })
            
        final_results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        top_100 = final_results[:100]
        top_100_ids = [x["candidate_id"] for x in top_100]
        top_20_ids = [x["candidate_id"] for x in top_100[:20]]
        
        # Build rank and score maps for current run
        current_rank_map = {x["candidate_id"]: rank for rank, x in enumerate(top_100, 1)}
        current_score_map = {x["candidate_id"]: x["final_score"] for x in top_100}
        
        if n == 10000:
            baseline_data = {x["candidate_id"]: {"rank": rank, "score": x["final_score"], "emb_score": x["embedding_score"]} for rank, x in enumerate(top_100, 1)}
            baseline_top_100_ids = set(top_100_ids)
            baseline_top_20_ids = set(top_20_ids)
            
            results_by_pool[n] = {
                "runtime": t_stage1 + t_emb,
                "candidates_embedded": n,
                "peak_ram": max(mem_before, mem_peak),
                "top_100_overlap": 100,
                "top_20_overlap": 20,
                "avg_score_diff": 0.0,
                "max_rank_move": 0,
                "entering_leaving": 0,
                "missing": []
            }
        else:
            overlap_100 = len(set(top_100_ids).intersection(baseline_top_100_ids))
            overlap_20 = len(set(top_20_ids).intersection(baseline_top_20_ids))
            entering_leaving = 100 - overlap_100
            
            # Avg score difference and max rank movement for overlapping candidates
            score_diffs = []
            max_move = 0
            for cid in top_100_ids:
                if cid in baseline_data:
                    score_diff = abs(current_score_map[cid] - baseline_data[cid]["score"])
                    score_diffs.append(score_diff)
                    rank_move = abs(current_rank_map[cid] - baseline_data[cid]["rank"])
                    max_move = max(max_move, rank_move)
            
            avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0
            
            missing = []
            for cid in baseline_top_100_ids:
                if cid not in top_100_ids:
                    missing.append({
                        "id": cid,
                        "struct_rank": structured_rank_map[cid],
                        "emb_score": baseline_data[cid]["emb_score"],
                        "baseline_rank": baseline_data[cid]["rank"]
                    })
            
            results_by_pool[n] = {
                "runtime": t_stage1 + t_emb,
                "candidates_embedded": n,
                "peak_ram": max(mem_before, mem_peak),
                "top_100_overlap": overlap_100,
                "top_20_overlap": overlap_20,
                "avg_score_diff": avg_score_diff,
                "max_rank_move": max_move,
                "entering_leaving": entering_leaving,
                "missing": missing
            }
    
    out = []
    out.append("# Enhanced Embedding Pool Size Experiment Results")
    out.append("*(Baseline: N=10,000)*\n")
    
    out.append("## Comparison Table")
    out.append("| Pool | Runtime (min) | Peak RAM (MB) | Top 100 Overlap | Top 20 Overlap | Avg Score Diff | Max Rank Move | Dropped/New |")
    out.append("|---|---|---|---|---|---|---|---|")
    
    for n in sorted(pools, reverse=True):
        res = results_by_pool[n]
        out.append(f"| {n} | {res['runtime']/60:.2f}m | {res['peak_ram']:.1f} | {res['top_100_overlap']}/100 | {res['top_20_overlap']}/20 | {res['avg_score_diff']:.4f} | {res['max_rank_move']} | {res['entering_leaving']} |")
        
    out.append("\n## Missing Candidates Analysis")
    for n in [5000, 2500, 1000]:
        missing = results_by_pool[n]["missing"]
        out.append(f"\n### Disappeared in N={n}")
        if not missing:
            out.append("None. Perfect overlap.")
        else:
            for m in sorted(missing, key=lambda x: x["baseline_rank"]):
                out.append(f"- **{m['id']}**: Baseline Rank = {m['baseline_rank']}, Structured Rank = {m['struct_rank']}, Embedding Score = {m['emb_score']:.4f}")

    with open("c:/Users/Akanksha Shirke/.gemini/antigravity-ide/brain/bbed9454-3261-48a5-b6e7-3ffdf2f92f50/enhanced_experiment_results.md", "w") as f:
        f.write("\n".join(out))
    
    print("Done! Wrote enhanced_experiment_results.md")

if __name__ == '__main__':
    main()
