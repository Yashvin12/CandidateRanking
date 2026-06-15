import time
import psutil
import os
import json
import statistics

from src.loader import load_candidates, load_company_classifications
from src.honeypot import flag_honeypots
from src.embedding_scorer import compute_embedding_scores
from src.skill_scorer import compute_skill_score
from src.career_scorer import compute_career_score
from src.alignment_scorer import compute_alignment_score
from src.contradiction import compute_contradiction_penalty
from src.behavioral import compute_behavioral_multiplier
from src.reasoning import generate_reasoning

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("Starting validation run...")
    t_start = time.time()
    mem_start = get_memory_mb()

    candidates = load_candidates("candidates.jsonl")
    company_map = load_company_classifications("data/company_classifications.json")

    mem_loaded = get_memory_mb()

    # Honeypot
    t_hp = time.time()
    honeypot_flags = flag_honeypots(candidates)
    t_hp_elapsed = time.time() - t_hp

    # Embeddings
    t_emb = time.time()
    embedding_scores = compute_embedding_scores(candidates)
    t_emb_elapsed = time.time() - t_emb
    mem_after_emb = get_memory_mb()

    # Scoring
    t_score = time.time()
    results = []
    
    contradiction_counts = {}
    behavioral_scores = []
    raw_scores = []
    final_scores = []
    
    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id", f"UNKNOWN_{i}")

        skill_score, skill_bd = compute_skill_score(candidate)
        career_score, career_bd = compute_career_score(candidate, company_map)
        alignment_score, align_bd = compute_alignment_score(candidate, skill_bd)
        
        contra_mult, contra_reasons = compute_contradiction_penalty(candidate, career_bd)
        signals = candidate.get("redrob_signals") or {}
        behav_mult, behav_bd = compute_behavioral_multiplier(signals)
        emb_score = embedding_scores[i]

        for r in contra_reasons:
            import re
            # generalize numbers so we group them
            r_general = re.sub(r'\d+', 'X', r)
            contradiction_counts[r_general] = contradiction_counts.get(r_general, 0) + 1

        is_honeypot, hp_reason = honeypot_flags.get(cid, (False, None))

        raw_fit = skill_score + career_score + alignment_score + emb_score
        raw_scores.append(raw_fit)
        
        if is_honeypot:
            final_score = 0.0
        else:
            final_score = raw_fit * contra_mult * behav_mult
            
        final_scores.append(final_score)
        behavioral_scores.append(behav_mult)

        results.append({
            "candidate_id": cid,
            "final_score": round(final_score, 4),
            "skill_score": skill_score,
            "career_score": career_score,
            "alignment_score": alignment_score,
            "embedding_score": emb_score,
            "contra_mult": contra_mult,
            "behav_mult": behav_mult,
            "is_honeypot": is_honeypot,
        })
        
    t_score_elapsed = time.time() - t_score
    t_total = time.time() - t_start
    mem_final = get_memory_mb()

    # Sort
    results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    top_100 = results[:100]

    out = []
    out.append("# DATA-DRIVEN VALIDATION: COMPLETE RANKING PIPELINE")
    
    out.append("## 1. Runtime Measurements")
    out.append(f"- Data Loading: {t_hp - t_start:.2f}s")
    out.append(f"- Honeypot Detection: {t_hp_elapsed:.2f}s")
    out.append(f"- Embedding Computation: {t_emb_elapsed:.2f}s")
    out.append(f"- Scoring Loop: {t_score_elapsed:.2f}s")
    out.append(f"- **Total Runtime**: {t_total:.2f}s ({t_total/60:.2f} min)")

    out.append("\n## 2. Memory Measurements")
    out.append(f"- Baseline (Start): {mem_start:.2f} MB")
    out.append(f"- After Data Load: {mem_loaded:.2f} MB (Dataset footprint: {mem_loaded - mem_start:.2f} MB)")
    out.append(f"- After Embeddings: {mem_after_emb:.2f} MB")
    out.append(f"- Peak/Final Memory: {mem_final:.2f} MB")

    out.append("\n## 3. Honeypot Statistics")
    hp_count = sum(1 for r in results if r["is_honeypot"])
    out.append(f"- Total Flagged: {hp_count}")
    out.append(f"- Flagging Rate: {(hp_count/len(candidates))*100:.2f}%")

    out.append("\n## 4. Contradiction Statistics")
    out.append("Frequency of contradiction reasons triggered (numbers generalized to X):")
    for k, v in sorted(contradiction_counts.items(), key=lambda x: -x[1]):
        out.append(f"- {v} times: {k}")

    out.append("\n## 5. Behavioral Score Distribution")
    out.append(f"- Mean: {statistics.mean(behavioral_scores):.4f}")
    out.append(f"- Median: {statistics.median(behavioral_scores):.4f}")
    out.append(f"- Min: {min(behavioral_scores):.4f}")
    out.append(f"- Max: {max(behavioral_scores):.4f}")
    
    out.append("\n## 6. Raw Score Distribution")
    out.append(f"- Mean: {statistics.mean(raw_scores):.2f}")
    out.append(f"- Median: {statistics.median(raw_scores):.2f}")
    out.append(f"- Min: {min(raw_scores):.2f}")
    out.append(f"- Max: {max(raw_scores):.2f}")

    out.append("\n## 7. Final Score Distribution")
    out.append(f"- Mean: {statistics.mean(final_scores):.2f}")
    out.append(f"- Median: {statistics.median(final_scores):.2f}")
    out.append(f"- Min: {min(final_scores):.2f}")
    out.append(f"- Max: {max(final_scores):.2f}")
    
    out.append("\n## 8. Top 100 Ranked Candidates")
    out.append("| Rank | Candidate ID | Final Score | Skill | Career | Align | Embed | Contra Mult | Behav Mult |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for idx, r in enumerate(top_100, 1):
        out.append(f"| {idx} | {r['candidate_id']} | {r['final_score']:.4f} | {r['skill_score']} | {r['career_score']} | {r['alignment_score']} | {r['embedding_score']} | {r['contra_mult']} | {r['behav_mult']} |")

    with open("c:/Users/Akanksha Shirke/.gemini/antigravity-ide/brain/bbed9454-3261-48a5-b6e7-3ffdf2f92f50/full_pipeline_validation.md", "w") as f:
        f.write("\n".join(out))
    print("Done. Wrote full_pipeline_validation.md")

if __name__ == '__main__':
    main()
