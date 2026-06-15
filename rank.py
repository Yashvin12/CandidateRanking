"""
rank.py — Main ranking pipeline entry point.
=============================================
Wires all scoring modules together, processes the full candidate dataset,
and outputs a ranked CSV of the top 100 candidates.

CLI
---
    python rank.py --candidates ./candidates.jsonl.gz --output ./submission.csv
    python rank.py --candidates ./data/sample_candidates.json --output ./submission.csv --sample
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for src imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.loader import load_candidates, load_company_classifications
from src.honeypot import flag_honeypots
from src.embedding_scorer import compute_embedding_scores
from src.skill_scorer import compute_skill_score
from src.career_scorer import compute_career_score
from src.alignment_scorer import compute_alignment_score
from src.contradiction import compute_contradiction_penalty
from src.behavioral import compute_behavioral_multiplier
from src.reasoning import generate_reasoning

# Try importing tqdm for a progress bar; fall back to a no-op wrapper.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidates against the Redrob Senior ML/AI Engineer JD.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidate data file (.json, .jsonl, .jsonl.gz)",
    )
    parser.add_argument(
        "--output",
        default="./submission.csv",
        help="Output CSV path (default: ./submission.csv)",
    )
    parser.add_argument(
        "--company-map",
        default="./data/company_classifications.json",
        help="Path to company classifications JSON (default: ./data/company_classifications.json)",
    )
    parser.add_argument(
        "--llm-features",
        default="./data/llm_features.jsonl",
        help=(
            "Path to offline LLM features JSONL produced by src/llm_extractor.py "
            "(default: ./data/llm_features.jsonl). "
            "If the file is missing the pipeline falls back to heuristic-only scoring."
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process only the first 100 candidates for quick testing.",
    )
    parser.add_argument(
        "--top-ids-output",
        default="./data/top_1000_ids.txt",
        help=(
            "Path to write the top-1000 candidate IDs after Stage 1 scoring. "
            "This file is then consumed by src/llm_extractor.py --top-ids. "
            "(default: ./data/top_1000_ids.txt)"
        ),
    )
    return parser.parse_args(argv)


def load_llm_features(path: str) -> dict[str, dict]:
    """Load offline LLM features from a JSONL checkpoint file.

    Returns a ``candidate_id -> feature dict`` mapping for O(1) lookup.
    Returns an empty dict (with a printed warning) if the file does not exist,
    so the pipeline degrades gracefully to heuristic-only scoring.
    """
    file_path = Path(path)
    if not file_path.exists():
        print(
            f"[INFO] LLM features file not found: '{path}'. "
            "Running in heuristic-only mode (no LLM adjustments)."
        )
        return {}

    features: dict[str, dict] = {}
    skipped = 0
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cid = record.get("candidate_id")
                if cid:
                    features[cid] = record
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    print(
        f"Loaded {len(features)} LLM feature records from {path}"
        + (f" ({skipped} skipped/malformed)" if skipped else "")
    )
    return features


def run_pipeline(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Load data
    # ═══════════════════════════════════════════════════════════════════
    candidates = load_candidates(args.candidates)
    company_map = load_company_classifications(args.company_map)
    llm_features = load_llm_features(args.llm_features)

    if args.sample:
        candidates = candidates[:100]
        print(f"[SAMPLE MODE] Using first {len(candidates)} candidates")

    print(f"Loaded {len(candidates)} candidates, {len(company_map)} company classifications")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Flag honeypots
    # ═══════════════════════════════════════════════════════════════════
    honeypot_flags = flag_honeypots(candidates)
    hp_count = sum(1 for v in honeypot_flags.values() if v[0])
    print(f"Flagged {hp_count} honeypots")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Stage 1 Structured Scoring
    # ═══════════════════════════════════════════════════════════════════
    t_stage1_start = time.perf_counter()
    stage1_results: list[dict] = []

    for i, candidate in enumerate(tqdm(candidates, desc="Stage 1: Structured Scoring")):
        cid = candidate.get("candidate_id", f"UNKNOWN_{i}")

        # a. Skill score
        skill_score, skill_bd = compute_skill_score(candidate)

        # b. Career score (pass LLM features for adjusted scoring)
        career_score, career_bd = compute_career_score(candidate, company_map, llm_features)

        # c. Alignment score
        alignment_score, align_bd = compute_alignment_score(candidate, skill_bd)

        # d. Contradiction penalty
        contra_mult, contra_reasons = compute_contradiction_penalty(candidate, career_bd)

        # e. Behavioral multiplier
        signals = candidate.get("redrob_signals") or {}
        behav_mult, behav_bd = compute_behavioral_multiplier(signals)

        # Stage 1 Score
        is_honeypot, hp_reason = honeypot_flags.get(cid, (False, None))
        
        if is_honeypot:
            stage1_score = 0.0
        else:
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
            # Do NOT store the full candidate dict for all 100K records — that
            # would use ~700 MB of RAM (fix 8B).  Store only the id; the full
            # dict is fetched from candidate_index for the top-1000 pool only.
            "skill_breakdown": skill_bd,
            "career_breakdown": career_bd,
            "align_breakdown": align_bd,
            "behav_breakdown": behav_bd,
            "contra_reasons": contra_reasons,
        })

    t_stage1_elapsed = time.perf_counter() - t_stage1_start
    print(f"Stage 1 completed in {t_stage1_elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Get Top 1000 using heapq.nlargest (O(n log k) vs O(n log n) sort)
    # ═══════════════════════════════════════════════════════════════════
    # Max embedding swing is 15 points — not enough for rank 1000 to
    # jump past rank 100 by structured score.  Keeping 1000 instead of
    # 5000 cuts embedding time by 5×.
    # heapq.nlargest is significantly faster than sort + slice for large n (fix 8A).
    top_pool = heapq.nlargest(
        1000,
        stage1_results,
        key=lambda r: (r["stage1_score"], r["candidate_id"]),  # stable tie-break
    )
    # Restore ascending id tie-break (heapq returns highest key first).
    top_pool.sort(key=lambda r: (-r["stage1_score"], r["candidate_id"]))

    # ── Write top-1000 IDs for the offline LLM extractor ─────────────────
    top_ids_path = Path(args.top_ids_output)
    top_ids_path.parent.mkdir(parents=True, exist_ok=True)
    with top_ids_path.open("w", encoding="utf-8") as f:
        for r in top_pool:
            f.write(r["candidate_id"] + "\n")
    print(f"Saved {len(top_pool)} top candidate IDs → {top_ids_path}")
    print(f"  → Run LLM extractor next:  python -m src.llm_extractor --candidates <your_file> --top-ids {top_ids_path}")

    # Build a fast id → candidate dict lookup from the original list (fix 8B).
    # Only materialise full dicts for the top 1000, not all 100K.
    candidate_index: dict[str, dict] = {
        c["candidate_id"]: c for c in candidates
        if c.get("candidate_id") in {r["candidate_id"] for r in top_pool}
    }
    for r in top_pool:
        r["candidate"] = candidate_index.get(r["candidate_id"], {})

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: Stage 2 Embedding Scoring on Top 1000
    # ═══════════════════════════════════════════════════════════════════
    t_emb_start = time.perf_counter()
    top_pool_candidates = [r["candidate"] for r in top_pool]
    print(f"Computing embeddings for top {len(top_pool_candidates)} candidates...")
    
    embedding_scores = compute_embedding_scores(top_pool_candidates)
    
    t_emb_elapsed = time.perf_counter() - t_emb_start
    print(f"Embedding stage completed in {t_emb_elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: Final Scoring & Re-rank
    # ═══════════════════════════════════════════════════════════════════
    results = []
    for i, r in enumerate(top_pool):
        emb_score = embedding_scores[i]
        r["embedding_score"] = emb_score
        
        if r["is_honeypot"]:
            final_score = 0.0
        else:
            raw_fit = r["skill_score"] + r["career_score"] + r["alignment_score"] + emb_score
            final_score = raw_fit * r["contra_mult"] * r["behav_mult"]
            
        r["final_score"] = round(final_score, 4)
        results.append(r)

    results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    top_100 = results[:100]

    # ═══════════════════════════════════════════════════════════════════
    # STEP 7: Verification
    # ═══════════════════════════════════════════════════════════════════
    hp_in_top = sum(1 for r in top_100 if r["is_honeypot"])
    print(f"\nHoneypots in top 100: {hp_in_top}")

    if top_100:
        scores = [r["final_score"] for r in top_100]
        print(f"Score range: {min(scores):.4f} to {max(scores):.4f}")

        print("\nTop 10 candidates:")
        for j, r in enumerate(top_100[:10], 1):
            c = r["candidate"]
            title = (c.get("profile") or {}).get("current_title", "N/A")
            company = (c.get("profile") or {}).get("current_company", "N/A")
            print(
                f"  {j:>3}. {r['candidate_id']}  "
                f"score={r['final_score']:.4f}  "
                f"{title} @ {company}"
            )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 8: Generate reasoning for top 100
    # ═══════════════════════════════════════════════════════════════════
    for r in top_100:
        subscores = {
            "skill_score": r["skill_score"],
            "career_score": r["career_score"],
            "embedding_score": r["embedding_score"],
            "behavioral": r["behav_mult"],
            "skill_breakdown": r["skill_breakdown"],
        }
        r["reasoning"] = generate_reasoning(r["candidate"], subscores)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 9: Write CSV
    # ═══════════════════════════════════════════════════════════════════
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        seen_ids: set[str] = set()
        for rank, r in enumerate(top_100, start=1):
            cid = r["candidate_id"]
            assert cid not in seen_ids, f"Duplicate candidate_id: {cid}"
            seen_ids.add(cid)

            writer.writerow([
                cid,
                rank,
                f"{r['final_score']:.4f}",
                r["reasoning"],
            ])

    # ═══════════════════════════════════════════════════════════════════
    # STEP 10: Done
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t_start
    print(f"\nWritten {output_path} with {len(top_100)} candidates")
    print(f"Total time: {elapsed:.1f}s")

    # Final validation
    assert len(top_100) == min(100, len(candidates)), (
        f"Expected {min(100, len(candidates))} rows, got {len(top_100)}"
    )


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
