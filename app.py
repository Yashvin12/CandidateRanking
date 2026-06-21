"""
app.py — Streamlit sandbox for the Redrob Candidate Ranker.
============================================================
Deploys on HuggingFace Spaces.  Handles large files (up to 1 GB).
Uses the same two-stage pipeline as rank.py:
  Stage 1 — cheap structured scoring on all candidates
  Stage 2 — expensive embedding scoring on top 1000 only

Works without company_classifications.json — falls back to seed lists only.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import heapq
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.loader import load_candidates
from src.honeypot import flag_honeypots
from src.embedding_scorer import compute_embedding_scores
from src.skill_scorer import compute_skill_score
from src.career_scorer import compute_career_score
from src.alignment_scorer import compute_alignment_score
from src.contradiction import compute_contradiction_penalty
from src.behavioral import compute_behavioral_multiplier
from src.reasoning import generate_reasoning


st.set_page_config(page_title="Redrob Ranker", layout="wide")

MAX_UPLOAD_MB = 1000

@st.cache_data(show_spinner=False)
def load_cached_candidates(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    return load_candidates(tmp_path)



# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("System Info")
    st.markdown("**Model:** all-MiniLM-L6-v2")
    st.markdown("**Scoring:** Structured + Semantic")
    st.markdown("---")
    st.subheader("Methodology")
    st.markdown(
        """
        - **Skill matching** — 4 must-have groups scored with depth bonuses
        - **Career quality** — company classification, title relevance, experience bands
        - **Semantic embedding** — cosine similarity to JD via sentence-transformers
        - **Behavioral signals** — recency, response rate, availability (geometric mean)
        - **Honeypot detection** — salary inversion, temporal paradox, experience paradox
        """
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main UI
# ═══════════════════════════════════════════════════════════════════════════
st.title("Redrob Candidate Ranker")
st.markdown("**Intelligent Candidate Discovery & Ranking System**")

uploaded = st.file_uploader(
    "Upload candidate file (.json or .jsonl)",
    type=["json", "jsonl"],
)

if uploaded is not None:
    # File size check
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Maximum is {MAX_UPLOAD_MB} MB.")
    else:
        run_btn = st.button("Run Ranking", type="primary")

        if run_btn:
            try:
                t0 = time.perf_counter()

                with st.spinner("Loading and processing candidates..."):
                    candidates = load_cached_candidates(uploaded)

                if not candidates:
                    st.warning("No valid candidates found in the file.")
                else:
                    # No company map in sandbox — empty dict forces seed-list fallback.
                    company_map: dict[str, str] = {}

                    # ───────────────────────────────────────────────────
                    # Honeypot detection
                    # ───────────────────────────────────────────────────
                    with st.spinner("Detecting honeypots..."):
                        honeypot_flags = flag_honeypots(candidates)
                        hp_count = sum(1 for v in honeypot_flags.values() if v[0])

                    # ───────────────────────────────────────────────────
                    # Stage 1: Cheap structured scoring on ALL candidates
                    # ───────────────────────────────────────────────────
                    progress_bar = st.progress(0, text="Stage 1: Structured scoring...")
                    stage1_results: list[dict] = []
                    n_candidates = len(candidates)

                    for i, candidate in enumerate(candidates):
                        cid = candidate.get("candidate_id", f"UNKNOWN_{i}")

                        skill_score, skill_bd = compute_skill_score(candidate)
                        career_score, career_bd = compute_career_score(
                            candidate, company_map
                        )
                        alignment_score, align_bd = compute_alignment_score(
                            candidate, skill_bd
                        )
                        contra_mult, contra_reasons = compute_contradiction_penalty(
                            candidate, career_bd
                        )
                        signals = candidate.get("redrob_signals") or {}
                        behav_mult, behav_bd = compute_behavioral_multiplier(signals)

                        is_hp, _ = honeypot_flags.get(cid, (False, None))

                        if is_hp:
                            stage1_score = 0.0
                        else:
                            raw_fit = skill_score + career_score + alignment_score
                            stage1_score = raw_fit * contra_mult * behav_mult

                        stage1_results.append({
                            "candidate_id": cid,
                            "stage1_score": stage1_score,
                            "skill_score": skill_score,
                            "career_score": career_score,
                            "alignment_score": alignment_score,
                            "contra_mult": contra_mult,
                            "behav_mult": behav_mult,
                            "is_honeypot": is_hp,
                            "skill_breakdown": skill_bd,
                            "career_breakdown": career_bd,
                        })

                        # Update progress bar every 1000 candidates
                        if (i + 1) % 1000 == 0 or i == n_candidates - 1:
                            progress_bar.progress(
                                (i + 1) / n_candidates,
                                text=f"Stage 1: Scored {i + 1}/{n_candidates} candidates",
                            )

                    progress_bar.progress(1.0, text="Stage 1 complete!")

                    # ───────────────────────────────────────────────────
                    # Select top 1000 for embedding (same as rank.py)
                    # ───────────────────────────────────────────────────
                    top_pool_size = min(1000, len(stage1_results))
                    top_pool = heapq.nlargest(
                        top_pool_size,
                        stage1_results,
                        key=lambda r: (r["stage1_score"], r["candidate_id"]),
                    )
                    top_pool.sort(key=lambda r: (-r["stage1_score"], r["candidate_id"]))

                    # Build id → candidate lookup for top pool only
                    top_ids = {r["candidate_id"] for r in top_pool}
                    candidate_index = {
                        c["candidate_id"]: c for c in candidates
                        if c.get("candidate_id") in top_ids
                    }

                    # ───────────────────────────────────────────────────
                    # Stage 2: Embedding scoring on top 1000 ONLY
                    # ───────────────────────────────────────────────────
                    with st.spinner(f"Stage 2: Computing embeddings for top {len(top_pool)} candidates..."):
                        top_pool_candidates = [
                            candidate_index.get(r["candidate_id"], {})
                            for r in top_pool
                        ]
                        embedding_scores = compute_embedding_scores(top_pool_candidates)

                    # ───────────────────────────────────────────────────
                    # Final scoring & re-rank
                    # ───────────────────────────────────────────────────
                    results: list[dict] = []
                    for i, r in enumerate(top_pool):
                        emb_score = embedding_scores[i]

                        if r["is_honeypot"]:
                            final_score = 0.0
                        else:
                            raw_fit = (
                                r["skill_score"]
                                + r["career_score"]
                                + r["alignment_score"]
                                + emb_score
                            )
                            final_score = raw_fit * r["contra_mult"] * r["behav_mult"]

                        subscores = {
                            "skill_score": r["skill_score"],
                            "career_score": r["career_score"],
                            "embedding_score": emb_score,
                            "behavioral": r["behav_mult"],
                            "skill_breakdown": r["skill_breakdown"],
                        }

                        candidate_data = candidate_index.get(r["candidate_id"], {})
                        results.append({
                            "candidate_id": r["candidate_id"],
                            "final_score": round(final_score, 4),
                            "candidate_data": candidate_data,
                            "subscores": subscores,
                            "career_breakdown": r.get("career_breakdown"),
                            "is_honeypot": r["is_honeypot"],
                        })

                    # Sort and take top 100
                    results.sort(
                        key=lambda r: (-r["final_score"], r["candidate_id"])
                    )
                    top_n = results[: min(100, len(results))]

                    # Generate reasoning with rank known
                    for rank, r in enumerate(top_n, start=1):
                        r["reasoning"] = generate_reasoning(
                            r["candidate_data"],
                            r["subscores"],
                            rank=rank,
                            career_breakdown=r.get("career_breakdown"),
                        )

                    elapsed = time.perf_counter() - t0

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Candidates Ranked", len(candidates))
                    col2.metric("Honeypots Detected", hp_count)
                    col3.metric("Time", f"{elapsed:.1f}s")

                    # Build display table
                    table_data = []
                    for rank, r in enumerate(top_n, 1):
                        table_data.append({
                            "Rank": rank,
                            "Candidate ID": r["candidate_id"],
                            "Score": r["final_score"],
                            "Reasoning": r["reasoning"],
                        })

                    st.dataframe(table_data, use_container_width=True)

                    # Build CSV for download using Pandas
                    df_data = []
                    for rank, r in enumerate(top_n, 1):
                        df_data.append({
                            "candidate_id": r["candidate_id"],
                            "rank": rank,
                            "score": f"{r['final_score']:.4f}",
                            "reasoning": r["reasoning"]
                        })
                    
                    df = pd.DataFrame(df_data)
                    csv_string = df.to_csv(index=False)

                    st.download_button(
                        label="Download CSV",
                        data=csv_string,
                        file_name="submission.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                raise
