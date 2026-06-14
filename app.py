"""
app.py — Streamlit sandbox for the Redrob Candidate Ranker.
============================================================
Deploys on HuggingFace Spaces.  Handles small sample files (≤100 candidates).
Works without company_classifications.json — falls back to seed lists only.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

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

MAX_UPLOAD_MB = 10


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
    "Upload candidate file (.json or .jsonl — small samples only)",
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

                # Save to a temp file so loader.py can read it.
                suffix = "." + uploaded.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, mode="wb"
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Loading candidates..."):
                    candidates = load_candidates(tmp_path)

                if not candidates:
                    st.warning("No valid candidates found in the file.")
                else:
                    # No company map in sandbox — empty dict forces seed-list fallback.
                    company_map: dict[str, str] = {}

                    with st.spinner("Detecting honeypots..."):
                        honeypot_flags = flag_honeypots(candidates)
                        hp_count = sum(1 for v in honeypot_flags.values() if v[0])

                    with st.spinner("Computing embeddings..."):
                        embedding_scores = compute_embedding_scores(candidates)

                    with st.spinner("Scoring candidates..."):
                        results: list[dict] = []
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
                            emb_score = embedding_scores[i]

                            is_hp, _ = honeypot_flags.get(cid, (False, None))

                            if is_hp:
                                final_score = 0.0
                            else:
                                raw_fit = (
                                    skill_score
                                    + career_score
                                    + alignment_score
                                    + emb_score
                                )
                                final_score = raw_fit * contra_mult * behav_mult

                            subscores = {
                                "skill_score": skill_score,
                                "career_score": career_score,
                                "embedding_score": emb_score,
                                "behavioral": behav_mult,
                                "skill_breakdown": skill_bd,
                            }

                            results.append({
                                "candidate_id": cid,
                                "final_score": round(final_score, 4),
                                "reasoning": generate_reasoning(candidate, subscores),
                                "is_honeypot": is_hp,
                            })

                    # Sort and take top 100
                    results.sort(
                        key=lambda r: (-r["final_score"], r["candidate_id"])
                    )
                    top_n = results[: min(100, len(results))]

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

                    # Build CSV for download
                    csv_buffer = io.StringIO()
                    writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
                    for rank, r in enumerate(top_n, 1):
                        writer.writerow([
                            r["candidate_id"],
                            rank,
                            f"{r['final_score']:.4f}",
                            r["reasoning"],
                        ])

                    st.download_button(
                        label="Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name="submission.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                raise
