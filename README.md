# Candidate Ranking Pipeline — Redrob Hackathon

A multi-stage candidate ranking system for the Redrob AI & Data hackathon. The pipeline ranks the top 100 candidates from a 100K+ pool against a Machine Learning Engineer job description, within a strict 5-minute CPU compute budget.

---

## Architecture Overview

The system uses **Architecture B: Offline LLM Feature Extraction + Structured Ranker**.

```
candidates.jsonl
       │
       ▼
[Stage 1: Structured Scorer]          (runs on ALL candidates)
  ├── skill_scorer.py   — Skill match against must-have groups  (0–40 pts)
  ├── career_scorer.py  — Company quality + title relevance     (0–30 pts)
  └── alignment_scorer.py — Location, education, notice period  (0–15 pts)
       │
       ▼
[Top-1000 Selection]                  (heapq.nlargest, O(n))
       │
       ▼
[Stage 2: Embedding Scorer]           (runs on Top-1000 only)
  └── embedding_scorer.py — Semantic similarity via MiniLM-L6  (0–15 pts)
       │
       ▼
[Final Rerank]
  ├── behavioral.py     — Geometric-mean engagement multiplier  (0.1–1.0×)
  ├── contradiction.py  — Soft-honeypot penalty multiplier      (0.5–1.0×)
  └── LLM features      — Pre-computed offline signals from llm_extractor.py
       │
       ▼
submission.csv (Top 100)
```

**Maximum possible score**: 40 (skill) + 30 (career) + 15 (alignment) + 15 (embedding) = **100 points**, then multiplied by behavioral and contradiction multipliers.

---

## Scoring Components

| Module | Max Score | Description |
|---|---|---|
| `skill_scorer.py` | 40 | Must-have skill groups (retrieval, ML, Python, NLP); endorsement/duration bonuses |
| `career_scorer.py` | 30 | Company quality (product > research > consulting), title relevance, experience depth |
| `alignment_scorer.py` | 15 | Location (Noida/Pune preferred), education tier+field, notice period |
| `embedding_scorer.py` | 15 | Cosine similarity between candidate text and JD (all-MiniLM-L6-v2) |
| `behavioral.py` | ×0.1–1.0 | Geometric mean of 5 engagement signals (activity, response rate, availability…) |
| `contradiction.py` | ×0.5–1.0 | Penalty for inconsistent profile signals (expert claims with zero duration, etc.) |

---

## Setup

### Prerequisites
- Python 3.10+
- 16 GB RAM (CPU-only)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Pre-computed data files (included in repo)

| File | Description |
|---|---|
| `data/llm_features.jsonl` | Offline LLM-extracted features for the Top-1000 candidates |
| `data/company_classifications.json` | Company → tier classification map |
| `data/top_1000_ids.txt` | Candidate IDs selected in Stage 1 for LLM processing |

---

## Reproducing the Submission

### Step 1 — Rank (produces submission.csv within 5 minutes)

```bash
python rank.py \
  --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \
  --output ./submission.csv
```

This is the **single command** that reproduces the submitted CSV. It reads `data/llm_features.jsonl` automatically if present. No network access or GPU required.

### Optional: Run on the small sample set

```bash
python rank.py \
  --candidates ./India_runs_data_and_ai_challenge/sample_candidates.json \
  --output ./submission.csv
```

### Optional: Re-generate LLM features (offline, requires API key)

This step is **not part of the 5-minute ranking window**. It is only needed if you want to re-extract features from scratch.

```bash
# Using AWS Bedrock (default)
python -m src.llm_extractor \
  --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \
  --top-ids data/top_1000_ids.txt \
  --resume

# Using Groq (free tier)
set GROQ_API_KEY=gsk_...
python -m src.llm_extractor \
  --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \
  --top-ids data/top_1000_ids.txt \
  --provider groq \
  --resume
```

---

## Project Structure

```
CandidateRanking/
├── rank.py                     # Main ranking entry point
├── app.py                      # Streamlit demo / sandbox
├── requirements.txt
├── submission_metadata.yaml
│
├── src/
│   ├── config.py               # All weights, keywords, thresholds
│   ├── loader.py               # Data loading (.json / .jsonl / .jsonl.gz)
│   ├── skill_scorer.py
│   ├── career_scorer.py
│   ├── alignment_scorer.py
│   ├── embedding_scorer.py
│   ├── behavioral.py
│   ├── contradiction.py
│   ├── honeypot.py
│   ├── reasoning.py            # Generates the reasoning column
│   └── llm_extractor.py        # Offline LLM feature pre-computation
│
├── data/
│   ├── llm_features.jsonl      # Pre-computed LLM features (Top-1000)
│   ├── company_classifications.json
│   └── top_1000_ids.txt
│
└── India_runs_data_and_ai_challenge/
    ├── candidates.jsonl         # Full 100K dataset (not in git)
    ├── sample_candidates.json
    ├── job_description.docx
    └── validate_submission.py
```

---

## Validating the Submission

```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
# Expected output: Submission is valid.
```

---

## Sandbox / Demo

A Streamlit app (`app.py`) is included for end-to-end demo on small candidate samples.

```bash
streamlit run app.py
```

🌐 **Live hosted app:** [candidateranking.streamlit.app](https://candidateranking.streamlit.app/)

---

## 🎬 Demo Video

A full end-to-end walkthrough of the ranking pipeline — including dataset upload, candidate ranking, and CSV export — is available here:

**[▶ Watch Demo on Google Drive](https://drive.google.com/file/d/1qsHsBmETFGYGECUbzFCI40EZZmmNJB0Q/view?usp=drive_link)**

The demo covers:
- Uploading the candidate dataset via the Streamlit UI
- Running the multi-stage ranking pipeline
- Viewing ranked results and exporting `submission.csv`
- Live walkthrough of the scoring logic

---

## Compute Constraints Compliance

| Constraint | Status |
|---|---|
| No hosted LLM API calls during ranking | ✅ — LLM features are pre-computed offline |
| No GPU use | ✅ — CPU-only (MiniLM runs on CPU) |
| ≤5 min on 16 GB CPU machine | ✅ — Tested locally |
| No network during ranking | ✅ |
