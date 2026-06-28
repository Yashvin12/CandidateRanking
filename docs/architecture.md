# Pipeline Architecture — CandidateRanking

## System Overview

Multi-stage structured ranker with offline LLM feature pre-computation (Architecture B).

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                  OFFLINE PRE-COMPUTATION                    │
                    │             (outside 5-minute ranking window)               │
                    │                                                             │
                    │  candidates.jsonl ──► src/llm_extractor.py                  │
                    │      (100K+)              │                                 │
                    │                           │  AWS Bedrock (Mistral)          │
                    │                           │  or Groq API (free tier)        │
                    │                           ▼                                 │
                    │                  data/llm_features.jsonl                    │
                    │                  (Top-1000 LLM signals)                     │
                    └──────────────────────────┬──────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────────┐
                    │                   5-MINUTE RANKING WINDOW                   │
                    │                                                             │
                    │   candidates.jsonl  +  data/llm_features.jsonl              │
                    │   (100K+ records)       (pre-computed)                      │
                    └──────────────────────────┬──────────────────────────────────┘
                                               │
                                               ▼
                    ╔═════════════════════════════════════════════════════════════╗
                    ║              STAGE 1: STRUCTURED SCORING                    ║
                    ║                  (ALL 100K candidates)                      ║
                    ╠═════════════════════════════════════════════════════════════╣
                    ║                                                             ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │  skill_scorer.py          0–40 pts                  │    ║
                    ║  │  ─────────────────────────────────────────────────  │    ║
                    ║  │  • Must-have group 1: Retrieval (OpenSearch, FAISS, │    ║
                    ║  │    Elasticsearch, pgvector, Milvus, Qdrant, Weaviate)│   ║
                    ║  │  • Must-have group 2: ML Frameworks (PyTorch, JAX,  │    ║
                    ║  │    TensorFlow, HuggingFace)                         │    ║
                    ║  │  • Must-have group 3: Python                        │    ║
                    ║  │  • Must-have group 4: NLP/IR (BM25, Semantic Search,│    ║ 
                    ║  │    RAG, Embeddings)                                 │    ║
                    ║  │  • Depth/duration/endorsement bonuses               │    ║
                    ║  └─────────────────────────────────────────────────────┘    ║
                    ║                           +                                 ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │  career_scorer.py         0–30 pts                  │    ║
                    ║  │  ─────────────────────────────────────────────────  │    ║
                    ║  │  • Company tier: product > research > consulting    │    ║
                    ║  │  • Title relevance: Staff/Principal > Senior > Mid  │    ║
                    ║  │  • Experience depth (yoe bands)                     │    ║
                    ║  │  • LLM signals injected here (offline pre-computed):│    ║
                    ║  │    production_retrieval, consulting_flag,           │    ║
                    ║  │    title_mismatch, applied_ml_years                 │    ║
                    ║  └─────────────────────────────────────────────────────┘    ║
                    ║                           +                                 ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │  alignment_scorer.py      0–15 pts                  │    ║
                    ║  │  ─────────────────────────────────────────────────  │    ║
                    ║  │  • Location preference (Noida/Pune > Other India)   │    ║
                    ║  │  • Education tier + field relevance                 │    ║
                    ║  │  • Notice period (immediate > 15d > 30d > 60d)      │    ║
                    ║  └─────────────────────────────────────────────────────┘    ║
                    ║                           ×                                 ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │  behavioral.py            ×0.1–1.0                  │    ║
                    ║  │  Geometric mean of 5 engagement signals:            │    ║
                    ║  │  recency · response_rate · availability ·           │    ║
                    ║  │  interview_completion · offer_acceptance            │    ║
                    ║  └─────────────────────────────────────────────────────┘    ║
                    ║                           ×                                 ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │  contradiction.py         ×0.5–1.0                  │    ║
                    ║  │  Penalty for inconsistent profile signals           │    ║
                    ║  │  (expert claims with zero duration, etc.)           │    ║
                    ║  └─────────────────────────────────────────────────────┘    ║
                    ╚══════════════════════════════╦══════════════════════════════╝
                                                   │
                                    heapq.nlargest(1000)
                                    O(n log k) — no full sort
                                                   │
                                                   ▼
                    ╔═════════════════════════════════════════════════════════════╗
                    ║             STAGE 2: EMBEDDING SCORING                      ║
                    ║                  (Top-1000 only)                            ║
                    ╠═════════════════════════════════════════════════════════════╣
                    ║                                                             ║
                    ║  embedding_scorer.py                        0–15 pts        ║
                    ║  ─────────────────────────────────────────────────────      ║
                    ║  • Model: all-MiniLM-L6-v2 (384-dim, CPU-only)              ║
                    ║  • Encodes candidate profile text → vector                  ║
                    ║  • Encodes JD text → vector (cached)                        ║
                    ║  • Score = cosine_similarity × 15                           ║
                    ╚══════════════════════════════╦══════════════════════════════╝
                                                   │
                                    Final re-rank + sort
                                                   │
                                                   ▼
                    ╔═════════════════════════════════════════════════════════════╗
                    ║                       TOP-100 OUTPUT                        ║
                    ╠═════════════════════════════════════════════════════════════╣
                    ║  reasoning.py  →  generates reasoning column                ║
                    ║  submission.csv  [candidate_id, rank, score, reasoning]     ║
                    ╚═════════════════════════════════════════════════════════════╝
```

## Score Composition

| Component | Max | Applied To |
|---|---|---|
| `skill_scorer.py` | 40 pts | All 100K |
| `career_scorer.py` | 30 pts | All 100K |
| `alignment_scorer.py` | 15 pts | All 100K |
| `embedding_scorer.py` | 15 pts | Top-1000 only |
| `behavioral.py` | ×0.1–1.0 | All 100K |
| `contradiction.py` | ×0.5–1.0 | All 100K |
| **Max total** | **~100** | — |

## Honeypot Detection

`honeypot.py` flags candidates with any of the following before scoring:
- **Salary inversion**: Expected CTC > Offered CTC
- **Temporal paradox**: Experience > Age allows (e.g., 15 yoe, age 22)
- **Experience paradox**: Expert-level claims with zero or negligible duration

Honeypot candidates are forced to score = 0.0 (excluded from final ranking).

## Data Flow Diagram

```
candidates.jsonl (100K+)
        │
        ├──► honeypot.py ──────────────────────── flag_map (id → bool)
        │
        ├──► skill_scorer.py ──────────────────── score_s (0–40)
        │                                            │
        ├──► career_scorer.py ◄── llm_features ── score_c (0–30)
        │                                            │
        ├──► alignment_scorer.py ─────────────── score_a (0–15)
        │                                            │
        ├──► behavioral.py ────────────────────── mult_b  (×0.1–1.0)
        │                                            │
        └──► contradiction.py ─────────────────── mult_x  (×0.5–1.0)
                                                    │
              stage1_score = (s+c+a) × b × x        │
              ─────────────────────────────────────►─┘
                        │
              heapq.nlargest(1000)
                        │
              embedding_scorer.py ─────────────── score_e (0–15)
                        │
              final_score = (s+c+a+e) × b × x
                        │
              top 100 sorted by final_score
                        │
              reasoning.py + CSV writer
                        │
              submission.csv
```

## Compute Environment

| Constraint | Approach |
|---|---|
| No GPU | all-MiniLM-L6-v2 runs on CPU (sentence-transformers) |
| No network during ranking | LLM features pre-computed offline to JSONL |
| ≤5 min on 16 GB / 8-core CPU | heapq + orjson + embedding only on Top-1000 |
| 100K+ JSONL records | orjson (3–5× faster than stdlib), streaming |

## Runtime (Observed)

- Stage 1 (structured, 100K candidates): **~45–55 seconds**
- Top-1000 selection (heapq): **< 1 second**
- Stage 2 (embeddings, 1000 candidates): **~10–15 seconds**
- Total: **~60 seconds** (well within 5-minute budget)
