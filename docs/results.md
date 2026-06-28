# Results & Performance

## Final Submission Stats

| Metric | Value |
|---|---|
| Dataset size | 100,000+ candidates |
| Candidates ranked | 100 (top 100 selected) |
| Score range (top 100) | 68.03 – 92.16 |
| Mean score (top 100) | 74.96 |
| Honeypots detected | 0 in top 100 (all honeypots correctly excluded) |
| Total pipeline runtime | ~60 seconds |
| Compute constraint | ≤ 5 minutes — ✅ |

## Runtime Breakdown

| Stage | Time |
|---|---|
| Load + orjson parse (100K candidates) | ~2–5 s |
| Honeypot detection (100K) | ~2 s |
| Stage 1 structured scoring (100K) | ~45–55 s |
| heapq Top-1000 selection | < 1 s |
| Stage 2 embedding scoring (1000 candidates) | ~10–15 s |
| CSV write + reasoning | < 1 s |
| **Total** | **~60 s** |

## Top 10 Candidates

| Rank | Candidate ID | Score | Title @ Company |
|---|---|---|---|
| 1 | CAND_0052328 | 92.16 | Recommendation Systems Engineer @ Amazon |
| 2 | CAND_0077337 | 91.34 | Staff ML Engineer @ Paytm |
| 3 | CAND_0052682 | 89.28 | NLP Engineer @ Aganitha |
| 4 | CAND_0027691 | 87.61 | NLP Engineer @ Haptik |
| 5 | CAND_0008295 | 87.06 | AI Research Engineer @ Razorpay |
| 6 | CAND_0041669 | 86.31 | Recommendation Systems Engineer @ CRED |
| 7 | CAND_0018499 | 84.15 | Senior ML Engineer @ Zomato |
| 8 | CAND_0064326 | 84.04 | Search Engineer @ Sarvam AI |
| 9 | CAND_0080766 | 83.98 | Staff ML Engineer @ Salesforce |
| 10 | CAND_0020708 | 83.57 | Search Engineer @ PolicyBazaar |

## Score Distribution Notes

- **Score 85+**: 3 candidates — elite fit (top product companies, retrieval stack, Pune/Noida location)
- **Score 80–85**: 8 candidates — strong fit (good retrieval skills + FAISS/pgvector/OpenSearch)
- **Score 70–80**: 54 candidates — solid fit (product company ML background, retrieval skills)
- **Score 68–70**: 35 candidates — moderate fit (some retrieval skill, some engagement flags)

## Honeypot Handling

The system detected honeypots across the full 100K pool using three signals:
- **Salary inversion**: Expected CTC > Offered CTC
- **Temporal paradox**: Total experience > plausible working age
- **Experience paradox**: Expert-level claims contradicted by duration

All honeypots scored exactly 0.0 and were excluded from the top-100 output.

## Key Signal Sources for Top Candidates

Top candidates consistently showed:
- Production retrieval systems (FAISS, OpenSearch, pgvector, Milvus) at product-tier companies
- 4–9 years of experience in ML/search roles
- Based in Noida, Pune, or major Indian tech hubs
- Active on platform: response rates 65–95%, recent activity
