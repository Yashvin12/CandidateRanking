# Module Prompts — Copy-Paste Ready

> **How to use:** Feed each prompt to your coding assistant (or yourself) ONE AT A TIME, in order.
> Each prompt is self-contained — it includes all the context needed to build that module without referencing other documents.
> After each module, TEST it before moving to the next.

---

## Module 1: `config.py`

```
CONTEXT:
You are building a candidate ranking system for a hackathon. The system ranks 100,000 job candidates against a specific job description (JD) for a Senior ML/AI Engineer role at a product company called Redrob. The JD values production ML experience at product companies, NLP/IR/ranking expertise, Python, and ranking evaluation skills. It explicitly penalizes pure-consulting careers, non-technical roles, and keyword-stuffing.

TASK:
Create `src/config.py` — a single source of truth for every constant, threshold, keyword list, and weight used across the entire ranking pipeline. No logic, no functions — just organized data.

REQUIREMENTS:

1. COMPANY SEED LISTS (3 dicts):
   - CONSULTING_FIRMS: set of ~25 known IT services/consulting companies.
     Include: TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant, Cognizant Technology Solutions, Capgemini, HCL, HCL Technologies, Tech Mahindra, Mindtree, LTIMindtree, Mphasis, L&T Infotech, LTI, Persistent Systems, Hexaware, NIIT Technologies, Cyient, Zensar, Virtusa, UST Global, UST.
   - PRODUCT_COMPANIES: set of ~50 known product companies.
     Big tech: Google, Amazon, Meta, Facebook, Microsoft, Apple, Netflix, Twitter, LinkedIn, Salesforce, Adobe, Atlassian, Spotify, Shopify, Stripe, Databricks.
     Indian product: Flipkart, Swiggy, Zomato, Razorpay, PhonePe, Paytm, Ola, Uber, Meesho, CRED, Zerodha, Groww, Freshworks, Zoho, Postman, BrowserStack, ShareChat, Dream11, Nykaa, UrbanCompany, BigBasket.
     Global product: Airbnb, Lyft, DoorDash, Snap, Pinterest, Reddit, Discord, Palantir, Snowflake, Confluent, MongoDB Inc, Cloudflare, Twilio.
     Fintech hybrids: Monzo, Revolut, Klarna, Robinhood.
   - FICTIONAL_COMPANIES: set of ~10 synthetic/fictional company names that appear in the dataset.
     Include: Dunder Mifflin, Acme Corp, Globex Inc, Initech, Pied Piper, Stark Industries, Wayne Enterprises, Umbrella Corp, Hooli, Prestige Worldwide.

2. INDUSTRY CLASSIFICATION HEURISTICS (dict):
   - CONSULTING_INDUSTRIES: set of industry strings that strongly suggest consulting/services. Include: "IT Services", "IT Services and IT Consulting", "Staffing and Recruiting", "Management Consulting", "Business Consulting".
   - PRODUCT_INDUSTRIES: set of industry strings suggesting product companies. Include: "Software", "Internet", "Technology", "Computer Software", "Financial Technology", "E-commerce", "SaaS", "Mobile Apps".

3. TITLE CLASSIFICATION (3 sets):
   - ML_AI_TITLES: titles directly related to ML/AI work. Include variations with "Senior", "Staff", "Lead", "Principal" prefixes. Core titles: ML Engineer, Machine Learning Engineer, Data Scientist, Research Scientist, Applied Scientist, NLP Engineer, Search Engineer, Ranking Engineer, Deep Learning Engineer, AI Engineer, AI/ML Engineer, Recommendation Systems Engineer.
   - ADJACENT_TECH_TITLES: technical but not ML-specific. Include: Software Engineer, Backend Engineer, Data Engineer, Full Stack Engineer, DevOps Engineer, Platform Engineer, Infrastructure Engineer, SRE, Analytics Engineer, and their "Senior" variants.
   - NON_TECH_TITLES: clearly non-technical. Include: Marketing Manager, Operations Manager, HR Manager, Accountant, Sales Executive, Content Writer, Graphic Designer, Customer Support, Business Analyst, Project Manager, Civil Engineer, Mechanical Engineer, Brand Designer, Financial Analyst, Supply Chain Manager.

4. SKILL MATCHING KEYWORD GROUPS (dict of lists):
   MUST_HAVE_SKILL_GROUPS — 4 groups, each a list of case-insensitive keywords:
   - "production_retrieval": sentence-transformers, OpenAI embeddings, BGE, E5, semantic search, embedding, retrieval, ranking system, recommendation system, search system, information retrieval, dense retrieval, neural ranking, reranking, re-ranking, BM25, TF-IDF, hybrid search, RAG, vector search, approximate nearest neighbor.
   - "vector_db": Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch, vector database, vector store, Chroma, Vespa, Annoy, ScaNN, pgvector.
   - "python": Python.
   - "eval_frameworks": NDCG, MRR, MAP, A/B testing, evaluation framework, offline evaluation, ranking evaluation, precision@k, recall@k, click-through rate, CTR, online evaluation, interleaving.

   NICE_TO_HAVE_SKILLS — flat list:
   LoRA, QLoRA, PEFT, fine-tuning, fine tuning, LLM fine-tuning, XGBoost, LightGBM, learning-to-rank, learning to rank, LambdaMART, distributed systems, inference optimization, HR-tech, recruiting tech, marketplace.

   WRONG_DOMAIN_SKILLS — flat list (for ratio-based checking, NOT blind penalty):
   OpenCV, YOLO, object detection, image classification, image segmentation, computer vision, CNN, speech recognition, TTS, text-to-speech, ASR, robotics, ROS, SLAM, autonomous driving, GANs, GAN, generative adversarial.

   CORE_AI_SKILLS — broader ML/AI list used for counting skill breadth:
   Python, TensorFlow, PyTorch, scikit-learn, sklearn, NLP, deep learning, machine learning, neural network, transformer, BERT, GPT, LLM, large language model, MLOps, Kubeflow, MLflow, Weights & Biases, W&B, Hugging Face, huggingface, feature engineering, model training, model deployment, Spark, SparkML. Also include all items from MUST_HAVE_SKILL_GROUPS and NICE_TO_HAVE_SKILLS.

5. CAREER DESCRIPTION KEYWORDS (3 lists):
   - PRODUCTION_EVIDENCE_KEYWORDS: shipped, deployed, production, real users, live traffic, serving, served, scale, at scale, million users, A/B test, launched, released, system design, end-to-end, built and maintained, owned, SLA, latency, throughput, uptime.
   - CODE_WRITING_EVIDENCE: built, implemented, wrote, developed, coded, engineered, designed and built, architected and built, Python, code review, pull request, codebase, refactored, optimized.
   - NON_TECH_DESCRIPTION_KEYWORDS: marketing, brand, SEO, content writing, editorial, accounting, financial reporting, tax filing, customer support, support team, ticket, sales, quota, revenue, prospecting, CAD, SolidWorks, Creo, FEA, ANSYS, packaging design, brand identity, logo, supply chain, warehouse, fulfillment.

6. SCORING WEIGHTS (dict):
   WEIGHTS = {"skill_match": 40, "career_quality": 30, "embedding": 15, "alignment": 15}

7. BEHAVIORAL MULTIPLIER THRESHOLDS (nested dict):
   BEHAVIORAL = {
     "recency_tiers": [(30, 1.0), (90, 0.8), (180, 0.5), (9999, 0.2)],
     "response_rate_tiers": [(0.6, 1.0), (0.3, 0.7), (0.0, 0.35)],
     "interview_rate_tiers": [(0.7, 1.0), (0.4, 0.7), (0.0, 0.3)],
     "offer_rate_map": {-1: 0.7, 0: 0.15},
     "offer_rate_default_above_zero": 1.0,
     "offer_rate_low_threshold": 0.5,
     "offer_rate_low_value": 0.6,
   }

8. LOCATION SCORING (2 sets):
   PREFERRED_LOCATIONS = {"Noida", "Pune"}
   TIER1_INDIA_CITIES = {"Bangalore", "Bengaluru", "Hyderabad", "Mumbai", "Delhi", "New Delhi", "Gurgaon", "Gurugram", "Chennai", "Kolkata"}

9. REFERENCE DATE:
   REFERENCE_DATE = "2026-06-01" — used for calculating "days since last active."

10. JD_TEXT_FOR_EMBEDDING — a single string summarizing the core JD requirements, used as the anchor for semantic similarity scoring:
   "Senior ML/AI Engineer building production ranking, retrieval, and matching systems. Experience with embeddings-based retrieval, vector databases, hybrid search, semantic search deployed to real users at scale. Strong Python, ranking evaluation frameworks like NDCG MRR MAP. Product company background preferred. NLP and information retrieval focus. Built recommendation systems, search systems, candidate matching systems. Shipped at meaningful scale. Hands-on coding, not architecture-only."

STYLE:
- Use type hints for every variable.
- Group sections with comment block headers (═══ style).
- No functions, no imports, no logic — pure data declarations.
- Make it easy to tune: a developer should be able to change a weight or add a company name without reading any other file.
```

---

## Module 2: `loader.py`

```
CONTEXT:
You are building a ranking pipeline for a hackathon. The candidate data comes in two formats:
1. `sample_candidates.json` — a JSON array of 50 candidates (for testing).
2. `candidates.jsonl.gz` — 100,000 candidates, one JSON object per line, gzipped.

TASK:
Create `src/loader.py` with a single function that loads candidates from any of these formats.

REQUIREMENTS:

1. Function signature:
   def load_candidates(path: str) -> list[dict]

2. Auto-detect format by file extension:
   - `.json` → load as JSON array (json.load)
   - `.jsonl` → load line-by-line (json.loads per line), skip blank lines
   - `.jsonl.gz` → same as .jsonl but open with gzip.open in text mode ("rt", encoding="utf-8")

3. Try to use `orjson` for speed (10-20x faster than stdlib json). If orjson is not installed, fall back to stdlib json. Do NOT crash if orjson is missing.

4. Print a summary after loading:
   "Loaded {count} candidates from {path}"

5. Validate that every loaded candidate has a "candidate_id" key. If any don't, print a warning with the index and skip that entry.

6. Handle edge cases:
   - Empty file → return empty list, print warning.
   - File not found → raise FileNotFoundError with a clear message.
   - Malformed JSON line → print warning with line number, skip, continue.

7. Also create a helper function:
   def load_company_classifications(path: str) -> dict[str, str]
   This loads a JSON file mapping company names to their classification ("product", "consulting", "research", "unknown"). Return an empty dict if the file doesn't exist (with a printed warning, not an exception).

STYLE:
- Use pathlib.Path for path handling.
- Type hints on all functions.
- Docstrings with usage examples.
- No dependencies beyond stdlib + optional orjson.
```

---

## Module 3: `honeypot.py`

```
CONTEXT:
You are building a ranking pipeline for a hackathon dataset of 100,000 synthetic candidate profiles. Approximately 80 of these are "honeypot" candidates — profiles with subtly impossible data designed to trap naive ranking systems. If more than 10 of the 80 honeypots appear in your top-100 output, your submission is disqualified.

Honeypots are NOT "bad candidates" — they are IMPOSSIBLE candidates. The difference matters: a Marketing Manager with no ML skills is a bad candidate (they should score low naturally). A person whose salary minimum is higher than their salary maximum is impossible (the data itself is self-contradictory).

TASK:
Create `src/honeypot.py` with a function that flags provably impossible profiles.

REQUIREMENTS:

1. Function signature:
   def check_honeypot(candidate: dict) -> tuple[bool, str | None]
   Returns (True, "reason string") if honeypot, (False, None) if clean.

2. Implement exactly these checks — each one tests for data that is LOGICALLY IMPOSSIBLE, not merely suspicious:

   CHECK 1: SALARY RANGE INVERSION
   - Access: candidate["redrob_signals"]["expected_salary_range_inr_lpa"]
   - If min > max → honeypot.
   - Reason: "Salary range inverted: min {min} > max {max}"
   - Edge case: handle missing keys gracefully (treat as non-honeypot if data is absent).

   CHECK 2: TEMPORAL PARADOX
   - Access: candidate["redrob_signals"]["signup_date"] and candidate["redrob_signals"]["last_active_date"]
   - Parse both as dates. If last_active_date is BEFORE signup_date → honeypot.
   - Reason: "Timeline paradox: last active {last_active} before signup {signup}"
   - Edge case: if either date is null/missing/unparseable, skip this check.

   CHECK 3: EXPERIENCE DURATION PARADOX
   - Sum all career_history[*].duration_months.
   - Get claimed experience: candidate["profile"]["years_of_experience"] * 12 (convert to months).
   - If total_career_months > (claimed_months + 24) (2-year buffer for overlaps/rounding) → honeypot.
   - Reason: "Career duration paradox: {total_career_months} months of roles but only {claimed_years} years claimed"
   - Why 2-year buffer: candidates can have overlapping roles (moonlighting, part-time), so small overruns are normal. Only flag LARGE discrepancies.

3. A candidate is flagged as honeypot if ANY single check triggers. Return on the FIRST triggered check.

4. DO NOT check for:
   - Title-description mismatches (these are handled by the career scorer as a penalty — not provably impossible, just suspicious).
   - Skill duration anomalies (a person CAN claim "expert" with 0 months — it's dishonest but not impossible).
   - These softer signals are handled by other modules.

5. Also create a batch function:
   def flag_honeypots(candidates: list[dict]) -> dict[str, tuple[bool, str | None]]
   Returns a dict mapping candidate_id → (is_honeypot, reason).
   Print summary: "Flagged {count} honeypots out of {total} candidates"

TESTING:
- CAND_0000009 in the sample data has salary min=16.0, max=7.3 → MUST be flagged.
- CAND_0000006 has signup_date=2026-04-26, last_active_date=2026-02-28 → MUST be flagged.
- CAND_0000001 is a normal candidate → MUST NOT be flagged.
```

---

## Module 4: `company_classifier.py`

```
CONTEXT:
You are ranking 100,000 candidates for a job that explicitly penalizes consulting-only careers and rewards product-company backgrounds. The dataset contains thousands of unique company names. Hard-coding 80 known companies leaves 90%+ of companies unclassified. You need a scalable, data-driven solution.

Every career_history entry in the dataset has three fields you can use:
  - company (string — the company name)
  - industry (string — e.g., "IT Services", "Software", "Manufacturing")
  - company_size (string — e.g., "10001+", "51-200", "11-50")

TASK:
Create `src/company_classifier.py` — an OFFLINE pre-computation script that analyzes all 100K candidates, extracts every unique company, and classifies each as "product", "consulting", "research", "non_tech", or "unknown". Outputs a JSON file.

REQUIREMENTS:

1. This script is run ONCE before the ranking step. It is NOT part of the 5-minute ranking constraint. It can take as long as it needs.

2. Classification Logic (applied in priority order — first match wins):

   TIER 1: SEED LIST MATCH
   - Import CONSULTING_FIRMS, PRODUCT_COMPANIES, FICTIONAL_COMPANIES from config.
   - If the company name matches (case-insensitive, stripped of whitespace) any seed list → use that classification.
   - Fictional companies → classify as "unknown".

   TIER 2: INDUSTRY-BASED HEURISTIC
   - Collect ALL career_history entries across ALL candidates for each unique company.
   - Find the most common "industry" value for that company.
   - If industry is in CONSULTING_INDUSTRIES from config → "consulting".
   - If industry is in PRODUCT_INDUSTRIES from config → "product".
   - If industry contains "Research", "Academic", "University", "Education" → "research".
   - If industry contains "Manufacturing", "Construction", "Mining", "Paper Products", "Retail", "Real Estate", "Hospitality" → "non_tech".

   TIER 3: EMPLOYEE SIGNAL AGGREGATION
   For companies still unclassified after Tier 1 and 2:
   - Collect all employees (candidates who list this company in career_history).
   - Analyze their titles:
     * Count titles matching patterns: "Consultant", "Implementation", "Delivery Manager", "Business Analyst" → consulting_signal += 1
     * Count titles matching patterns: "Product Manager", "ML Engineer", "Data Scientist", "Software Engineer", "SRE", "Frontend", "Backend" → product_signal += 1
   - Analyze their career descriptions (concatenate all descriptions at this company):
     * Count occurrences of: "client", "deliverable", "engagement", "SOW", "consulting" → consulting_signal += 1 per hit
     * Count occurrences of: "our product", "our users", "shipped", "deployed", "production", "A/B test", "user growth", "MAU", "DAU" → product_signal += 1 per hit
   - Decision:
     * If consulting_signal > product_signal * 2 → "consulting"
     * If product_signal > consulting_signal * 2 → "product"
     * Else → "unknown"

   TIER 4: FALLBACK
   - If still unclassified (e.g., only 1 employee, no description) → "unknown".

3. Output:
   - Write to `data/company_classifications.json`
   - Format: {"Company Name": "product", "Another Corp": "consulting", ...}
   - Print summary: "{n} companies classified: {x} product, {y} consulting, {z} research, {w} non_tech, {u} unknown"

4. CLI interface:
   python src/company_classifier.py --candidates ./candidates.jsonl.gz --output ./data/company_classifications.json

5. Performance: processing 100K candidates should take < 60 seconds. Use efficient data structures (defaultdict for grouping).

IMPORTANT:
- This file is a SCRIPT with a __main__ block, not just a library.
- It must be re-runnable (overwrite output file if it exists).
- Include a --sample flag that processes only the first 1000 candidates for quick testing.
```

---

## Module 5: `skill_scorer.py`

```
CONTEXT:
You are scoring candidates against a JD for a Senior ML/AI Engineer. The JD has 4 "must-have" skill categories (production retrieval, vector DBs, Python, evaluation frameworks) and several "nice-to-have" skills. The JD explicitly warns that candidates who keyword-stuff their profiles with AI buzzwords but have no real depth should be downranked, not upranked.

TASK:
Create `src/skill_scorer.py` with a function that computes a skill match score from 0 to 40.

REQUIREMENTS:

1. Function signature:
   def compute_skill_score(candidate: dict) -> tuple[float, dict]
   Returns (score from 0.0 to 40.0, breakdown dict with details).

2. Import MUST_HAVE_SKILL_GROUPS, NICE_TO_HAVE_SKILLS, WRONG_DOMAIN_SKILLS from config.

3. STEP 1: Must-Have Group Matching (max 40 points, 10 per group)
   For each of the 4 groups in MUST_HAVE_SKILL_GROUPS:
     - Iterate through candidate["skills"] (a list of skill dicts, each with "name", "proficiency", "endorsements", "duration_months").
     - For each skill, check if skill["name"] contains ANY keyword from the group (case-insensitive substring match).
     - If at least one match is found:
       base = 7.0
       If the BEST matching skill (by duration) has duration_months >= 12: +1.5
       If the BEST matching skill has endorsements >= 10: +1.0
       If the BEST matching skill has proficiency in ("advanced", "expert"): +0.5
       Group score = min(base + bonuses, 10.0)
     - If no match: group score = 0.
   
   Step 1 total = sum of 4 group scores. Max = 40.

4. STEP 2: Nice-to-Have Bonus (max +5, cannot push total above 40)
   Count how many skills from NICE_TO_HAVE_SKILLS the candidate has (case-insensitive name match).
   Bonus = min(count * 1.5, 5.0)
   Running total = min(step1 + bonus, 40.0)

5. STEP 3: Credibility Penalty
   Count skills where proficiency in ("expert", "advanced") AND duration_months < 6.
   If count >= 5: penalty = -10
   Elif count >= 3: penalty = -5
   Else: penalty = 0
   Running total = max(0.0, running_total + penalty)

6. STEP 4: Assessment Score Bonus (max +5, cannot push total above 40)
   Access candidate["redrob_signals"]["skill_assessment_scores"] — a dict of {skill_name: score_0_to_100}.
   For each assessment where the skill name matches any MUST_HAVE keyword:
     If score >= 70: +2
     If 50 <= score < 70: +1
   Cap assessment bonus at 5.
   Final = min(40.0, running_total + assessment_bonus)

7. STEP 5: Domain Ratio Check (this is NOT applied here — it's handled by alignment_scorer.py). Do NOT penalize wrong-domain skills in this module. Just record the count for the breakdown.
   Count how many skills match WRONG_DOMAIN_SKILLS → store as breakdown["wrong_domain_count"].
   Count how many skills match any MUST_HAVE group → store as breakdown["core_nlp_ir_count"].

8. Breakdown dict should include:
   {"production_retrieval": 8.5, "vector_db": 0, "python": 10, "eval_frameworks": 7, 
    "nice_to_have_bonus": 3, "credibility_penalty": -5, "assessment_bonus": 2,
    "wrong_domain_count": 3, "core_nlp_ir_count": 5, "total": 25.5}

EDGE CASES:
- Candidate has empty skills list → score = 0.
- Candidate has no skill_assessment_scores key → skip Step 4.
- Skill name is None or empty string → skip that skill.
```

---

## Module 6: `career_scorer.py`

```
CONTEXT:
You are scoring the career quality of job candidates against a JD for a Senior ML/AI Engineer. The JD explicitly:
- Rewards product-company backgrounds (Uber, Swiggy, Google, etc.)
- Punishes pure consulting-only careers (TCS, Infosys, Wipro — entire career)
- Wants 5-9 years experience but says "it's soft — judgment matters more than tenure"
- Wants people who WRITE CODE, not pure architects/managers
- Warns that title-description mismatches indicate trap candidates

You have a pre-computed company classification file (company_classifications.json) mapping company names to "product"/"consulting"/"research"/"non_tech"/"unknown".

TASK:
Create `src/career_scorer.py` with a function that computes career quality score from 0 to 30.

REQUIREMENTS:

1. Function signature:
   def compute_career_score(candidate: dict, company_map: dict[str, str]) -> tuple[float, dict]

2. Import ML_AI_TITLES, ADJACENT_TECH_TITLES, NON_TECH_TITLES, PRODUCTION_EVIDENCE_KEYWORDS, CODE_WRITING_EVIDENCE, NON_TECH_DESCRIPTION_KEYWORDS, CONSULTING_FIRMS from config.

3. DIMENSION A: Company Quality (0-12 points)
   For each role in candidate["career_history"]:
     Look up role["company"] in company_map (case-insensitive).
     If not found in company_map, try matching against CONSULTING_FIRMS and PRODUCT_COMPANIES from config as a fallback.
     
     Points per role:
       "product" → +4
       "consulting" → +0
       "research" → +1
       "non_tech" → +0
       "unknown" → +2 (benefit of doubt)
     
   company_score = min(sum_of_points, 12)
   
   SPECIAL FLAG: consulting_only = True if ALL companies in career are classified as "consulting".

4. DIMENSION B: Title Relevance (0-10 points)
   current_title = candidate["profile"]["current_title"]
   Use fuzzy matching: check if current_title is contained in or contains any title from the sets (don't require exact match — "Senior ML Engineer" should match ML_AI_TITLES even if the exact string isn't listed).
   
   If matches ML_AI_TITLES: base = 8
   Elif matches ADJACENT_TECH_TITLES: base = 5
   Elif matches NON_TECH_TITLES: base = 0
   Else: base = 3 (unknown title — neutral)
   
   Historical bonus: count career_history roles whose title matches ML_AI_TITLES.
   title_score = min(base + (ml_history_count * 1.0), 10)

5. DIMENSION C: Experience Depth (0-8 points)
   years = candidate["profile"]["years_of_experience"]
   Scoring bands:
     5 <= years <= 9:  8 points (JD's sweet spot)
     4 <= years < 5:   6 points ("early bloomers welcome")
     9 < years <= 12:  6 points (slightly senior, fine)
     3 <= years < 4:   4 points (junior side)
     12 < years <= 15: 4 points (quite senior)
     years > 15:       3 points (very senior — might not code)
     years < 3:        2 points (too junior for this role)

6. DIMENSION D: Career Description Analysis (adjusts final score, can add up to +3 or subtract up to -8)
   Concatenate all career_history[*].description into one string. Lowercase it.
   
   Count hits from PRODUCTION_EVIDENCE_KEYWORDS → production_count
   Count hits from CODE_WRITING_EVIDENCE → code_count
   Count hits from NON_TECH_DESCRIPTION_KEYWORDS → nontech_count
   
   If production_count >= 3: bonus = +3
   Elif production_count >= 1: bonus = +1
   Else: bonus = 0
   
   If nontech_count >= 5 AND production_count == 0: penalty = -5
   If nontech_count >= 8 AND production_count == 0: penalty = -8
   Else: penalty = 0
   
   TITLE-DESCRIPTION MISMATCH FLAG:
   If current_title matches ML_AI_TITLES AND nontech_count >= 3 AND production_count == 0:
     mismatch_penalty = -8
     Set flag: title_description_mismatch = True
   Else: mismatch_penalty = 0

   career_score = max(0, min(30, company + title + experience + bonus + penalty + mismatch_penalty))

7. Breakdown dict:
   {"company_score": 8, "title_score": 6, "exp_score": 8, "production_bonus": 3,
    "nontech_penalty": 0, "mismatch_penalty": 0, "consulting_only": False,
    "title_description_mismatch": False, "total": 25}

EDGE CASES:
- Empty career_history → company_score = 0, no descriptions to analyze.
- career_history with only 1 entry → that one entry determines everything.
- Company name not in company_map AND not in seed lists → default to "unknown" = 2 points.
```

---

## Module 7: `alignment_scorer.py`

```
CONTEXT:
You are scoring how well a candidate aligns with secondary requirements: location preference (Noida/Pune preferred, India tier-1 OK, outside India penalized), education relevance, notice period feasibility, and domain fit (NLP/IR preferred, CV/speech penalized ONLY if it dominates the profile).

TASK:
Create `src/alignment_scorer.py` — scores 0 to 15.

REQUIREMENTS:

1. Function signature:
   def compute_alignment_score(candidate: dict, skill_breakdown: dict) -> tuple[float, dict]
   Note: skill_breakdown is the breakdown dict from skill_scorer, which contains "wrong_domain_count" and "core_nlp_ir_count".

2. DIMENSION A: Location (0-8 points)
   Access candidate["profile"]["location"] and candidate["profile"]["country"].
   Access candidate["redrob_signals"]["willing_to_relocate"].
   
   Check if any word from PREFERRED_LOCATIONS appears in location (case-insensitive substring):
     → 8 points
   Elif country == "India" AND any word from TIER1_INDIA_CITIES in location:
     If willing_to_relocate: → 7
     Else: → 5
   Elif country == "India" (other cities):
     If willing_to_relocate: → 5
     Else: → 3
   Elif country != "India":
     → 2

3. DIMENSION B: Education (0-4 points)
   For each entry in candidate["education"]:
     field = entry["field_of_study"].lower()
     tier = entry.get("tier", "tier_4")
     
     Field score:
       If field contains any of: "computer science", "machine learning", "artificial intelligence", "data science", "information technology", "statistics", "mathematics" → 2.0
       Elif field contains: "electronics", "electrical", "computer engineering" → 1.5
       Elif field contains: "physics", "engineering" → 1.0
       Else → 0.5
     
     Tier score: tier_1 → 1.5, tier_2 → 1.0, tier_3 → 0.5, tier_4 → 0, anything else → 0
     
     Entry score = field_score + tier_score
   
   Take the MAX across all education entries. Cap at 4.

4. DIMENSION C: Notice Period (0-3 points)
   notice = candidate["redrob_signals"]["notice_period_days"]
   open_to_work = candidate["redrob_signals"]["open_to_work_flag"]
   
   If notice <= 30: 3.0
   Elif notice <= 60: 2.5
   Elif notice <= 90 AND open_to_work: 2.0
   Elif notice <= 90: 1.5
   Elif notice > 90 AND open_to_work: 1.0
   Elif notice > 90: 0.5

5. DOMAIN FIT ADJUSTMENT (applies as a penalty to the total, not a separate dimension)
   Use skill_breakdown["wrong_domain_count"] and skill_breakdown["core_nlp_ir_count"].
   
   If wrong_domain_count > 0:
     If core_nlp_ir_count == 0:
       domain_penalty = -6  (All CV/speech, zero NLP/IR → bad fit)
     Elif wrong_domain_count > core_nlp_ir_count * 3:
       domain_penalty = -3  (Mostly wrong domain, tiny NLP/IR)
     Else:
       domain_penalty = 0   (Has NLP/IR foundation, CV is bonus knowledge)
   Else:
     domain_penalty = 0
   
   alignment_score = max(0, location + education + notice + domain_penalty)
   Cap at 15.

6. Breakdown dict:
   {"location_score": 7, "education_score": 3.5, "notice_score": 2.5,
    "domain_penalty": 0, "total": 13.0}
```

---

## Module 8: `contradiction.py`

```
CONTEXT:
In the hackathon dataset, ~80 honeypots have "subtly impossible" profiles. Some are caught by the honeypot detector (salary inversion, timeline paradox). Others slip through because their individual data points are plausible — but the COMBINATION is contradictory. For example: claiming "expert" in 8 skills but having zero endorsements and zero assessment scores. Each claim alone is possible; together they're implausible.

This module catches those that slip through the hard honeypot check.

TASK:
Create `src/contradiction.py` — returns a penalty multiplier from 0.5 to 1.0.

REQUIREMENTS:

1. Function signature:
   def compute_contradiction_penalty(candidate: dict, career_flags: dict) -> tuple[float, list[str]]
   career_flags comes from career_scorer's breakdown (has "title_description_mismatch" bool).
   Returns (multiplier 0.5 to 1.0, list of contradiction reason strings).

2. Start with contradiction_count = 0. Apply each check:

   CHECK 1: Expert claims + zero duration
   Count skills where proficiency in ("expert", "advanced") AND duration_months < 6.
   If count >= 6: contradiction_count += 2, reason: "X skills claimed advanced/expert with <6mo duration"
   Elif count >= 3: contradiction_count += 1, reason: same

   CHECK 2: Many skills + zero assessments
   If len(candidate["skills"]) >= 10 AND candidate["redrob_signals"]["skill_assessment_scores"] is empty:
     contradiction_count += 1, reason: "{n} skills claimed but zero assessment scores"

   CHECK 3: Low profile completeness + expert claims
   If profile_completeness_score < 35 AND count of expert/advanced skills >= 3:
     contradiction_count += 1, reason: "Profile {pct}% complete but claims {n} advanced/expert skills"

   CHECK 4: Title-description mismatch (from career scorer)
   If career_flags.get("title_description_mismatch", False):
     contradiction_count += 1, reason: "Title does not match career description content"

   CHECK 5: Career history gap
   total_career_months = sum(career_history[*].duration_months)
   claimed_months = years_of_experience * 12
   If total_career_months < claimed_months * 0.4:
     contradiction_count += 1, reason: "Career history only accounts for {pct}% of claimed experience"

   CHECK 6: High endorsements + low connections
   If endorsements_received > 40 AND connection_count < 30:
     contradiction_count += 1, reason: "{e} endorsements but only {c} connections"

   CHECK 7: Summary mentions different role than title
   summary = candidate["profile"]["summary"].lower()
   title = candidate["profile"]["current_title"].lower()
   If summary contains "marketing manager" but title does NOT contain "marketing": contradiction_count += 0.5
   If summary contains "marketing manager" and title contains "engineer" or "scientist": contradiction_count += 1, reason: "Summary mentions 'marketing manager' but title is '{title}'"
   (Also check for other mismatches: summary says "accountant" but title says "engineer", etc.)

3. Convert count to multiplier:
   If contradiction_count == 0: 1.0
   If 0 < contradiction_count <= 1: 0.85
   If 1 < contradiction_count <= 2: 0.7
   If 2 < contradiction_count <= 3: 0.55
   If contradiction_count > 3: 0.5

4. Return the multiplier and the list of reason strings.

EDGE CASES:
- Candidate with no skills → skip all skill-related checks.
- Candidate with no career_history → skip CHECK 5.
- Missing redrob_signals keys → treat as non-contradictory for that check.
```

---

## Module 9: `behavioral.py`

```
CONTEXT:
The JD explicitly says: "Your ranking system should weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately."

This module computes a MULTIPLICATIVE modifier. A value of 0.2 means an amazing fit-score of 90 becomes 18. This is intentional — an unavailable candidate is worse than a mediocre available one.

TASK:
Create `src/behavioral.py` — returns a multiplier from 0.1 to 1.0.

REQUIREMENTS:

1. Function signature:
   def compute_behavioral_multiplier(signals: dict) -> tuple[float, dict]
   signals = candidate["redrob_signals"]
   Returns (multiplier 0.1 to 1.0, breakdown dict).

2. Import BEHAVIORAL thresholds and REFERENCE_DATE from config.

3. SIGNAL 1: Activity Recency
   Parse signals["last_active_date"] as a date.
   Calculate days_since = (REFERENCE_DATE - last_active_date).days
   If days_since is negative (future date), treat as 0 (very active).
   Look up in BEHAVIORAL["recency_tiers"]: find first tier where days_since < tier[0], use tier[1].

4. SIGNAL 2: Recruiter Response Rate
   rr = signals["recruiter_response_rate"]
   Tiers from config: >0.6 → 1.0, >0.3 → 0.7, else → 0.35

5. SIGNAL 3: Availability Composite
   open_to_work = signals["open_to_work_flag"]
   notice = signals["notice_period_days"]
   
   If open_to_work: 1.0
   Elif notice <= 60: 0.8
   Elif notice <= 90: 0.6
   Else: 0.3

6. SIGNAL 4: Interview Completion Rate
   icr = signals["interview_completion_rate"]
   Tiers: >0.7 → 1.0, >0.4 → 0.7, else → 0.3

7. SIGNAL 5: Offer Acceptance Rate
   oar = signals["offer_acceptance_rate"]
   If oar == -1: 0.7 (no data — slight penalty)
   If oar == 0: 0.15 (never accepted — huge red flag)
   If oar > 0.5: 1.0
   If 0 < oar <= 0.5: 0.6

8. COMBINE via geometric mean:
   raw = (s1 * s2 * s3 * s4 * s5) ** (1/5)
   multiplier = max(0.1, min(1.0, raw))

9. Breakdown dict:
   {"recency": 0.8, "response_rate": 1.0, "availability": 0.6,
    "interview_completion": 0.7, "offer_acceptance": 0.7,
    "combined": 0.75}

EDGE CASES:
- last_active_date is null or unparseable → use recency = 0.2 (assume very stale).
- Any signal value is null → use the worst-case value for that signal.
```

---

## Module 10: `embedding_scorer.py`

```
CONTEXT:
The hackathon JD warns: "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit." Embeddings catch these semantic matches that keyword rules miss.

This module adds a 0-15 point semantic similarity score. It's the ONLY module that requires an external ML library (sentence-transformers).

TASK:
Create `src/embedding_scorer.py` — batch-computes semantic similarity between a JD and all candidates' career text.

REQUIREMENTS:

1. Function signature:
   def compute_embedding_scores(candidates: list[dict]) -> list[float]
   Returns a list of scores (0.0 to 15.0), one per candidate, same order as input.

2. Import JD_TEXT_FOR_EMBEDDING from config.

3. MODEL: Use "all-MiniLM-L6-v2" from sentence-transformers.
   - This is a 22M parameter model, runs fast on CPU (~10K encodings/sec).
   - Load it once at module level or inside the function (lazy loading is fine).
   - If sentence-transformers is not installed, return a list of 7.5 (neutral mid-score) for all candidates and print a warning.

4. For each candidate, build the text to embed:
   text = candidate["profile"]["summary"]
   for role in candidate["career_history"]:
       text += " " + role["description"]
   # Truncate to 512 characters if very long (the model has a 256-token limit anyway)

5. Batch encode ALL candidate texts at once for speed:
   all_texts = [build_text(c) for c in candidates]
   candidate_embeddings = model.encode(all_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
   jd_embedding = model.encode(JD_TEXT_FOR_EMBEDDING, normalize_embeddings=True)

6. Compute cosine similarity:
   Since embeddings are normalized, cosine_similarity = dot product.
   similarities = candidate_embeddings @ jd_embedding  (matrix-vector dot product)
   Each similarity is 0.0 to 1.0.

7. Scale to 0-15:
   scores = [max(0.0, min(15.0, sim * 15.0)) for sim in similarities]

8. Performance target: < 60 seconds for 100K candidates on CPU.

DEPENDENCIES:
   sentence-transformers (which installs torch CPU automatically)
   numpy

EDGE CASES:
- Candidate with empty summary AND empty career_history descriptions → text is empty string → similarity will be near 0 → score near 0 (correct behavior — they have no information).
- Very long text → truncate to 512 chars before encoding.
```

---

## Module 11: `reasoning.py`

```
CONTEXT:
Each row in the output CSV requires a "reasoning" column — a 1-2 sentence explanation of WHY this candidate was ranked here. The hackathon judges review these at Stage 4 for quality. Good reasoning uses specific data from the candidate's profile. Bad reasoning is generic ("strong candidate with relevant experience").

TASK:
Create `src/reasoning.py` — generates a specific, data-grounded reasoning string per candidate.

REQUIREMENTS:

1. Function signature:
   def generate_reasoning(candidate: dict, subscores: dict) -> str
   subscores is a dict containing all breakdown dicts from previous modules.
   Returns a string, max 200 characters.

2. Build the reasoning from specific candidate data:

   Part 1 — Career headline:
   "{years}yr {title} at {company}"
   years = profile.years_of_experience (format as integer or 1 decimal)
   title = profile.current_title
   company = profile.current_company

   Part 2 — Key strength (pick the HIGHEST-scoring dimension):
   If skill_score >= 30: "strong retrieval/ranking skill match"
   Elif career_score >= 25: "deep product-company ML track record"
   Elif embedding_score >= 12: "semantically aligned career trajectory"
   Elif behavioral >= 0.85: "highly active and responsive"
   Else: "relevant technical background"

   Part 3 — Top matched skills (up to 3):
   Extract skill names that matched must-have groups from the skill breakdown.
   Format: "Skills: Python, FAISS, NDCG"

   Part 4 — Location + notice:
   "{location}, {notice}d notice"

   Part 5 — Behavioral signal:
   "{response_rate}% response rate"
   response_rate = recruiter_response_rate * 100, formatted as integer.

3. Combine parts with "; " separator. Truncate to 200 chars if needed.

4. CRITICAL: The reasoning field in the CSV must be properly quoted if it contains commas. Use Python's csv.writer which handles this automatically — but do NOT include bare commas or newlines in the string yourself.

5. If any field is missing/null, skip that part gracefully — don't crash, just omit it.

EXAMPLE OUTPUT:
"7yr ML Engineer at Swiggy; strong retrieval/ranking skill match; Skills: FAISS, Python, BM25; Hyderabad, 30d notice; 91% response rate"
```

---

## Module 12: `ranker.py` — Main Pipeline

```
CONTEXT:
This is the entry point. It wires all modules together. It must:
- Process all 100K candidates
- Score every single one (no filtering, no removal except honeypots getting score=0)
- Output exactly 100 rows as a CSV
- Run in under 5 minutes on CPU with no network

TASK:
Create `ranker.py` (in the project root, not src/) — the main executable.

REQUIREMENTS:

1. CLI interface:
   python rank.py --candidates ./candidates.jsonl.gz --output ./submission.csv
   Optional: --company-map ./data/company_classifications.json (defaults to this path)
   Optional: --sample (only process first 100 candidates, for quick testing)

2. Use argparse for CLI parsing.

3. PIPELINE (in this exact order):

   STEP 1: Load data
   candidates = load_candidates(args.candidates)
   company_map = load_company_classifications(args.company_map)
   Print: "Loaded {n} candidates, {m} company classifications"

   STEP 2: Flag honeypots
   honeypot_flags = flag_honeypots(candidates)
   Print: "Flagged {n} honeypots"

   STEP 3: Compute embedding scores (batch — all at once)
   embedding_scores = compute_embedding_scores(candidates)
   Print: "Computed embedding scores"

   STEP 4: Score each candidate
   For each candidate (use tqdm progress bar if available):
     a. skill_score, skill_bd = compute_skill_score(candidate)
     b. career_score, career_bd = compute_career_score(candidate, company_map)
     c. alignment_score, align_bd = compute_alignment_score(candidate, skill_bd)
     d. contra_mult, contra_reasons = compute_contradiction_penalty(candidate, career_bd)
     e. behav_mult, behav_bd = compute_behavioral_multiplier(candidate["redrob_signals"])
     f. emb_score = embedding_scores[i]
     
     g. If honeypot_flags[candidate_id] is True:
          final_score = 0.0
        Else:
          raw_fit = skill_score + career_score + alignment_score + emb_score
          final_score = raw_fit * contra_mult * behav_mult
     
     h. Store result dict with all scores and breakdowns.

   STEP 5: Sort by final_score descending. Tiebreak by candidate_id ascending.
   
   STEP 6: Take top 100.

   STEP 7: Verify:
   - Print "Honeypots in top 100: {n}" — assert this is 0 or very low.
   - Print score distribution: "Score range: {min} to {max}"
   - Print title distribution of top 10 (sanity check).

   STEP 8: Generate reasoning for top 100.

   STEP 9: Write CSV with columns: candidate_id, rank, score, reasoning
   - rank goes from 1 to 100
   - score should have 4 decimal places
   - Use csv.writer with proper quoting

   STEP 10: Print "Written {output_path} with {n} candidates"

4. TIMING: Wrap the entire pipeline in a timer. Print "Total time: {seconds}s" at the end.

5. IMPORTANT VALIDATION RULES (from submission_spec):
   - Exactly 100 rows (not 99, not 101)
   - Ranks 1-100 (not 0-99)
   - No duplicate candidate_ids
   - Scores must be non-increasing (rank 1 has highest score)
   - All candidate_ids must exist in the input file
   - CSV columns: candidate_id,rank,score,reasoning (this exact header)

EDGE CASES:
- If fewer than 100 candidates have score > 0 (unlikely but possible), still output 100 — include the best-scoring ones even if their score is low.
- If two candidates have identical final_score, sort by candidate_id alphabetically as tiebreak.
```

---

## Module 13: `app.py` — Streamlit Sandbox

```
CONTEXT:
The hackathon requires a "sandbox" — a hosted environment (HuggingFace Spaces, Streamlit Cloud, etc.) where organizers can verify your ranking system runs. It only needs to handle a SMALL sample (≤100 candidates), not the full 100K.

TASK:
Create `app.py` in the project root — a Streamlit app for deployment on HuggingFace Spaces.

REQUIREMENTS:

1. UI Layout:
   - Title: "Redrob Candidate Ranker"
   - Subtitle: "Intelligent Candidate Discovery & Ranking System"
   - File upload widget: accepts .json or .jsonl files (small samples only)
   - "Run Ranking" button
   - Results displayed as a dataframe table with columns: Rank, Candidate ID, Score, Reasoning
   - Download button for the output CSV

2. On "Run Ranking":
   - Load uploaded candidates using loader.py
   - Run the full pipeline (same as rank.py but without CLI)
   - Display results in a table
   - Show timing: "Ranked {n} candidates in {t:.1f} seconds"
   - Show honeypot count: "Honeypots detected: {n}"

3. Sidebar:
   - Show system info: "Model: all-MiniLM-L6-v2", "Scoring: Structured + Semantic"
   - Brief methodology description (3-4 bullet points)

4. Error handling:
   - If file is too large (>10MB), show error.
   - If file format is wrong, show clear error message.
   - If pipeline crashes, catch exception and show error.

5. For HuggingFace Spaces deployment, include a requirements.txt with:
   streamlit, sentence-transformers, orjson, tqdm, numpy

6. Styling: Use st.set_page_config(page_title="Redrob Ranker", layout="wide").
   Keep it clean and functional — no need for fancy CSS.

IMPORTANT: The sandbox must work WITHOUT the company_classifications.json file (fall back to seed lists only). This makes deployment simpler since you don't need to upload the pre-computed file.
```

---

## Build Order (Final)

```
MODULE  | FILE                      | DEPENDS ON              | TEST WITH
--------|---------------------------|-------------------------|------------------
  1     | src/config.py             | Nothing                 | Import, print lists
  2     | src/loader.py             | config                  | Load sample_candidates.json
  3     | src/honeypot.py           | —                       | Flag CAND_0000009, CAND_0000006
  4     | src/company_classifier.py | config, loader          | Run on sample, check output JSON
  5     | src/skill_scorer.py       | config                  | Score 5 sample candidates manually
  6     | src/career_scorer.py      | config                  | Score 5 sample candidates manually
  7     | src/alignment_scorer.py   | config                  | Score 5 sample candidates
  8     | src/contradiction.py      | —                       | Check CAND_0000006 gets penalized
  9     | src/behavioral.py         | config                  | Check active vs stale candidates
  10    | src/embedding_scorer.py   | config                  | Batch score 50 samples
  11    | src/reasoning.py          | —                       | Generate for 5 candidates
  12    | rank.py                   | ALL above               | Full pipeline on 50 samples
  13    | app.py                    | ALL above               | streamlit run app.py
```
