"""
config.py — Single source of truth for the Redrob Candidate Ranking Pipeline.

All constants, thresholds, keyword lists, and weights live here.
No logic, no functions, no imports — pure data declarations.
To tune the pipeline, edit this file only.
"""

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE DATE
# Used for calculating "days since last active" across the pipeline.
# ══════════════════════════════════════════════════════════════════════════════

REFERENCE_DATE: str = "2026-06-15"


# ══════════════════════════════════════════════════════════════════════════════
# JD TEXT FOR EMBEDDING
# Anchor string used for semantic similarity scoring via embeddings.
# ══════════════════════════════════════════════════════════════════════════════

JD_TEXT_FOR_EMBEDDING: str = (
    "Senior ML/AI Engineer building production ranking, retrieval, and matching systems. "
    "Experience with embeddings-based retrieval, vector databases, hybrid search, semantic search "
    "deployed to real users at scale. Strong Python, ranking evaluation frameworks like NDCG MRR MAP. "
    "Product company background preferred. NLP and information retrieval focus. "
    "Built recommendation systems, search systems, candidate matching systems. "
    "Shipped at meaningful scale. Hands-on coding, not architecture-only."
)


# ══════════════════════════════════════════════════════════════════════════════
# COMPANY SEED LISTS
# Used to classify employer history as consulting, product, or fictional.
# Add or remove company names here to tune classification.
# ══════════════════════════════════════════════════════════════════════════════

CONSULTING_FIRMS: set[str] = {
    # Indian IT Services / Global Consulting
    "TCS",
    "Tata Consultancy Services",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Cognizant Technology Solutions",
    "Capgemini",
    "HCL",
    "HCL Technologies",
    "Tech Mahindra",
    "Mindtree",
    "LTIMindtree",
    "Mphasis",
    "L&T Infotech",
    "LTI",
    "Persistent Systems",
    "Hexaware",
    "NIIT Technologies",
    "Cyient",
    "Zensar",
    "Virtusa",
    "UST Global",
    "UST",
}

PRODUCT_COMPANIES: set[str] = {
    # Big Tech
    "Google",
    "Amazon",
    "Meta",
    "Facebook",
    "Microsoft",
    "Apple",
    "Netflix",
    "Twitter",
    "LinkedIn",
    "Salesforce",
    "Adobe",
    "Atlassian",
    "Spotify",
    "Shopify",
    "Stripe",
    "Databricks",
    # Indian Product Companies
    "Flipkart",
    "Swiggy",
    "Zomato",
    "Razorpay",
    "PhonePe",
    "Paytm",
    "Ola",
    "Uber",
    "Meesho",
    "CRED",
    "Zerodha",
    "Groww",
    "Freshworks",
    "Zoho",
    "Postman",
    "BrowserStack",
    "ShareChat",
    "Dream11",
    "Nykaa",
    "UrbanCompany",
    "BigBasket",
    # Global Product Companies
    "Airbnb",
    "Lyft",
    "DoorDash",
    "Snap",
    "Pinterest",
    "Reddit",
    "Discord",
    "Palantir",
    "Snowflake",
    "Confluent",
    "MongoDB Inc",
    "Cloudflare",
    "Twilio",
    # Fintech Hybrids
    "Monzo",
    "Revolut",
    "Klarna",
    "Robinhood",
}

FICTIONAL_COMPANIES: set[str] = {
    # Synthetic / fictional company names present in the dataset
    "Dunder Mifflin",
    "Acme Corp",
    "Globex Inc",
    "Initech",
    "Pied Piper",
    "Stark Industries",
    "Wayne Enterprises",
    "Umbrella Corp",
    "Hooli",
    "Prestige Worldwide",
}


# ══════════════════════════════════════════════════════════════════════════════
# INDUSTRY CLASSIFICATION HEURISTICS
# Maps LinkedIn / profile industry strings to consulting vs. product signals.
# ══════════════════════════════════════════════════════════════════════════════

CONSULTING_INDUSTRIES: set[str] = {
    "IT Services",
    "IT Services and IT Consulting",
    "Staffing and Recruiting",
    "Management Consulting",
    "Business Consulting",
}

PRODUCT_INDUSTRIES: set[str] = {
    "Software",
    "Internet",
    "Technology",
    "Computer Software",
    "Financial Technology",
    "E-commerce",
    "SaaS",
    "Mobile Apps",
}


# ══════════════════════════════════════════════════════════════════════════════
# TITLE CLASSIFICATION
# Classifies job titles into ML/AI, adjacent tech, or non-technical buckets.
# ══════════════════════════════════════════════════════════════════════════════

ML_AI_TITLES: set[str] = {
    # Core ML/AI titles
    "ML Engineer",
    "Machine Learning Engineer",
    "Data Scientist",
    "Research Scientist",
    "Applied Scientist",
    "NLP Engineer",
    "Search Engineer",
    "Ranking Engineer",
    "Deep Learning Engineer",
    "AI Engineer",
    "AI/ML Engineer",
    "Recommendation Systems Engineer",
    # Senior variants
    "Senior ML Engineer",
    "Senior Machine Learning Engineer",
    "Senior Data Scientist",
    "Senior Research Scientist",
    "Senior Applied Scientist",
    "Senior NLP Engineer",
    "Senior Search Engineer",
    "Senior Ranking Engineer",
    "Senior Deep Learning Engineer",
    "Senior AI Engineer",
    "Senior AI/ML Engineer",
    "Senior Recommendation Systems Engineer",
    # Staff variants
    "Staff ML Engineer",
    "Staff Machine Learning Engineer",
    "Staff Data Scientist",
    "Staff Research Scientist",
    "Staff NLP Engineer",
    "Staff Search Engineer",
    "Staff AI Engineer",
    "Staff AI/ML Engineer",
    # Lead variants
    "Lead ML Engineer",
    "Lead Machine Learning Engineer",
    "Lead Data Scientist",
    "Lead NLP Engineer",
    "Lead Search Engineer",
    "Lead AI Engineer",
    # Principal variants
    "Principal ML Engineer",
    "Principal Machine Learning Engineer",
    "Principal Data Scientist",
    "Principal Research Scientist",
    "Principal NLP Engineer",
    "Principal Search Engineer",
    "Principal AI Engineer",
}

ADJACENT_TECH_TITLES: set[str] = {
    # Core adjacent titles
    "Software Engineer",
    "Backend Engineer",
    "Data Engineer",
    "Full Stack Engineer",
    "DevOps Engineer",
    "Platform Engineer",
    "Infrastructure Engineer",
    "SRE",
    "Analytics Engineer",
    # Senior variants
    "Senior Software Engineer",
    "Senior Backend Engineer",
    "Senior Data Engineer",
    "Senior Full Stack Engineer",
    "Senior DevOps Engineer",
    "Senior Platform Engineer",
    "Senior Infrastructure Engineer",
    "Senior SRE",
    "Senior Analytics Engineer",
}

NON_TECH_TITLES: set[str] = {
    "Marketing Manager",
    "Operations Manager",
    "HR Manager",
    "Accountant",
    "Sales Executive",
    "Content Writer",
    "Graphic Designer",
    "Customer Support",
    "Business Analyst",
    "Project Manager",
    "Civil Engineer",
    "Mechanical Engineer",
    "Brand Designer",
    "Financial Analyst",
    "Supply Chain Manager",
}


# ══════════════════════════════════════════════════════════════════════════════
# SKILL MATCHING KEYWORD GROUPS
# All keyword matching is case-insensitive at runtime.
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# MUST-HAVE skill groups: each group is a distinct capability signal.
# A candidate scores per-group, not per-keyword, to avoid keyword stuffing.
# ---------------------------------------------------------------------------
MUST_HAVE_SKILL_GROUPS: dict[str, list[str]] = {
    "production_retrieval": [
        "sentence-transformers",
        "OpenAI embeddings",
        "BGE",
        "E5",
        "semantic search",
        "embedding",
        "retrieval",
        "ranking system",
        "recommendation system",
        "search system",
        "information retrieval",
        "dense retrieval",
        "neural ranking",
        "reranking",
        "re-ranking",
        "BM25",
        "TF-IDF",
        "hybrid search",
        "RAG",
        "vector search",
        "approximate nearest neighbor",
    ],
    "vector_db": [
        "Pinecone",
        "Weaviate",
        "Qdrant",
        "Milvus",
        "FAISS",
        "OpenSearch",
        "Elasticsearch",
        "vector database",
        "vector store",
        "Chroma",
        "Vespa",
        "Annoy",
        "ScaNN",
        "pgvector",
    ],
    "python": [
        "Python",
    ],
    "eval_frameworks": [
        "NDCG",
        "MRR",
        "MAP",
        "A/B testing",
        "evaluation framework",
        "offline evaluation",
        "ranking evaluation",
        "precision@k",
        "recall@k",
        "click-through rate",
        "CTR",
        "online evaluation",
        "interleaving",
    ],
}

# ---------------------------------------------------------------------------
# NICE-TO-HAVE skills: flat list, used for bonus scoring.
# ---------------------------------------------------------------------------
NICE_TO_HAVE_SKILLS: list[str] = [
    "LoRA",
    "QLoRA",
    "PEFT",
    "fine-tuning",
    "fine tuning",
    "LLM fine-tuning",
    "XGBoost",
    "LightGBM",
    "learning-to-rank",
    "learning to rank",
    "LambdaMART",
    "distributed systems",
    "inference optimization",
    "HR-tech",
    "recruiting tech",
]

# ---------------------------------------------------------------------------
# WRONG-DOMAIN skills: used for ratio-based domain mismatch detection.
# NOT a blind penalty — high ratio signals a CV focused on unrelated domains.
# ---------------------------------------------------------------------------
WRONG_DOMAIN_SKILLS: list[str] = [
    "OpenCV",
    "YOLO",
    "object detection",
    "image classification",
    "image segmentation",
    "computer vision",
    "CNN",
    "speech recognition",
    "TTS",
    "text-to-speech",
    "ASR",
    "robotics",
    "ROS",
    "SLAM",
    "autonomous driving",
    "GANs",
    "GAN",
    "generative adversarial",
]

# ---------------------------------------------------------------------------
# CORE AI SKILLS: broader ML/AI vocabulary for counting skill breadth.
# Includes base skills + all items from MUST_HAVE_SKILL_GROUPS and
# NICE_TO_HAVE_SKILLS (deduplication handled at runtime).
# ---------------------------------------------------------------------------
_must_have_flat: list[str] = [
    kw for group in MUST_HAVE_SKILL_GROUPS.values() for kw in group
]

CORE_AI_SKILLS: list[str] = [
    # Foundation ML/AI
    "Python",
    "TensorFlow",
    "PyTorch",
    "scikit-learn",
    "sklearn",
    "NLP",
    "deep learning",
    "machine learning",
    "neural network",
    "transformer",
    "BERT",
    "GPT",
    "LLM",
    "large language model",
    # MLOps & Tooling
    "MLOps",
    "Kubeflow",
    "MLflow",
    "Weights & Biases",
    "W&B",
    "Hugging Face",
    "huggingface",
    # Engineering Fundamentals
    "feature engineering",
    "model training",
    "model deployment",
    "Spark",
    "SparkML",
    # All must-have and nice-to-have skills (deduplication at runtime)
    *_must_have_flat,
    *NICE_TO_HAVE_SKILLS,
]


# ══════════════════════════════════════════════════════════════════════════════
# CAREER DESCRIPTION KEYWORDS
# Used to parse free-text descriptions for evidence of coding, production work,
# or non-technical roles.
# ══════════════════════════════════════════════════════════════════════════════

PRODUCTION_EVIDENCE_KEYWORDS: list[str] = [
    "shipped",
    "deployed",
    "production",
    "real users",
    "live traffic",
    "serving",
    "served",
    "scale",
    "at scale",
    "million users",
    "A/B test",
    "launched",
    "released",
    "system design",
    "end-to-end",
    "built and maintained",
    "owned",
    "SLA",
    "latency",
    "throughput",
    "uptime",
]

CODE_WRITING_EVIDENCE: list[str] = [
    "built",
    "implemented",
    "wrote",
    "developed",
    "coded",
    "engineered",
    "designed and built",
    "architected and built",
    "Python",
    "code review",
    "pull request",
    "codebase",
    "refactored",
    "optimized",
]

NON_TECH_DESCRIPTION_KEYWORDS: list[str] = [
    # Marketing / Content
    "marketing",
    "brand",
    "SEO",
    "content writing",
    "editorial",
    # Finance / Admin
    "accounting",
    "financial reporting",
    "tax filing",
    # Customer / Sales
    "customer support",
    "support team",
    "ticket",
    "sales",
    "quota",
    "revenue",
    "prospecting",
    # Engineering (non-software)
    "CAD",
    "SolidWorks",
    "Creo",
    "FEA",
    "ANSYS",
    # Design / Branding
    "packaging design",
    "brand identity",
    "logo",
    # Supply Chain
    "supply chain",
    "warehouse",
    "fulfillment",
]


# ══════════════════════════════════════════════════════════════════════════════
# SCORING WEIGHTS
# Must sum to 100. Adjust here to rebalance the overall ranking formula.
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS: dict[str, int] = {
    "skill_match":    40,   # Keyword/skill coverage score
    "career_quality": 30,   # Product vs. consulting, seniority, trajectory
    "embedding":      15,   # Semantic similarity to JD embedding
    "alignment":      15,   # Role title + domain alignment
}


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL MULTIPLIER THRESHOLDS
# Applied as multiplicative adjustments on top of the raw composite score.
# Tiers are evaluated in order; the first matching threshold wins.
# ══════════════════════════════════════════════════════════════════════════════

BEHAVIORAL: dict = {
    # Days since last active → multiplier
    # Format: (max_days_inclusive, multiplier)
    "recency_tiers": [
        (30,   1.0),
        (90,   0.8),
        (180,  0.5),
        (9999, 0.2),
    ],

    # Candidate response rate → multiplier
    # Format: (min_rate_inclusive, multiplier) — evaluated highest-first
    "response_rate_tiers": [
        (0.6, 1.0),
        (0.3, 0.7),
        (0.0, 0.35),
    ],

    # Recruiter interview rate for this candidate → multiplier
    # Format: (min_rate_inclusive, multiplier) — evaluated highest-first
    "interview_rate_tiers": [
        (0.7, 1.0),
        (0.4, 0.7),
        (0.0, 0.3),
    ],

    # Offer rate: special sentinel values mapped to a fixed multiplier
    # -1 = no data / never reached offer stage
    #  0 = reached offer stage, zero offers
    "offer_rate_map": {
        -1: 0.7,
        0:  0.15,
    },

    # Default multiplier when offer_rate > 0
    "offer_rate_default_above_zero": 1.0,

    # If offer_rate is above zero but below this threshold → apply low penalty
    "offer_rate_low_threshold": 0.5,
    "offer_rate_low_value":     0.6,
}


# ══════════════════════════════════════════════════════════════════════════════
# LOCATION SCORING
# Candidates in preferred locations get a bonus; Tier 1 cities are neutral.
# ══════════════════════════════════════════════════════════════════════════════

PREFERRED_LOCATIONS: set[str] = {
    "Noida",
    "Pune",
}

TIER1_INDIA_CITIES: set[str] = {
    "Bangalore",
    "Bengaluru",
    "Hyderabad",
    "Mumbai",
    "Delhi",
    "New Delhi",
    "Gurgaon",
    "Gurugram",
    "Chennai",
    "Kolkata",
}
