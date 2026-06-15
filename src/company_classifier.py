"""
src/company_classifier.py
=========================
Offline pre-computation script.  Reads the full candidate dataset, extracts
every unique company name, and classifies each as one of:

    product | consulting | research | non_tech | unknown

Classification follows a strict four-tier priority system:
    1. Seed-list match (from config.py)
    2. Industry-based heuristic (most common industry per company)
    3. Employee signal aggregation (title + description keyword counting)
    4. Fallback → "unknown"

This script is meant to run ONCE before the ranking step.  It is NOT part
of the 5-minute ranking constraint and can take as long as it needs (though
it typically finishes in <60 seconds for 100K candidates).

CLI
---
    python src/company_classifier.py \\
        --candidates ./candidates.jsonl.gz \\
        --output ./data/company_classifications.json \\
        [--sample]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CONSULTING_FIRMS,
    CONSULTING_INDUSTRIES,
    FICTIONAL_COMPANIES,
    PRODUCT_COMPANIES,
    PRODUCT_INDUSTRIES,
)
from src.loader import load_candidates


# ═══════════════════════════════════════════════════════════════════════════
# Build normalised lookup tables from seed lists
# ═══════════════════════════════════════════════════════════════════════════

def _normalise(name: str) -> str:
    """Lowercase, strip whitespace — the canonical key for company lookups."""
    return name.strip().lower()


_CONSULTING_LOOKUP: set[str] = {_normalise(c) for c in CONSULTING_FIRMS}
_PRODUCT_LOOKUP:    set[str] = {_normalise(c) for c in PRODUCT_COMPANIES}
_FICTIONAL_LOOKUP:  set[str] = {_normalise(c) for c in FICTIONAL_COMPANIES}


# ═══════════════════════════════════════════════════════════════════════════
# Tier-3 keyword patterns
# ═══════════════════════════════════════════════════════════════════════════

_CONSULTING_TITLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bConsultant\b",
        r"\bImplementation\b",
        r"\bDelivery Manager\b",
        r"\bBusiness Analyst\b",
    )
]

_PRODUCT_TITLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bProduct Manager\b",
        r"\bML Engineer\b",
        r"\bData Scientist\b",
        r"\bSoftware Engineer\b",
        r"\bSRE\b",
        r"\bFrontend\b",
        r"\bBackend\b",
    )
]

_CONSULTING_DESC_KEYWORDS: list[str] = [
    "client", "deliverable", "engagement", "sow", "consulting",
]

_PRODUCT_DESC_KEYWORDS: list[str] = [
    "our product", "our users", "shipped", "deployed",
    "production", "a/b test", "user growth", "mau", "dau",
]

# Tier-2 keyword-in-industry substrings for research / non-tech
_RESEARCH_SUBSTRINGS:  list[str] = ["research", "academic", "university", "education"]
_NON_TECH_SUBSTRINGS:  list[str] = [
    "manufacturing", "construction", "mining", "paper products",
    "retail", "real estate", "hospitality",
]


# ═══════════════════════════════════════════════════════════════════════════
# Core classification logic
# ═══════════════════════════════════════════════════════════════════════════

def classify_companies(candidates: list[dict]) -> tuple[dict[str, str], int]:
    """Classify every unique company found across all candidates.

    Parameters
    ----------
    candidates:
        Full list of candidate dicts (each must contain ``career_history``).

    Returns
    -------
    tuple[dict[str, str], int]
        Mapping of original company name → classification string, and the number of unique (company, industry, company_size) tuples.
    """

    # ── Pass 1: collect per-company data ─────────────────────────────────
    # For each unique company, gather all industries, titles, and
    # description fragments from every candidate who listed that company.

    industries_by_company:   dict[str, list[str]]  = defaultdict(list)
    titles_by_company:       dict[str, list[str]]  = defaultdict(list)
    descriptions_by_company: dict[str, list[str]]  = defaultdict(list)
    original_names:          dict[str, str]         = {}  # norm → first-seen form
    unique_tuples:           set[tuple[str, str, str]] = set()

    for candidate in candidates:
        for role in candidate.get("career_history", []):
            company_raw = role.get("company", "")
            if not company_raw or not isinstance(company_raw, str):
                continue
                
            industry_raw = role.get("industry", "")
            if not isinstance(industry_raw, str): industry_raw = ""
            
            size_raw = role.get("company_size", "")
            if not isinstance(size_raw, str): size_raw = ""

            unique_tuples.add((company_raw.strip(), industry_raw.strip(), size_raw.strip()))

            key = _normalise(company_raw)

            # Keep the first-seen capitalisation for the output JSON
            if key not in original_names:
                original_names[key] = company_raw.strip()

            if industry_raw:
                industries_by_company[key].append(industry_raw.strip())

            title = role.get("title", "")
            if title and isinstance(title, str):
                titles_by_company[key].append(title)

            desc = role.get("description", "")
            if desc and isinstance(desc, str):
                descriptions_by_company[key].append(desc)

    # ── Pass 2: classify each company ────────────────────────────────────
    classifications: dict[str, str] = {}

    for key, display_name in original_names.items():

        # TIER 1 — seed list match
        label = _tier1_seed(key)
        if label is not None:
            classifications[display_name] = label
            continue

        # TIER 2 — industry heuristic
        label = _tier2_industry(industries_by_company.get(key, []))
        if label is not None:
            classifications[display_name] = label
            continue

        # TIER 3 — employee signal aggregation
        label = _tier3_employee_signals(
            titles_by_company.get(key, []),
            descriptions_by_company.get(key, []),
        )
        if label is not None:
            classifications[display_name] = label
            continue

        # TIER 4 — fallback
        classifications[display_name] = "unknown"

    return classifications, len(unique_tuples)


# ─── Tier helpers ────────────────────────────────────────────────────────

def _tier1_seed(norm_name: str) -> str | None:
    if norm_name in _FICTIONAL_LOOKUP:
        return "unknown"
    if norm_name in _CONSULTING_LOOKUP:
        return "consulting"
    if norm_name in _PRODUCT_LOOKUP:
        return "product"
    return None


def _tier2_industry(industries: list[str]) -> str | None:
    if not industries:
        return None

    most_common = Counter(industries).most_common(1)[0][0]

    if most_common in CONSULTING_INDUSTRIES:
        return "consulting"
    if most_common in PRODUCT_INDUSTRIES:
        return "product"

    mc_lower = most_common.lower()
    for sub in _RESEARCH_SUBSTRINGS:
        if sub in mc_lower:
            return "research"
    for sub in _NON_TECH_SUBSTRINGS:
        if sub in mc_lower:
            return "non_tech"

    return None


def _tier3_employee_signals(
    titles: list[str],
    descriptions: list[str],
) -> str | None:
    if not titles and not descriptions:
        return None

    consulting_signal = 0
    product_signal = 0

    # Title patterns
    for title in titles:
        for pat in _CONSULTING_TITLE_PATTERNS:
            if pat.search(title):
                consulting_signal += 1
        for pat in _PRODUCT_TITLE_PATTERNS:
            if pat.search(title):
                product_signal += 1

    # Description keywords
    combined_desc = " ".join(descriptions).lower()
    for kw in _CONSULTING_DESC_KEYWORDS:
        consulting_signal += combined_desc.count(kw)
    for kw in _PRODUCT_DESC_KEYWORDS:
        product_signal += combined_desc.count(kw)

    if consulting_signal > product_signal * 2:
        return "consulting"
    if product_signal > consulting_signal * 2:
        return "product"
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Output + CLI
# ═══════════════════════════════════════════════════════════════════════════

def _write_output(classifications: dict[str, str], unique_count: int, output_path: str) -> None:
    """Write the classification map to JSON and print a summary."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(classifications, f, indent=2, ensure_ascii=False)

    counts: dict[str, int] = Counter(classifications.values())
    print(f"Summary:")
    print(f"  total unique companies found: {unique_count}")
    print(f"  classified as product: {counts.get('product', 0)}")
    print(f"  classified as consulting: {counts.get('consulting', 0)}")
    print(f"  classified as unknown: {counts.get('unknown', 0)}")
    print(f"  classified as research: {counts.get('research', 0)}")
    print(f"  classified as non_tech: {counts.get('non_tech', 0)}")
    print(f"Written to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify companies from candidate career histories.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidate data file (.json, .jsonl, .jsonl.gz)",
    )
    parser.add_argument(
        "--output",
        default="./data/company_classifications.json",
        help="Output JSON path (default: ./data/company_classifications.json)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process only the first 1000 candidates for quick testing.",
    )

    args = parser.parse_args(argv)

    candidates = load_candidates(args.candidates)

    if args.sample:
        candidates = candidates[:1000]
        print(f"[SAMPLE MODE] Using first {len(candidates)} candidates")

    classifications, unique_count = classify_companies(candidates)
    _write_output(classifications, unique_count, args.output)


if __name__ == "__main__":
    main()
