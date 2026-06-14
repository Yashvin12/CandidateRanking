"""
src/loader.py
=============
Loads candidate data from any supported format and company-classification
mappings from a JSON file.

Supported candidate formats
---------------------------
- ``.json``        — JSON array  (json.load / orjson.loads)
- ``.jsonl``       — One JSON object per line  (line-by-line)
- ``.jsonl.gz``    — Gzipped JSONL  (gzip.open in text mode)

Usage example
-------------
    from src.loader import load_candidates, load_company_classifications

    # Load the 50-candidate test set
    candidates = load_candidates("data/sample_candidates.json")

    # Load the full 100K dataset
    candidates = load_candidates("data/candidates.jsonl.gz")

    # Load pre-computed company classifications
    company_map = load_company_classifications("data/company_classifications.json")
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional fast JSON backend (10-20× faster than stdlib for bulk parsing)
# ---------------------------------------------------------------------------
try:
    import orjson  # type: ignore

    def _loads(s: str | bytes) -> dict:  # type: ignore[return]
        return orjson.loads(s)

    def _load_file(fp) -> list:  # type: ignore[return]
        return orjson.loads(fp.read())

    _JSON_BACKEND = "orjson"

except ModuleNotFoundError:
    def _loads(s: str | bytes) -> dict:  # type: ignore[return]
        return json.loads(s)

    def _load_file(fp) -> list:  # type: ignore[return]
        return json.load(fp)

    _JSON_BACKEND = "stdlib json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> list[dict]:
    """Load candidates from a ``.json``, ``.jsonl``, or ``.jsonl.gz`` file.

    Parameters
    ----------
    path:
        Absolute or relative path to the candidate file.

    Returns
    -------
    list[dict]
        A list of candidate dicts, each guaranteed to have a ``candidate_id``
        key.  Entries missing that key are skipped with a printed warning.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file extension is not one of the three supported formats.

    Examples
    --------
    >>> candidates = load_candidates("data/sample_candidates.json")
    Loaded 50 candidates from data/sample_candidates.json

    >>> candidates = load_candidates("data/candidates.jsonl.gz")
    Loaded 100000 candidates from data/candidates.jsonl.gz
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: '{path}'. "
            "Check the path and try again."
        )

    suffix_lower = "".join(file_path.suffixes).lower()  # e.g. ".jsonl.gz"

    raw_records: list[dict] = []

    # ── .json ────────────────────────────────────────────────────────────────
    if suffix_lower == ".json":
        with file_path.open("rb") as fh:
            content = fh.read()
        if not content.strip():
            print(f"[WARNING] Empty file: '{path}'. Returning empty list.")
            return []
        raw_records = _loads(content)  # type: ignore[assignment]
        if not isinstance(raw_records, list):
            raw_records = [raw_records]

    # ── .jsonl ───────────────────────────────────────────────────────────────
    elif suffix_lower == ".jsonl":
        raw_records = _read_jsonl_lines(file_path, path, compressed=False)

    # ── .jsonl.gz ────────────────────────────────────────────────────────────
    elif suffix_lower == ".jsonl.gz":
        raw_records = _read_jsonl_lines(file_path, path, compressed=True)

    else:
        raise ValueError(
            f"Unsupported file format '{suffix_lower}' for '{path}'. "
            "Expected one of: .json, .jsonl, .jsonl.gz"
        )

    # ── Validate candidate_id ────────────────────────────────────────────────
    validated: list[dict] = []
    for idx, record in enumerate(raw_records):
        if not isinstance(record, dict):
            print(f"[WARNING] Entry at index {idx} is not a dict — skipping.")
            continue
        if "candidate_id" not in record:
            print(
                f"[WARNING] Entry at index {idx} is missing 'candidate_id' — skipping."
            )
            continue
        validated.append(record)

    print(f"Loaded {len(validated)} candidates from {path}")
    return validated


def load_company_classifications(path: str) -> dict[str, str]:
    """Load a JSON file mapping company names → classification string.

    Expected format::

        {
            "Google": "product",
            "TCS": "consulting",
            "MIT CSAIL": "research",
            "Hooli": "unknown"
        }

    Parameters
    ----------
    path:
        Path to the company classifications JSON file.

    Returns
    -------
    dict[str, str]
        Mapping of company name → classification.  Returns an empty dict
        (with a printed warning) if the file does not exist, so the pipeline
        can run without a pre-computed map.

    Examples
    --------
    >>> company_map = load_company_classifications("data/company_classifications.json")
    Loaded 3842 company classifications from data/company_classifications.json

    >>> company_map = load_company_classifications("data/missing.json")
    [WARNING] Company classifications file not found: 'data/missing.json'. Continuing without it.
    {}
    """
    file_path = Path(path)

    if not file_path.exists():
        print(
            f"[WARNING] Company classifications file not found: '{path}'. "
            "Continuing without it."
        )
        return {}

    with file_path.open("rb") as fh:
        data: dict[str, str] = _loads(fh.read())  # type: ignore[assignment]

    if not isinstance(data, dict):
        print(
            f"[WARNING] Expected a JSON object in '{path}', "
            f"got {type(data).__name__}. Returning empty dict."
        )
        return {}

    print(f"Loaded {len(data)} company classifications from {path}")
    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_jsonl_lines(
    file_path: Path,
    display_path: str,
    *,
    compressed: bool,
) -> list[dict]:
    """Read a JSONL file (optionally gzipped) line by line.

    Blank lines are silently skipped.  Malformed JSON lines trigger a
    printed warning and are skipped; parsing continues for subsequent lines.
    An entirely empty file triggers a warning and returns an empty list.
    """
    records: list[dict] = []

    open_fn = gzip.open if compressed else open
    open_kwargs: dict = {"mode": "rt", "encoding": "utf-8"}

    with open_fn(file_path, **open_kwargs) as fh:  # type: ignore[call-overload]
        lines_seen = False

        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue  # skip blank lines

            lines_seen = True

            try:
                record = _loads(stripped)
            except (ValueError, KeyError) as exc:
                print(
                    f"[WARNING] Malformed JSON on line {line_no} of "
                    f"'{display_path}' ({exc}) — skipping."
                )
                continue

            records.append(record)

        if not lines_seen:
            print(
                f"[WARNING] File '{display_path}' appears to be empty. "
                "Returning empty list."
            )

    return records
