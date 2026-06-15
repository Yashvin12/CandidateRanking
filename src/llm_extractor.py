"""
src/llm_extractor.py
====================
Offline LLM Feature Extractor — Architecture B.

Reads ``candidates.jsonl`` (or any supported format), calls an LLM API in
massive parallel async batches, and emits a ``data/llm_features.jsonl``
checkpoint file that is consumed by ``career_scorer.py`` at ranking time.

Because this script runs *offline* (before the 5-minute evaluation window)
it has no compute-time constraint.

Supported LLM back-ends (set via env vars — see README at bottom):
  • Groq      — GROQ_API_KEY          (fastest for inference)
  • OpenAI    — OPENAI_API_KEY
  • Ollama    — OLLAMA_BASE_URL       (local, free, no key needed)

Usage
-----
    # Groq (recommended — 30 req/s free tier)
    set GROQ_API_KEY=gsk_...
    python -m src.llm_extractor --candidates data/candidates.jsonl.gz

    # OpenAI
    set OPENAI_API_KEY=sk-...
    python -m src.llm_extractor --candidates data/candidates.jsonl.gz --provider openai

    # Local Ollama
    set OLLAMA_BASE_URL=http://localhost:11434
    python -m src.llm_extractor --candidates data/candidates.jsonl.gz --provider ollama --model llama3

    # Resume after a crash (reads existing checkpoint, skips already-done IDs)
    python -m src.llm_extractor --candidates data/candidates.jsonl.gz --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llm_extractor")

# Ensure src imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loader import load_candidates  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants & tunables
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT      = Path("data/llm_features.jsonl")
DEFAULT_BATCH_SIZE  = 64       # concurrent API calls in-flight at once
MAX_RETRIES         = 15       # per-candidate retry budget
BASE_BACKOFF_S      = 1.0      # seconds before first retry
MAX_BACKOFF_S       = 120.0    # cap on exponential backoff (was 300)
CHECKPOINT_EVERY    = 200      # flush buffer to disk every N completions
REQUEST_TIMEOUT_S   = 30       # aiohttp per-request timeout

MAX_OUTPUT_TOKENS    = 256      # max_tokens we request from the LLM

# ── Rate-limit sentinel returned from _call_api on unrecoverable failure ──
_FAILED = "__FAILED__"

# ═══════════════════════════════════════════════════════════════════════════
# Prompt engineering
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a recruitment data-extraction assistant. Given a candidate's career \
data you MUST reply with ONLY a valid JSON object — no markdown fences, no \
commentary, nothing else.
"""

_USER_TEMPLATE = """\
Candidate ID: {candidate_id}

Current title: {current_title}

Profile summary:
{profile_summary}

Career history (newest first):
{career_text}

---
Extract the following signals and return them as a single flat JSON object:

{{
  "candidate_id": "<same as above>",
  "has_production_retrieval": <true|false>,
  "is_pure_consulting": <true|false>,
  "title_desc_mismatch": <true|false>,
  "years_applied_ml": <integer>
}}

Definitions:
- has_production_retrieval: true if they BUILT or DEPLOYED a vector DB, RAG \
system, semantic search, or embedding-based retrieval to REAL users in production. \
Mentioning the technology in passing does NOT count.
- is_pure_consulting: true if their ENTIRE career (every role) is at services \
or outsourcing firms (TCS, Infosys, Wipro, Accenture, Cognizant, HCL, Capgemini, \
IBM GBS, Deloitte Consulting, etc.) with ZERO experience at product or research \
companies.
- title_desc_mismatch: true if their title contains "ML", "Machine Learning", \
"AI", or "Data Scientist" BUT the job descriptions clearly describe \
Sales, HR, Marketing, Recruitment, or Finance work with no technical content.
- years_applied_ml: integer estimate of years actually spent DOING machine \
learning (training models, feature engineering, MLOps, etc.). \
Roles clearly unrelated to ML should NOT be counted. \
Return 0 if no evidence of applied ML work.

Respond with ONLY the JSON object.
"""


def _build_prompt(candidate: dict) -> tuple[str, str]:
    """Return (system, user) prompt strings for one candidate."""
    cid     = candidate.get("candidate_id", "UNKNOWN")
    profile = candidate.get("profile") or {}
    title   = profile.get("current_title") or "N/A"
    summary = profile.get("summary") or profile.get("bio") or "N/A"

    career_history = candidate.get("career_history") or []
    lines: list[str] = []
    for role in career_history[:3]:           # top 3 recent roles (TPM-safe)
        company = role.get("company") or "Unknown"
        rtitle  = role.get("title") or "N/A"
        desc    = (role.get("description") or "")[:200]  # 200 char cap per-role
        lines.append(f"• {rtitle} @ {company}: {desc}")

    career_text = "\n".join(lines) if lines else "No career history available."

    user = _USER_TEMPLATE.format(
        candidate_id    = cid,
        current_title   = title,
        profile_summary = str(summary)[:200],
        career_text     = career_text,
    )
    return _SYSTEM_PROMPT, user


# ═══════════════════════════════════════════════════════════════════════════
# LLM provider back-ends
# ═══════════════════════════════════════════════════════════════════════════

class _ProviderConfig:
    """Thin struct holding everything needed to call one provider's chat API."""
    name:      str
    url:       str
    headers:   dict[str, str]
    model:     str
    is_ollama: bool = False
    is_bedrock: bool = False

    def __init__(
        self,
        name:      str,
        url:       str,
        headers:   dict[str, str],
        model:     str,
        is_ollama: bool = False,
        is_bedrock: bool = False,
    ) -> None:
        self.name      = name
        self.url       = url
        self.headers   = headers
        self.model     = model
        self.is_ollama = is_ollama
        self.is_bedrock = is_bedrock


def build_provider(provider: str, model: str | None) -> _ProviderConfig:
    """
    Build a provider config from env-vars.

    Raises RuntimeError with a helpful message if required env vars are missing.
    """
    provider = provider.lower()

    if provider == "bedrock":
        import boto3
        # Fast fail if credentials aren't accessible (e.g. implicitly via env)
        boto3.client('bedrock-runtime')
        return _ProviderConfig(
            name       = "bedrock",
            url        = "",
            headers    = {},
            model      = model or "mistral.mistral-large-2402-v1:0",
            is_bedrock = True,
        )

    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY env var is not set. "
                "Get a free key at https://console.groq.com"
            )
        chosen_model = model or "llama-3.1-8b-instant"
        return _ProviderConfig(
            name      = "groq",
            url       = "https://api.groq.com/openai/v1/chat/completions",
            headers   = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            model     = chosen_model,
        )

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY env var is not set."
            )
        return _ProviderConfig(
            name    = "openai",
            url     = "https://api.openai.com/v1/chat/completions",
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            model   = model or "gpt-4o-mini",
        )

    if provider == "ollama":
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return _ProviderConfig(
            name      = "ollama",
            url       = f"{base.rstrip('/')}/api/chat",
            headers   = {"Content-Type": "application/json"},
            model     = model or "llama3",
            is_ollama = True,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Choose: groq, openai, ollama, bedrock"
    )


def _build_payload(cfg: _ProviderConfig, system: str, user: str) -> dict:
    """Build the request body for the provider's chat endpoint."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    if cfg.is_ollama:
        return {
            "model":    cfg.model,
            "messages": messages,
            "stream":   False,
            "format":   "json",
            "options":  {"temperature": 0},
        }
    return {
        "model":       cfg.model,
        "messages":    messages,
        "temperature": 0,
        "max_tokens":  256,
        "response_format": {"type": "json_object"},
    }


def _extract_text(cfg: _ProviderConfig, response_json: dict) -> str:
    """Pull the assistant's text out of the provider response."""
    if cfg.is_ollama:
        return response_json["message"]["content"]
    return response_json["choices"][0]["message"]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# Core async API caller with exponential backoff
# ═══════════════════════════════════════════════════════════════════════════

async def _call_api(
    session:   aiohttp.ClientSession,
    cfg:       _ProviderConfig,
    candidate: dict,
    semaphore: asyncio.Semaphore,
    stats:     dict,
) -> dict | str:
    """
    Call the LLM API for one candidate.

    Returns a validated feature dict on success, or _FAILED on unrecoverable error.
    Implements exponential backoff with jitter for 429 / 5xx errors.
    """
    system, user = _build_prompt(candidate)
    cid          = candidate.get("candidate_id", "UNKNOWN")

    backoff = BASE_BACKOFF_S
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if cfg.is_bedrock:
                    import boto3
                    from botocore.exceptions import ClientError
                    def _do_bedrock():
                        client = boto3.client('bedrock-runtime')
                        return client.converse(
                            modelId=cfg.model,
                            messages=[{"role": "user", "content": [{"text": user}]}],
                            system=[{"text": system}],
                            inferenceConfig={"temperature": 0, "maxTokens": MAX_OUTPUT_TOKENS}
                        )
                    try:
                        resp = await asyncio.to_thread(_do_bedrock)
                        raw_text = resp['output']['message']['content'][0]['text']
                    except ClientError as exc:
                        status_code = exc.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 500)
                        error_code = exc.response.get('Error', {}).get('Code', 'Unknown')
                        if status_code == 429 or error_code == 'ThrottlingException':
                            wait = min(backoff + random.uniform(0, 1), MAX_BACKOFF_S)
                            log.warning("Rate limited (Bedrock) for %s. Waiting %.1fs (attempt %d/%d)", cid, wait, attempt, MAX_RETRIES)
                            stats["rate_limit_hits"] += 1
                            await asyncio.sleep(wait)
                            backoff = min(backoff * 2, MAX_BACKOFF_S)
                            continue
                        elif status_code >= 500:
                            wait = min(backoff + random.uniform(0, 1), MAX_BACKOFF_S)
                            log.warning("Server error %s for %s. Waiting %.1fs (attempt %d/%d)", status_code, cid, wait, attempt, MAX_RETRIES)
                            await asyncio.sleep(wait)
                            backoff = min(backoff * 2, MAX_BACKOFF_S)
                            continue
                        else:
                            log.error("Fatal Bedrock error for %s: %s", cid, exc)
                            stats["failures"] += 1
                            return _FAILED
                else:
                    payload = _build_payload(cfg, system, user)
                    async with session.post(
                        cfg.url,
                        headers = cfg.headers,
                        json    = payload,
                        timeout = timeout,
                    ) as resp:

                        # ── Rate limited ──────────────────────────────────────
                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", backoff))
                            wait = min(retry_after + random.uniform(0, 1), MAX_BACKOFF_S)
                            log.warning(
                                "Rate limited (429) for %s. Waiting %.1fs (attempt %d/%d)",
                                cid, wait, attempt, MAX_RETRIES
                            )
                            stats["rate_limit_hits"] += 1
                            await asyncio.sleep(wait)
                            backoff = min(backoff * 2, MAX_BACKOFF_S)
                            continue

                        # ── Server error ──────────────────────────────────────
                        if resp.status >= 500:
                            wait = min(backoff + random.uniform(0, 1), MAX_BACKOFF_S)
                            log.warning(
                                "Server error %d for %s. Waiting %.1fs (attempt %d/%d)",
                                resp.status, cid, wait, attempt, MAX_RETRIES
                            )
                            await asyncio.sleep(wait)
                            backoff = min(backoff * 2, MAX_BACKOFF_S)
                            continue

                        # ── Auth / bad-request errors — no point retrying ─────
                        if resp.status in (401, 403, 400):
                            body = await resp.text()
                            log.error(
                                "Fatal HTTP %d for %s: %s", resp.status, cid, body[:200]
                            )
                            stats["failures"] += 1
                            return _FAILED

                        resp.raise_for_status()
                        response_json = await resp.json(content_type=None)

            except asyncio.TimeoutError:
                wait = min(backoff + random.uniform(0, 1), MAX_BACKOFF_S)
                log.warning(
                    "Timeout for %s. Waiting %.1fs (attempt %d/%d)",
                    cid, wait, attempt, MAX_RETRIES
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                continue

            except aiohttp.ClientError as exc:
                wait = min(backoff + random.uniform(0, 1), MAX_BACKOFF_S)
                log.warning(
                    "Network error for %s: %s. Waiting %.1fs (attempt %d/%d)",
                    cid, exc, wait, attempt, MAX_RETRIES
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                continue

            # ── Parse the JSON response ───────────────────────────────────
            try:
                if not cfg.is_bedrock:
                    raw_text = _extract_text(cfg, response_json)
                features = _parse_and_validate(raw_text, cid)
                stats["successes"] += 1
                return features

            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                log.warning(
                    "Parse error for %s (attempt %d/%d): %s",
                    cid, attempt, MAX_RETRIES, exc
                )
                # Give the LLM another chance — maybe next attempt is cleaner.
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                await asyncio.sleep(min(backoff, 5.0))
                continue

    log.error("Exhausted retries for %s — recording as failed.", cid)
    stats["failures"] += 1
    return _FAILED


# ═══════════════════════════════════════════════════════════════════════════
# JSON validation / coercion
# ═══════════════════════════════════════════════════════════════════════════

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_and_validate(raw: str, candidate_id: str) -> dict:
    """
    Parse LLM output into a validated feature dict.

    Strips markdown fences if the model ignores the instruction.
    Coerces types and fills defaults for missing keys.
    Raises ValueError / JSONDecodeError on unrecoverable parse failure.
    """
    text = raw.strip()

    # Strip markdown code fences if present.
    fence_match = _JSON_FENCE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    data: dict[str, Any] = json.loads(text)

    # Type coercion helpers.
    def _bool(v: Any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else default

    def _int(v: Any, default: int = 0) -> int:
        if isinstance(v, int):
            return max(0, v)
        try:
            return max(0, int(float(str(v))))
        except (ValueError, TypeError):
            return default

    return {
        "candidate_id":            candidate_id,
        "has_production_retrieval": _bool(data.get("has_production_retrieval")),
        "is_pure_consulting":       _bool(data.get("is_pure_consulting")),
        "title_desc_mismatch":      _bool(data.get("title_desc_mismatch")),
        "years_applied_ml":         _int(data.get("years_applied_ml")),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint helpers (JSONL append — crash-safe)
# ═══════════════════════════════════════════════════════════════════════════

def _load_checkpoint(output_path: Path) -> set[str]:
    """
    Read an existing JSONL checkpoint and return the set of already-processed
    candidate_ids.  Safe to call even if the file does not exist.
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cid = record.get("candidate_id")
                if cid:
                    done.add(cid)
            except json.JSONDecodeError:
                pass
    log.info("Checkpoint: %d candidates already processed.", len(done))
    return done


def _flush_buffer(buffer: list[dict], output_path: Path) -> None:
    """Append buffered results to the JSONL checkpoint file."""
    if not buffer:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        for record in buffer:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Main async orchestrator
# ═══════════════════════════════════════════════════════════════════════════

async def _run_extraction(
    candidates:   list[dict],
    cfg:          _ProviderConfig,
    output_path:  Path,
    batch_size:   int,
    resume:       bool,
) -> None:
    """Process all candidates asynchronously with bounded concurrency."""

    # ── Resume: skip already-done candidates ──────────────────────────────
    done_ids: set[str] = set()
    if resume:
        done_ids = _load_checkpoint(output_path)
    elif output_path.exists():
        log.info(
            "Output file %s already exists. "
            "Use --resume to continue from checkpoint, or delete it to restart.",
            output_path,
        )
        # Non-resume run: overwrite.
        output_path.unlink()

    todo = [c for c in candidates if c.get("candidate_id") not in done_ids]
    log.info(
        "Total candidates: %d | Already done: %d | To process: %d",
        len(candidates), len(done_ids), len(todo),
    )
    if not todo:
        log.info("Nothing to do. All candidates already processed.")
        return

    semaphore = asyncio.Semaphore(batch_size)
    stats = {"successes": 0, "failures": 0, "rate_limit_hits": 0}

    # Write buffer — flushed every CHECKPOINT_EVERY completions.
    buffer:  list[dict] = []
    lock = asyncio.Lock()

    t_start  = time.perf_counter()
    n_done   = 0
    n_total  = len(todo)

    connector = aiohttp.TCPConnector(limit=batch_size * 2, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:

        tasks = [
            _call_api(session, cfg, cand, semaphore, stats)
            for cand in todo
        ]

        # Process as tasks complete (not in submission order) so we never
        # stall the checkpoint flush waiting for one slow request.
        for coro in asyncio.as_completed(tasks):
            result = await coro
            n_done += 1

            if result != _FAILED:
                async with lock:
                    buffer.append(result)
                    if len(buffer) >= CHECKPOINT_EVERY:
                        _flush_buffer(buffer, output_path)
                        log.info(
                            "Checkpoint: %d/%d (%.1f%%) | "
                            "rate_limit_hits=%d | failures=%d",
                            n_done, n_total, 100 * n_done / n_total,
                            stats["rate_limit_hits"], stats["failures"],
                        )

            # Progress report every 500 completions even without a flush.
            if n_done % 500 == 0:
                elapsed = time.perf_counter() - t_start
                rate    = n_done / elapsed if elapsed > 0 else 0
                eta_s   = (n_total - n_done) / rate if rate > 0 else 0
                log.info(
                    "Progress: %d/%d (%.1f%%) | %.1f req/s | ETA %.0fm %.0fs",
                    n_done, n_total, 100 * n_done / n_total,
                    rate, eta_s // 60, eta_s % 60,
                )

    # Flush any remaining results.
    _flush_buffer(buffer, output_path)

    elapsed = time.perf_counter() - t_start
    log.info(
        "\n══════════════════════════════════════════\n"
        " Extraction complete in %.0fm %.0fs\n"
        " Successes  : %d\n"
        " Failures   : %d\n"
        " Output     : %s\n"
        "══════════════════════════════════════════",
        elapsed // 60, elapsed % 60,
        stats["successes"], stats["failures"],
        output_path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline LLM feature extractor for the CandidateRanking pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidate data file (.json, .jsonl, .jsonl.gz)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--provider",
        default="bedrock",
        choices=["groq", "openai", "ollama", "bedrock"],
        help="LLM provider to use (default: bedrock)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name override. Defaults: groq=llama-3.1-8b-instant, "
            "openai=gpt-4o-mini, ollama=llama3, bedrock=mistral.mistral-large-2402-v1:0"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Max concurrent API calls (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint file (skip already-done IDs).",
    )
    parser.add_argument(
        "--top-ids",
        default=None,
        metavar="FILE",
        help=(
            "Path to a text file containing one candidate_id per line "
            "(produced by rank.py --top-ids-output). "
            "When provided, ONLY these candidates are sent to the LLM — "
            "all others are skipped. This is the recommended workflow: "
            "run rank.py first, then run this script with --top-ids."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Further limit to the first N candidates after --top-ids filtering (for testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load .env file FIRST so env vars are available when building the provider.
    load_dotenv()

    log.info("Provider: %s | Model: %s | Batch size: %d",
             args.provider, args.model or "default", args.batch_size)

    # Build provider config (validates env vars early).
    try:
        cfg = build_provider(args.provider, args.model)
    except (RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Loading candidates from %s ...", args.candidates)
    candidates = load_candidates(args.candidates)

    # ── Filter to top-IDs list if provided (the recommended workflow) ───────
    if args.top_ids:
        top_ids_path = Path(args.top_ids)
        if not top_ids_path.exists():
            log.error(
                "--top-ids file not found: '%s'. "
                "Run rank.py first to generate it.",
                args.top_ids,
            )
            sys.exit(1)

        with top_ids_path.open("r", encoding="utf-8") as fh:
            target_ids: set[str] = {line.strip() for line in fh if line.strip()}

        before = len(candidates)
        candidates = [c for c in candidates if c.get("candidate_id") in target_ids]
        log.info(
            "Filtered to top-IDs list: %d candidates selected from %d total "
            "(file: %s)",
            len(candidates), before, top_ids_path,
        )

        if not candidates:
            log.error(
                "No candidates matched the IDs in '%s'. "
                "Check that --candidates points to the same dataset used by rank.py.",
                args.top_ids,
            )
            sys.exit(1)
    else:
        log.warning(
            "No --top-ids file specified. Processing ALL %d candidates. "
            "This may take a very long time and use significant API quota. "
            "Recommended: run rank.py first, then pass --top-ids data/top_1000_ids.txt",
            len(candidates),
        )

    # ── Optional further slice for quick smoke tests ───────────────────────
    if args.sample:
        candidates = candidates[: args.sample]
        log.info("[SAMPLE MODE] Further limited to first %d candidates.", len(candidates))

    output_path = Path(args.output)
    asyncio.run(
        _run_extraction(
            candidates  = candidates,
            cfg         = cfg,
            output_path = output_path,
            batch_size  = args.batch_size,
            resume      = args.resume,
        )
    )


if __name__ == "__main__":
    main()
