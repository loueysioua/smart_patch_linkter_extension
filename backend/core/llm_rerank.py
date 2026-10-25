"""
LLM reranking of patch candidates — pure logic, no heavy imports.

Deliberately dependency-free (stdlib only) so the serving pipeline can import
it without pulling in ``core.engine`` / ``core.gerrit`` or an ``openai``
install. The caller passes an already-constructed client; this module never
imports a provider SDK.

Shared by:
  * ``scripts/llm_rerank.py``        (standalone CLI, OpenAI client)
  * ``scripts/ml_to_rag_pipeline.py`` (optional Stage 3)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ``backend/.env`` relative to this file -- core/ -> backend/
_DOTENV_PATHS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"),
    os.path.join(os.getcwd(), ".env"),
)


def _load_dotenv() -> None:
    """Best-effort .env load. Absent python-dotenv is not an error."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # CWD search covers the normal case (run from backend/); the explicit paths
    # cover being invoked from elsewhere. Existing env vars are never overwritten.
    load_dotenv()
    for path in _DOTENV_PATHS:
        if os.path.exists(path):
            load_dotenv(path)

# Descriptions in this dataset run long (mean description is ~440 chars, but
# change_log blobs are far bigger). Untruncated, 10 candidates can blow past a
# model's context window and burn tokens on boilerplate, so every text field is
# clipped before it reaches the prompt.
MAX_TITLE_CHARS = 300
MAX_DESCRIPTION_CHARS = 1500
MAX_FILES_LISTED = 25


def _strip_markdown_fences(text: str) -> str:
    """Remove optional ```json / ``` fences from *text*."""
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        return "n/a"
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " [...]"


def _files_summary(files: Any, limit: int = MAX_FILES_LISTED) -> str:
    if not files:
        return "n/a"
    files = list(files)
    shown = ", ".join(str(f) for f in files[:limit])
    if len(files) > limit:
        shown += f", ... (+{len(files) - limit} more)"
    return shown


def build_rerank_prompt(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_description_chars: int = MAX_DESCRIPTION_CHARS,
    max_title_chars: int = MAX_TITLE_CHARS,
    max_files: int = MAX_FILES_LISTED,
) -> str:
    """Build the reranking prompt. Exposed separately so it can be inspected
    (and length-tested) without spending an API call.

    The three limits are context-window bounds applied at *send* time. Stored
    text is never trimmed -- ``scripts/hydrate_candidates.py`` keeps raw values
    so raising a limit never requires re-hydrating.
    """
    candidate_block = "\n".join(
        f"""
--- Candidate {i + 1} ---
- **ID:** {c.get('patch_id')}
- **Title:** {_clip(c.get('title'), max_title_chars)}
- **Files:** {_files_summary(c.get('files'), max_files)}
- **Description:**
{_clip(c.get('description'), max_description_chars)}
"""
        for i, c in enumerate(candidates)
    )

    return f"""You are an expert software developer. Your task is to rerank \
a list of candidate code changes (patches) based on their relevance to a target patch.
The candidates are currently ranked by an ML + RAG pipeline. \
Re-evaluate and rank them from most to least relevant.

Judge relevance by whether the patches plausibly address the same underlying \
problem: shared intent, shared ticket or bug reference, or changes to the same \
subsystem. Prefer substantive links over superficial keyword overlap.

### Target Patch
- **ID:** {target.get('patch_id')}
- **Title:** {_clip(target.get('title'), max_title_chars)}
- **Files:** {_files_summary(target.get('files'), max_files)}
- **Description:**
{_clip(target.get('description'), max_description_chars)}

### Candidate Patches to Rerank:
{candidate_block}
---
Output ONLY a valid JSON array of the candidate IDs ordered from most to least \
relevant. Include every candidate exactly once.
Example: ["id_2", "id_1", "id_3"]
Do NOT include any explanation, markdown, or extra text.
"""


def apply_rerank_order(
    reranked_ids: list[str],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reorder *candidates* to match *reranked_ids*.

    Guarantees the output is a permutation of the input: unknown ids are
    dropped and any candidate the model forgot is appended in its original
    relative order. This matters because Recall@top_k is invariant under a
    permutation -- the LLM can improve ordering, never the retrieved set.
    """
    id_to_cand = {str(c["patch_id"]): c for c in candidates}
    by_id: dict[str, dict[str, Any]] = {}
    for c in candidates:
        by_id.setdefault(str(c["patch_id"]), c)

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pid in reranked_ids:
        key = str(pid)
        if key in by_id and key not in seen:
            ordered.append(by_id[key])
            seen.add(key)
    for c in candidates:                      # append the forgotten ones
        key = str(c["patch_id"])
        if key not in seen:
            ordered.append(c)
            seen.add(key)
    return ordered


def parse_rerank_response(
    response_text: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Parse an LLM reply into a reordered candidate list.

    Returns ``(reranked, None)`` on success or ``(None, reason)`` if the reply
    was not a usable JSON array of known ids. Callers should keep their
    original ranking on failure.
    """
    try:
        parsed = json.loads(_strip_markdown_fences(response_text))
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"not valid JSON ({exc})"

    if not isinstance(parsed, list):
        return None, f"expected a JSON array, got {type(parsed).__name__}"
    if not all(isinstance(x, str) for x in parsed):
        return None, "array contained non-string entries"

    known = {str(c["patch_id"]) for c in candidates}
    unknown = [x for x in parsed if x not in known]
    if unknown:
        logger.debug("LLM returned %d unknown id(s), ignored: %s", len(unknown), unknown[:5])

    return apply_rerank_order(parsed, candidates), None


def rerank_candidates(
    client,
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    model_name: str = "gpt-4o-mini",
    max_description_chars: int = MAX_DESCRIPTION_CHARS,
    max_title_chars: int = MAX_TITLE_CHARS,
    max_files: int = MAX_FILES_LISTED,
    timeout: float | None = 60.0,
) -> list[dict[str, Any]]:
    """Ask a chat model to rerank *candidates* by relevance to *target*.

    Any failure -- transport error, unparseable reply, unknown ids -- returns
    the input ranking unchanged, so an LLM outage degrades to the ML+RAG
    result rather than losing it.

    Args:
        client: Any object exposing ``client.chat.completions.create``.
        target: Target patch dict.
        candidates: Candidate patch dicts (must carry ``patch_id``).
        model_name: Provider model identifier.
        max_description_chars: Per-field clip length for the prompt.
        max_title_chars: Title clip length.
        max_files: Maximum file paths listed per patch.
        timeout: Per-request timeout in seconds.

    Returns:
        Reranked list (a permutation of *candidates*).
    """
    if not candidates:
        return []

    prompt = build_rerank_prompt(
        target, candidates,
        max_description_chars=max_description_chars,
        max_title_chars=max_title_chars,
        max_files=max_files,
    )
    logger.debug("Sending %d candidates to %s for reranking…", len(candidates), model_name)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        response = client.chat.completions.create(**kwargs)
        response_text: str = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - never let the LLM break the pipeline
        logger.warning("LLM call failed (%s) - keeping the ML+RAG ranking.", exc)
        return candidates

    reranked, reason = parse_rerank_response(response_text, candidates)
    if reranked is None:
        logger.warning("Could not use LLM response (%s) - keeping the ML+RAG ranking.", reason)
        logger.debug("Raw LLM response: %s", response_text)
        return candidates

    logger.debug("Reranked %d candidates via %s.", len(reranked), model_name)
    return reranked


def build_openai_client(base_url: str | None = None, timeout: float | None = 60.0):
    """Construct an OpenAI-compatible client.

    ``openai`` is imported lazily inside this function, so importing this module
    stays dependency-free (see the module docstring) and
    ``scripts/llm_rerank_stage.py`` can use it without pulling in
    sentence-transformers / faiss / joblib.

    Args:
        base_url: Any OpenAI-compatible endpoint -- Ollama, llama.cpp, vLLM,
            LM Studio. When set, no API key is required.
        timeout: Default per-request timeout in seconds.

    Returns:
        An ``openai.OpenAI`` instance.

    Raises:
        RuntimeError: if the SDK is missing, or no key/base_url is available.
    """
    import os

    # Load .env before reading the environment. The old scripts/llm_rerank.py did
    # this; the split stages did not, so a key present in backend/.env was
    # invisible and the stage failed with "needs OPENAI_API_KEY". Real env vars
    # still win -- load_dotenv does not overwrite them.
    _load_dotenv()

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "LLM rerank needs the 'openai' package: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not base_url:
        raise RuntimeError(
            "LLM rerank needs OPENAI_API_KEY (checked the environment and "
            f"{_DOTENV_PATHS}), or --llm-base-url for a local endpoint."
        )

    kwargs: dict[str, Any] = {"api_key": api_key or "not-needed"}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)
