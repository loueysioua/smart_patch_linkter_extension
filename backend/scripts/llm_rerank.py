"""
LLM Reranker for SmartPatch — OpenAI-powered post-retrieval ranking.

Retrieves top-k candidate patches with the RAG engine, then asks an OpenAI
chat model to rerank them by true semantic relevance to the target patch.

Usage::

    # From the backend/ directory:
    python scripts/llm_rerank.py \\
        --project onap \\
        --csv datasets/onap/all_candidates.csv \\
        --target-id <patch_id> \\
        --top-k 5 \\
        --llm-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

# Allow running this script from the backend/ directory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

from core.engine import SmartPatchEngine
from core.gerrit import GerritClient
from core.llm_rerank import rerank_candidates  # shared with the serving pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_patch_details(
    engine: SmartPatchEngine, project: str, patch_id: str
) -> dict[str, Any] | None:
    """Return patch metadata from the loaded dataset or the Gerrit API.

    Args:
        engine: Loaded :class:`~core.engine.SmartPatchEngine` instance.
        project: Project key.
        patch_id: Target patch identifier.

    Returns:
        Patch dict or ``None`` if the patch cannot be found.
    """
    df = engine.datasets.get(project)
    if df is not None:
        existing = df[df.patch_id == patch_id]
        if not existing.empty:
            row = existing.iloc[0]
            return {
                "patch_id": row.patch_id,
                "title": row.title,
                "description": row.description,
                "created_time": row.created_time,
                "files": engine._safe_parse_list(row.files),
            }

    logger.info("Patch %s not in dataset — fetching from Gerrit API.", patch_id)
    return GerritClient.get_patch_details(project, patch_id)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank top-k RAG candidates using an OpenAI chat model."
    )
    parser.add_argument("--project", required=True, help="Project key (e.g. onap)")
    parser.add_argument("--csv", required=True, help="Path to all_candidates.csv")
    parser.add_argument("--target-id", required=True, help="Patch ID of the target query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to rerank (default: 5)")
    parser.add_argument("--window-days", type=int, default=14, help="Time window in days (default: 14)")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    engine = SmartPatchEngine()
    engine.load_project(args.project, args.csv)

    logger.info("Fetching details for target patch %s…", args.target_id)
    target_patch = get_patch_details(engine, args.project, args.target_id)
    if not target_patch:
        logger.error("Could not find target patch %s.", args.target_id)
        sys.exit(1)

    logger.info("Running RAG prediction (top_k=%d)…", args.top_k)
    initial_results = engine.predict(
        args.project, target_patch, top_k=args.top_k, window_days=args.window_days
    )

    if not initial_results:
        logger.info("No candidates found in the time window.")
        sys.exit(0)

    # Enrich with description for the LLM (candidates may lack it)
    df = engine.datasets.get(args.project)
    enhanced: list[dict[str, Any]] = []
    for res in initial_results:
        cand_id = res["patch_id"]
        description = "No description available"
        if df is not None:
            cand_row = df[df.patch_id == cand_id]
            if not cand_row.empty:
                description = str(cand_row.iloc[0].description)
        enhanced.append({**res, "description": description})

    print("\n--- Original RAG Ranking ---")
    for i, c in enumerate(enhanced):
        print(f"{i + 1}. [{c['patch_id']}] (Score: {c['score']:.3f}) {c['title']}")

    reranked = rerank_candidates(client, target_patch, enhanced, model_name=args.llm_model)

    print("\n--- LLM Reranked Results ---")
    for i, c in enumerate(reranked):
        print(f"{i + 1}. [{c['patch_id']}] (RAG Score: {c['score']:.3f}) {c['title']}")


if __name__ == "__main__":
    main()
