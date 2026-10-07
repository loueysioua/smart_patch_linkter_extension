"""
Stage 1.5: hydrate a candidate hand-off with the text an LLM actually needs.

Stage 1 (``ml_to_rag_pipeline.py --candidates-out FILE``) writes one JSON line
per query containing ranked candidate **ids** only. Stage 2
(``llm_rerank_stage.py``) needs the title, description and file list for those
patches -- without them the LLM receives a list of numbers and an instruction to
rank them, and returns an arbitrary permutation that still looks like a valid
result.

This stage joins the ids against the patch CSV and writes a *deduplicated*
sidecar: one line per unique patch, text stored once even when the patch is a
candidate in hundreds of queries. Stage 2 then fetches the ~11 patches it needs
for the current query, builds one prompt, sends it, and discards it. Nothing
large circulates and no hydrated corpus is ever materialised.

Design notes:
  * **Untrimmed by default.** Title/description/files are stored raw. Truncation
    is a context-window constraint, not a storage one, so it stays in
    ``core.llm_rerank.build_rerank_prompt`` where the caps are visible and
    configurable -- and so both the fused and split paths clip identically.
  * **Chunked read.** Only ``usecols`` are read, in chunks, retaining just the
    rows whose patch_id is needed, so peak memory is bounded by chunk size
    rather than dataset size.
  * **~0.4 GB, ~15 s** on the 102k-row onap dataset, versus ~6.5 GB for Stage 1.
    The difference is ``change_log``: 896 MB of JSON this stage never touches.

Usage::

    python scripts/hydrate_candidates.py \\
        --candidates /tmp/stage1.jsonl \\
        --out /tmp/texts.jsonl \\
        --csv data/onap/all_candidates.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, BACKEND)

# Kept in sync with core.llm_rerank; imported so a change there is picked up.
from core.llm_rerank import (  # noqa: E402
    MAX_DESCRIPTION_CHARS,
    MAX_FILES_LISTED,
    MAX_TITLE_CHARS,
)

TEXT_COLUMNS = ("patch_id", "title", "description", "files")
DEFAULT_CHUNK = 20_000


def collect_needed_ids(candidates_path: str) -> tuple[set[str], int]:
    """Patch ids referenced anywhere in the hand-off (targets + candidates)."""
    needed: set[str] = set()
    rows = 0
    with open(candidates_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            src = row.get("source_patch_id")
            if src is not None:
                needed.add(str(src))
            for pid in row.get("candidates", []) or []:
                # hand-off may hold bare ids (current format) or objects
                needed.add(str(pid["patch_id"] if isinstance(pid, dict) else pid))
    return needed, rows


def parse_files(value) -> list[str]:
    """``files`` is a stringified Python list in the CSV."""
    if isinstance(value, list):
        return [str(f) for f in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [str(f) for f in parsed] if isinstance(parsed, list) else []


def load_texts(csv_path: str, needed: set[str], chunk_size: int = DEFAULT_CHUNK) -> dict[str, dict]:
    """Read only the needed rows, in chunks, keeping text for ``needed`` ids."""
    import pandas as pd

    found: dict[str, dict] = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=list(TEXT_COLUMNS),
        dtype=str,
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk = chunk[chunk["patch_id"].isin(needed)]
        if chunk.empty:
            continue
        for pid, title, desc, files in chunk.itertuples(index=False, name=None):
            found[str(pid)] = {
                "title": "" if title is None or pd.isna(title) else str(title),
                "description": "" if desc is None or pd.isna(desc) else str(desc),
                "files": parse_files(files),
            }
    return found


def count_oversized(texts: dict[str, dict]) -> dict[str, int]:
    """How many patches exceed the prompt caps -- visibility, not silent loss."""
    over_desc = sum(1 for t in texts.values() if len(t["description"]) > MAX_DESCRIPTION_CHARS)
    over_title = sum(1 for t in texts.values() if len(t["title"]) > MAX_TITLE_CHARS)
    over_files = sum(1 for t in texts.values() if len(t["files"]) > MAX_FILES_LISTED)
    return {"description": over_desc, "title": over_title, "files": over_files}


def hydrate(
    candidates_path: str,
    out_path: str,
    csv_path: str,
    chunk_size: int = DEFAULT_CHUNK,
) -> dict:
    """Write a deduplicated ``patch_id -> text`` sidecar for the hand-off."""
    if not os.path.exists(candidates_path):
        raise SystemExit(f"candidate file not found: {candidates_path}")
    if not os.path.exists(csv_path):
        raise SystemExit(f"patch csv not found: {csv_path}")

    started = time.perf_counter()
    needed, n_rows = collect_needed_ids(candidates_path)
    print(f"hand-off      : {n_rows} queries, {len(needed)} unique patch ids needed")

    texts = load_texts(csv_path, needed, chunk_size)
    missing = needed - set(texts)
    print(f"csv           : {len(texts)}/{len(needed)} ids resolved"
          + (f"  ({len(missing)} MISSING)" if missing else ""))
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        print(f"   missing ids will be sent to the LLM with no text: {sample}")

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for pid in sorted(texts):
            fh.write(json.dumps({"patch_id": pid, **texts[pid]}) + "\n")
            written += 1

    over = count_oversized(texts)
    total = max(len(texts), 1)
    print(f"\nsidecar       : {out_path}")
    print(f"   {written} patches, {os.path.getsize(out_path)/1024**2:.1f} MB "
          f"({written*1024//total} bytes/patch avg, stored once each)")
    print("\npatches exceeding the prompt caps (clipped at send time, not stored trimmed):")
    print(f"   description > {MAX_DESCRIPTION_CHARS:>5} chars : {over['description']:>6} "
          f"({100*over['description']/total:.1f}%)")
    print(f"   title       > {MAX_TITLE_CHARS:>5} chars : {over['title']:>6} "
          f"({100*over['title']/total:.1f}%)")
    print(f"   files       > {MAX_FILES_LISTED:>5}       : {over['files']:>6} "
          f"({100*over['files']/total:.1f}%)")
    print("\n   raise a cap with --max-desc-chars / --max-title-chars / --max-files on")
    print("   llm_rerank_stage.py if these counts are high enough to matter.")

    print(f"\nelapsed       : {time.perf_counter()-started:.1f} s")
    return {
        "queries": n_rows,
        "ids_needed": len(needed),
        "ids_resolved": len(texts),
        "ids_missing": len(missing),
        "written": written,
        "oversized": over,
        "elapsed_sec": time.perf_counter() - started,
        "out": out_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage 1.5: hydrate a candidate hand-off with patch text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--candidates", required=True, help="JSONL from --candidates-out")
    p.add_argument("--out", required=True, help="Deduplicated text sidecar (JSONL)")
    p.add_argument("--csv", required=True, help="Patch CSV (all_candidates.csv)")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK,
                   help=f"CSV rows per chunk (default {DEFAULT_CHUNK}); bounds peak memory")
    args = p.parse_args()

    hydrate(args.candidates, args.out, args.csv, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
