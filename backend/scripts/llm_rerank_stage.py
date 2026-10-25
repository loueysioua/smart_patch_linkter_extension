"""
Stage 2 of the decoupled LLM rerank: apply an LLM to a candidate hand-off.

Stage 1 (``scripts/ml_to_rag_pipeline.py --candidates-out FILE``) does the
memory-heavy work -- loads the model, the dataset and the embeddings, scores
every candidate -- and writes one JSON line per query with the ranked candidate
ids and that query's reachable targets.

This script replays that file against an LLM and recomputes the metrics. It is
deliberately **independent**: it imports nothing from the retrieval pipeline, no
sentence-transformers, no faiss, no joblib. Peak RSS is a few tens of MB rather
than several GB, so the stages never compete for RAM, and this one can run on a
different machine entirely.

Because MRR and Recall@k are pure functions of (ordered id list, target set),
the metrics here are identical to what the fused ``--llm-rerank`` path would
produce. ``scripts/check_stage_parity.py`` asserts exactly that.

**--texts is required.** Stage 1 writes candidate *ids* only, and an LLM cannot
rank a list of numbers. ``--texts`` points at the deduplicated sidecar built by
``scripts/hydrate_candidates.py``; this script then fetches the ~11 patches the
current query needs, builds one prompt, sends it and discards it. Nothing large
circulates and no hydrated corpus is materialised.

Usage::

    # Stage 1 (heavy, ~6.5 GB)
    python scripts/ml_to_rag_pipeline.py --evaluate \\
        --candidates-out /tmp/stage1.jsonl --details-out /tmp/stage1_metrics.csv

    # Stage 1.5 (~0.4 GB): join ids to patch text, stored once per patch
    python scripts/hydrate_candidates.py \\
        --candidates /tmp/stage1.jsonl --out /tmp/texts.jsonl \\
        --csv data/onap/all_candidates.csv

    # Stage 2 (~25 MB + sidecar): rerank
    python scripts/llm_rerank_stage.py \\
        --candidates /tmp/stage1.jsonl --texts /tmp/texts.jsonl \\
        --out /tmp/stage2_metrics.csv --reranked-out /tmp/reranked.jsonl \\
        --llm-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.llm_rerank import (  # noqa: E402
    MAX_DESCRIPTION_CHARS,
    MAX_FILES_LISTED,
    MAX_TITLE_CHARS,
    build_openai_client,
    rerank_candidates,
)

logger = logging.getLogger("llm_rerank_stage")

DEFAULT_KS = (1, 2, 4, 6, 8, 10)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def hhmmss(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_lat(seconds: float) -> str:
    """Sub-second calls need ms; averaging 0.00s hides real regressions."""
    if seconds < 10:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def count_lines(path: str) -> int:
    """Total query rows, for the progress denominator."""
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def progress_line(stats, reciprocal_ranks, started) -> str:
    """One compact line carrying what you need to judge the run's health.

    Fields chosen deliberately: progress, whether quality is moving, throughput,
    ETA, cache/failure counts, and mean latency -- the last being what tells you
    whether the remaining requests still fit in the session.
    """
    done = stats["queries"]
    total = stats["n_lines"] or done
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    pct = 100.0 * done / total if total else 100.0
    mrr = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mean_lat = (stats["latency_sum"] / stats["api_calls"]) if stats["api_calls"] else 0.0
    eta_s = hhmmss(eta) if eta == eta else "?:??"
    return (f"[{done:>5}/{total}] {pct:5.1f}% | MRR {mrr:.4f} | {rate:5.2f}/s "
            f"| eta {eta_s:<6} | ok {stats['api_calls'] - stats['failures']:<5} "
            f"cache {stats['cache_hits']:<5} fail {stats['failures']:<3} "
            f"| avg {fmt_lat(mean_lat)}")


def cache_key(target_id: str, head: list[str], model: str, llm_top_k: int) -> str:
    payload = json.dumps(
        {"model": model, "llm_top_k": llm_top_k, "target": str(target_id), "ids": head},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def load_cache(path: str | None) -> dict[str, list[str]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read cache %s (%s) - starting empty", path, exc)
        return {}


def save_cache(path: str | None, cache: dict) -> None:
    if not path or not cache:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)


def score(ranked: list[str], targets: set[str], ks) -> tuple[float, dict[int, float]]:
    """MRR + Recall@k from an ordered id list. Matches evaluate_pipeline()."""
    rr = 0.0
    for rank, pid in enumerate(ranked, 1):
        if pid in targets:
            rr = 1.0 / rank
            break
    recalls = {}
    for k in ks:
        recalls[k] = len(set(ranked[:k]) & targets) / len(targets) if targets else 0.0
    return rr, recalls


def apply_order(order: list[str], head: list[str], tail: list[str]) -> list[str]:
    """Rebuild the full ranking: reranked head, then the untouched tail."""
    by_id = {}
    for pid in head:
        by_id.setdefault(str(pid), str(pid))
    out, seen = [], set()
    for pid in order:
        key = str(pid)
        if key in by_id and key not in seen:
            out.append(key)
            seen.add(key)
    for pid in head:                      # forgotten by the model
        if str(pid) not in seen:
            out.append(str(pid))
            seen.add(str(pid))
    out.extend(str(p) for p in tail)
    return out


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def load_texts(texts_path: str) -> dict[str, dict]:
    """Load the deduplicated patch-text sidecar into a lookup."""
    if not os.path.exists(texts_path):
        raise SystemExit(
            f"text sidecar not found: {texts_path}\n"
            "Run hydrate_candidates.py (Stage 1.5) to build it."
        )
    out: dict[str, dict] = {}
    with open(texts_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["patch_id"])] = {
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "files": row.get("files", []),
            }
    return out


def assemble(row: dict, texts: dict[str, dict]) -> tuple[dict, list[dict], int]:
    """Build the target and candidate payloads for ONE query.

    Stage 1 writes bare id strings; this joins them against the sidecar and
    returns candidates carrying real text, fetched for this query only. Objects
    are also accepted so an already-hydrated hand-off works unchanged.

    Returns ``(target, candidates, n_missing)``. Raises when *nothing* resolved,
    which means the wrong or empty sidecar was supplied -- proceeding would send
    the LLM a list of numbers and quietly produce an arbitrary permutation.
    """
    src = str(row.get("source_patch_id"))
    t_text = texts.get(src)
    target = {
        "patch_id": src,
        "title": (t_text or {}).get("title", ""),
        "description": (t_text or {}).get("description", ""),
        "files": (t_text or {}).get("files", []),
    }

    candidates, missing = [], 0
    for item in row.get("candidates", []) or []:
        pid = str(item["patch_id"] if isinstance(item, dict) else item)
        c_text = texts.get(pid)
        if c_text is None:
            missing += 1
            candidates.append({"patch_id": pid, "title": "",
                               "description": "", "files": []})
        else:
            candidates.append({"patch_id": pid, **c_text})

    resolved = sum(1 for c in candidates if c["title"] or c["description"])
    if candidates and resolved == 0:
        raise SystemExit(
            f"none of the {len(candidates)} candidate ids resolved against "
            f"{len(texts)} entries in the text sidecar. The LLM would receive ids "
            "only and return an arbitrary ranking. Rebuild the sidecar with "
            "scripts/hydrate_candidates.py for this candidate file."
        )
    return target, candidates, missing


def rerank_file(
    candidates_path: str,
    texts_path: str,
    out_path: str | None = None,
    reranked_out: str | None = None,
    client=None,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    llm_top_k: int = 10,
    cache_path: str | None = None,
    limit: int = 0,
    ks=DEFAULT_KS,
    timeout: float = 60.0,
    progress_every: int = 50,
    slow_call_seconds: float = 20.0,
    max_desc_chars: int = MAX_DESCRIPTION_CHARS,
    max_title_chars: int = MAX_TITLE_CHARS,
    max_files: int = MAX_FILES_LISTED,
) -> dict:
    """Rerank every query in *candidates_path* and write the metrics.

    Streams the file one line at a time, so memory stays flat regardless of how
    many queries there are. Both outputs are written incrementally and flushed,
    so an interrupted run keeps everything completed so far.

    Args:
        candidates_path: JSONL produced by ``--candidates-out`` (ids only).
        texts_path: Deduplicated text sidecar from ``hydrate_candidates.py``.
        out_path: CSV path for per-query metrics.
        reranked_out: JSONL path recording what the LLM actually did.
        client: Pre-built OpenAI-compatible client (tests inject a fake).
        model: Model id passed to the provider.
        base_url: OpenAI-compatible endpoint. Built into *client* if absent.
        llm_top_k: How many top candidates the LLM sees. 0 = all.
        cache_path: JSON file caching orderings, keyed by (target, head, model).
        limit: Stop after N queries. 0 = all.
        ks: k values for Recall@k.
        timeout: Per-request timeout in seconds.
        progress_every: Queries between progress lines. 0 disables them.
        slow_call_seconds: Per-call latency above which a warning is logged.
        max_desc_chars: Description clip applied at send time (not in storage).
        max_title_chars: Title clip applied at send time.
        max_files: Maximum file paths sent per patch.

    Returns:
        Summary dict: counts, MRR, mean Recall@k, and LLM call statistics.
    """
    if not os.path.exists(candidates_path):
        raise SystemExit(f"candidate file not found: {candidates_path}")

    texts = load_texts(texts_path)
    logger.info("loaded text for %d patches from %s", len(texts), texts_path)

    cache = load_cache(cache_path)
    if cache:
        logger.info("loaded %d cached orderings from %s", len(cache), cache_path)

    if client is None:
        try:
            client = build_openai_client(base_url=base_url, timeout=timeout)
        except RuntimeError as exc:
            raise SystemExit(f"cannot build LLM client: {exc}")

    stats = {"queries": 0, "api_calls": 0, "cache_hits": 0, "failures": 0,
             "reordered": 0, "missing_text": 0, "slow_calls": 0,
             "latency_sum": 0.0, "n_lines": 0}
    reciprocal_ranks: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    # quality of the incoming ML+RAG order, so the summary can state whether the
    # LLM helped instead of only reporting the post-LLM number
    base_rr: list[float] = []
    base_recalls: dict[int, list[float]] = {k: [] for k in ks}

    out_dir = os.path.dirname(out_path) if out_path else ""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rr_dir = os.path.dirname(reranked_out) if reranked_out else ""
    if rr_dir:
        os.makedirs(rr_dir, exist_ok=True)

    metrics_fh = open(out_path, "w", newline="", encoding="utf-8") if out_path else None
    reranked_fh = open(reranked_out, "w", encoding="utf-8") if reranked_out else None
    writer = None
    if metrics_fh:
        writer = csv.DictWriter(
            metrics_fh,
            fieldnames=["source_patch_id", "n_targets_reachable", "reciprocal_rank",
                        "n_returned", "llm_reordered", "from_cache"]
                        + [f"recall_at_{k}" for k in ks],
        )
        writer.writeheader()

    started = time.perf_counter()
    try:
        with open(candidates_path, encoding="utf-8") as src:
            for line_no, line in enumerate(src, 1):
                if limit and stats["queries"] >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                if not stats["n_lines"]:
                    stats["n_lines"] = count_lines(candidates_path)
                row = json.loads(line)

                target, cands, n_missing = assemble(row, texts)
                stats["missing_text"] += n_missing
                candidates = [c["patch_id"] for c in cands]
                targets = {str(t) for t in row.get("reachable_targets", [])}
                if not candidates:
                    continue

                head_n = len(candidates) if llm_top_k <= 0 else min(llm_top_k, len(candidates))
                head, tail = candidates[:head_n], candidates[head_n:]

                key = cache_key(row.get("source_patch_id", ""), head, model, llm_top_k)
                from_cache = key in cache
                if from_cache:
                    stats["cache_hits"] += 1
                    final = apply_order(cache[key], head, tail)
                else:
                    payload = cands[:head_n]
                    before_n = len(head)
                    t_req = time.perf_counter()
                    try:
                        reordered = rerank_candidates(
                            client, target, payload,
                            model_name=model, timeout=timeout,
                            max_description_chars=max_desc_chars,
                            max_title_chars=max_title_chars,
                            max_files=max_files,
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats["failures"] += 1
                        logger.warning("query %s failed (%s) - keeping input order",
                                       row.get("source_patch_id"), exc)
                        reordered = payload
                    took = time.perf_counter() - t_req
                    stats["api_calls"] += 1
                    stats["latency_sum"] += took
                    if took > slow_call_seconds:
                        stats["slow_calls"] += 1
                        # Rate-limited on purpose: a systematically slow endpoint
                        # would emit one warning per query -- exactly the flood
                        # this logging rework removed. The total still surfaces.
                        if stats["slow_calls"] <= 5:
                            logger.warning("slow LLM call %s on %s (> %s)",
                                           fmt_lat(took), row.get("source_patch_id"),
                                           fmt_lat(slow_call_seconds))
                        elif stats["slow_calls"] == 6:
                            logger.warning("further slow-call warnings suppressed; "
                                           "total is in the summary")
                    order = [str(c["patch_id"]) for c in reordered]
                    final = apply_order(order, head, tail)
                    changed = order != head
                    stats["reordered"] += int(changed)
                    cache[key] = order
                    if len(order) != before_n:
                        logger.warning("query %s: LLM returned %d ids for %d candidates",
                                       row.get("source_patch_id"), len(order), before_n)

                rr, rec = score(final, targets, ks)
                reciprocal_ranks.append(rr)
                for k in ks:
                    recalls[k].append(rec[k])
                b_rr, b_rec = score(candidates, targets, ks)
                base_rr.append(b_rr)
                for k in ks:
                    base_recalls[k].append(b_rec[k])
                stats["queries"] += 1

                if writer:
                    writer.writerow({
                        "source_patch_id": row.get("source_patch_id"),
                        "n_targets_reachable": len(targets),
                        "reciprocal_rank": rr,
                        "n_returned": len(final),
                        "llm_reordered": int(final != candidates),
                        "from_cache": int(from_cache),
                        **{f"recall_at_{k}": rec[k] for k in ks},
                    })
                    metrics_fh.flush()

                if reranked_fh:
                    reranked_fh.write(json.dumps({
                        "source_patch_id": row.get("source_patch_id"),
                        "input_order": candidates,
                        "output_order": final,
                        "changed": final != candidates,
                        "from_cache": from_cache,
                        "missing_text": n_missing,
                        "target_title": target.get("title", "")[:120],
                        "reachable_targets": sorted(targets),
                    }) + "\n")
                    reranked_fh.flush()

                if progress_every and stats["queries"] % progress_every == 0:
                    save_cache(cache_path, cache)
                    print("   " + progress_line(stats, reciprocal_ranks, started),
                          flush=True)
    finally:
        save_cache(cache_path, cache)
        if metrics_fh:
            metrics_fh.close()
        if reranked_fh:
            reranked_fh.close()

    elapsed = time.perf_counter() - started
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_recalls = {
        k: (sum(v) / len(v) if v else 0.0) for k, v in recalls.items()
    }

    base_mrr = (sum(base_rr) / len(base_rr)) if base_rr else 0.0
    base_means = {k: (sum(v) / len(v) if v else 0.0) for k, v in base_recalls.items()}
    mean_lat = (stats["latency_sum"] / stats["api_calls"]) if stats["api_calls"] else 0.0

    print("\n" + "=" * 66)
    print("STAGE 2 RESULTS (LLM rerank)")
    print("=" * 66)
    print(f"   queries evaluated : {stats['queries']}/{stats['n_lines'] or stats['queries']}")
    print(f"   elapsed           : {hhmmss(elapsed)}   "
          f"mean latency {fmt_lat(mean_lat)}   "
          f"{stats['queries']/elapsed if elapsed else 0:.2f} req/s")
    if stats["slow_calls"]:
        print(f"   slow calls        : {stats['slow_calls']} over {fmt_lat(slow_call_seconds)}")

    print("\n   metric        ML+RAG      +LLM       delta")
    print(f"   {'MRR':<12} {base_mrr:8.4f}  {mrr:8.4f}  {mrr - base_mrr:+8.4f}")
    for k in ks:
        d = mean_recalls[k] - base_means[k]
        print(f"   {'Recall@' + str(k):<12} {base_means[k]:8.4f}  {mean_recalls[k]:8.4f}  {d:+8.4f}")

    print(f"\n   LLM  api_calls={stats['api_calls']}  cache_hits={stats['cache_hits']}  "
          f"failures={stats['failures']}  orderings_changed={stats['reordered']}")
    if stats["missing_text"]:
        print(f"   ⚠️  {stats['missing_text']} candidate(s) had no text in the sidecar "
              f"and were sent with empty fields")

    delta_mrr = mrr - base_mrr
    if abs(delta_mrr) < 1e-9:
        verdict = "no change - the LLM did not improve on ML+RAG"
    elif delta_mrr > 0:
        verdict = f"improved MRR by {delta_mrr:+.4f}"
    else:
        verdict = f"REGRESSED MRR by {delta_mrr:+.4f} - the LLM made it worse"
    print(f"\n   verdict: {verdict}")
    if out_path:
        print(f"   metrics  -> {out_path}")
    if reranked_out:
        print(f"   reranked -> {reranked_out}")

    return {
        "queries": stats["queries"],
        "n_lines": stats["n_lines"],
        "mrr": mrr,
        "baseline_mrr": (sum(base_rr) / len(base_rr)) if base_rr else 0.0,
        "baseline_recall": {k: (sum(v) / len(v) if v else 0.0)
                            for k, v in base_recalls.items()},
        "mean_latency": mean_lat,
        "recall": mean_recalls,
        "elapsed_sec": elapsed,
        **stats,
        "out": out_path,
        "reranked_out": reranked_out,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage 2: LLM-rerank a Stage 1 candidate hand-off.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--candidates", required=True,
                   help="JSONL from ml_to_rag_pipeline.py --candidates-out")
    p.add_argument("--texts", required=True,
                   help="Deduplicated patch-text sidecar from hydrate_candidates.py. "
                        "REQUIRED: without it the LLM receives ids only and cannot rank.")
    p.add_argument("--out", default=None, help="CSV path for per-query metrics")
    p.add_argument("--reranked-out", default=None,
                   help="JSONL recording input/output order per query (inspect what the LLM did)")
    p.add_argument("--llm-top-k", type=int, default=10,
                   help="How many top candidates the LLM sees (default 10; 0 = all)")
    p.add_argument("--llm-model", default="gpt-4o-mini",
                   help="Chat model id (default gpt-4o-mini)")
    p.add_argument("--llm-base-url", default=None,
                   help="OpenAI-compatible endpoint instead of the OpenAI API "
                        "(Azure, vLLM, llama.cpp) - no API key needed")
    p.add_argument("--llm-timeout", type=float, default=60.0, help="Per-request timeout (s)")
    p.add_argument("--llm-cache", default=None, help="JSON file caching orderings across runs")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N queries (0 = all)")
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                   help="Comma-separated k values for Recall@k")
    p.add_argument("--progress-every", type=int, default=50, help="Queries between progress lines")
    p.add_argument("--max-desc-chars", type=int, default=MAX_DESCRIPTION_CHARS,
                   help=f"Description clip at send time (default {MAX_DESCRIPTION_CHARS}). "
                        "Text is stored untrimmed; this only bounds the prompt.")
    p.add_argument("--max-title-chars", type=int, default=MAX_TITLE_CHARS,
                   help=f"Title clip at send time (default {MAX_TITLE_CHARS})")
    p.add_argument("--max-files", type=int, default=MAX_FILES_LISTED,
                   help=f"Max file paths per patch at send time (default {MAX_FILES_LISTED})")
    p.add_argument("--log-file", default=None,
                   help="Mirror the run log to this file so you can tail/grep it "
                        "instead of scrolling the notebook output")
    p.add_argument("--verbose", action="store_true",
                   help="Debug logging (includes one line per LLM request)")
    args = p.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    view = "all" if args.llm_top_k <= 0 else str(args.llm_top_k)
    key_state = "OPENAI_API_KEY set" if os.environ.get("OPENAI_API_KEY") else "no API key"
    print(f"LLM rerank stage: {args.llm_model}, top-{view}, {key_state}"
          f"{', ' + args.llm_base_url if args.llm_base_url else ''}")

    rerank_file(
        candidates_path=args.candidates,
        texts_path=args.texts,
        out_path=args.out,
        reranked_out=args.reranked_out,
        model=args.llm_model,
        base_url=args.llm_base_url,
        llm_top_k=args.llm_top_k,
        cache_path=args.llm_cache,
        limit=args.limit,
        ks=ks,
        timeout=args.llm_timeout,
        progress_every=args.progress_every,
        max_desc_chars=args.max_desc_chars,
        max_title_chars=args.max_title_chars,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
