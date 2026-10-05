"""
Shared feature + embedding-cache utilities for SmartPatch.

Canonical home for the pairwise patch-linkage features (used by BOTH training
and serving) and for the embeddings-cache validity check. One implementation
on purpose: training and serving each carried their own copy, they drifted, and
the drift was silent because reindex(columns=feature_cols, fill_value=0)
zero-filled the features only one side knew about.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path_to_segments(path: str) -> list[str]:
    """Split a file path string into its directory/file segments.

    Args:
        path: A POSIX-style file path string (e.g. ``"foo/bar/baz.py"``).

    Returns:
        A list of non-empty path segments, or an empty list for non-strings.
    """
    if not isinstance(path, str):
        return []
    return [seg for seg in path.strip().split("/") if seg]


def _top_level(path: str) -> str | None:
    """First path segment of a file path, or ``None`` if there is none."""
    segments = _path_to_segments(path)
    return segments[0] if segments else None


def _longest_common_prefix(path_a: str, path_b: str) -> int:
    """Count matching segments from the start of two paths."""
    segs_a = _path_to_segments(path_a)
    segs_b = _path_to_segments(path_b)
    return sum(1 for a, b in zip(segs_a, segs_b) if a == b)


def _longest_common_suffix(path_a: str, path_b: str) -> int:
    """Count matching segments from the end of two paths."""
    segs_a = _path_to_segments(path_a)
    segs_b = _path_to_segments(path_b)
    return sum(1 for a, b in zip(reversed(segs_a), reversed(segs_b)) if a == b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_path_similarity_stats(
    files_a: list[str],
    files_b: list[str],
) -> dict[str, float]:
    """Compute pairwise path similarity statistics between two file lists.

    Metrics returned:

    * ``jaccard`` – Jaccard index of the two file sets.
    * ``nb_shared`` – Number of files present in both lists.
    * ``abs_delta_files`` – Absolute difference in list length. Absolute, not
      signed: every pairwise feature must satisfy f(A, B) == f(B, A) for
      LambdaRank groups to be order-independent.
    * ``min_len_files`` / ``max_len_files`` – Smaller / larger file-list length.
    * ``same_component`` – Both lists touch at least one common top-level dir.
    * ``LCP_mean`` / ``LCP_max`` – Normalised longest-common-prefix averages.
    * ``LCSuff_mean`` / ``LCSuff_max`` – Normalised longest-common-suffix averages.

    Args:
        files_a: File paths from the first patch.
        files_b: File paths from the second patch.

    Returns:
        Dictionary of metric name → float value.
    """
    stats: dict[str, float] = {}

    # --- Set-level metrics --------------------------------------------------
    set_a, set_b = set(files_a), set(files_b)
    union = set_a | set_b
    stats["jaccard"] = len(set_a & set_b) / len(union) if union else 0.0
    stats["nb_shared"] = float(len(set_a & set_b))
    stats["abs_delta_files"] = float(abs(len(files_a) - len(files_b)))
    stats["min_len_files"] = float(min(len(files_a), len(files_b)))
    stats["max_len_files"] = float(max(len(files_a), len(files_b)))

    comp_a = {_top_level(f) for f in files_a}
    comp_b = {_top_level(f) for f in files_b}
    comp_a.discard(None)
    comp_b.discard(None)
    stats["same_component"] = float(bool(comp_a & comp_b))

    # --- Path-level metrics (skip if either list is empty) ------------------
    if not files_a or not files_b:
        for key in ("LCP_mean", "LCP_max", "LCSuff_mean", "LCSuff_max"):
            stats[key] = 0.0
        return stats

    def _normalised_lcp(fa: str, fb: str) -> float:
        denom = max(len(_path_to_segments(fa)), len(_path_to_segments(fb)), 1)
        return _longest_common_prefix(fa, fb) / denom

    def _normalised_lcsuff(fa: str, fb: str) -> float:
        denom = max(len(_path_to_segments(fa)), len(_path_to_segments(fb)), 1)
        return _longest_common_suffix(fa, fb) / denom

    lcp_scores = [_normalised_lcp(fa, fb) for fa in files_a for fb in files_b]
    lcsuff_scores = [_normalised_lcsuff(fa, fb) for fa in files_a for fb in files_b]

    stats["LCP_mean"] = float(np.mean(lcp_scores))
    stats["LCP_max"] = float(np.max(lcp_scores))
    stats["LCSuff_mean"] = float(np.mean(lcsuff_scores))
    stats["LCSuff_max"] = float(np.max(lcsuff_scores))

    return stats


def tokenize(text: str) -> set[str]:
    """Lowercased alphanumeric token set used for Jaccard similarity."""
    return set(w.lower() for w in TOKEN_RE.findall(text or ""))


def token_jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    a, b = tokenize(text_a), tokenize(text_b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def build_pair_features(
    df,
    emb: np.ndarray,
    emb_disc: np.ndarray,
    i: int,
    j: int,
) -> dict[str, float]:
    """Build the 17 pairwise features for rows ``i`` and ``j`` of ``df``.

    This is THE feature contract. Training (``ml_pipeline``) and serving
    (``scripts/ml_to_rag_pipeline.py``) both call this so a saved model's
    ``feature_cols`` can never drift away from what serving computes.

    ``emb`` / ``emb_disc`` must be row-aligned with ``df`` -- use
    ``assert_embeddings_aligned`` if that alignment is not already guaranteed
    by the caller.

    Args:
        df: Frame with ``title``, ``description``, ``created_time``,
            ``files_parsed``, ``discussion_text``, ``has_discussion`` and
            ``ticket_refs`` columns.
        emb: Row-aligned title+description embeddings.
        emb_disc: Row-aligned discussion-text embeddings.
        i: Row index of the anchor patch.
        j: Row index of the candidate patch.

    Returns:
        Feature name -> float. Identical for ``(i, j)`` and ``(j, i)``.
    """
    row_i, row_j = df.iloc[i], df.iloc[j]

    feats = get_path_similarity_stats(row_i["files_parsed"], row_j["files_parsed"])

    feats["sim_cosine"] = float(cosine_similarity(
        emb[i].reshape(1, -1), emb[j].reshape(1, -1)
    )[0][0])

    feats["token_jaccard"] = token_jaccard(
        f"{row_i['title']} {row_i['description']}",
        f"{row_j['title']} {row_j['description']}",
    )

    feats["delta_time_hours"] = abs(
        (row_i["created_time"] - row_j["created_time"]).total_seconds() / 3600
    )

    has_i = bool(row_i.get("has_discussion"))
    has_j = bool(row_j.get("has_discussion"))
    if has_i and has_j:
        feats["sim_cosine_discussion"] = float(cosine_similarity(
            emb_disc[i].reshape(1, -1), emb_disc[j].reshape(1, -1)
        )[0][0])
        feats["token_jaccard_discussion"] = token_jaccard(
            row_i["discussion_text"], row_j["discussion_text"]
        )
    else:
        feats["sim_cosine_discussion"] = 0.0
        feats["token_jaccard_discussion"] = 0.0

    feats["has_discussion_both"] = float(has_i and has_j)
    feats["shares_ticket_ref"] = float(bool(
        (row_i.get("ticket_refs") or set()) & (row_j.get("ticket_refs") or set())
    ))

    return feats


# ---------------------------------------------------------------------------
# Embeddings cache
# ---------------------------------------------------------------------------
# An embeddings cache is only reusable if it encodes the SAME rows in the SAME
# order with the SAME model. Row count alone cannot detect a reordering -- a
# permutation preserves n_rows -- so a cache written before a sort_values()
# removal stays "valid" while every embedding is silently attributed to the
# wrong patch. The fingerprint below hashes the exact (patch_id, text) payload
# that was encoded, which does detect reordering, row edits and text edits.

def title_desc_texts(df) -> list[str]:
    """The exact title+description payload encoded for title/desc embeddings."""
    return (df["title"] + " " + df["description"]).tolist()


def embeddings_fingerprint(ids, texts) -> str:
    """Stable fingerprint of an encoded payload: row identity AND text, in order."""
    digest = hashlib.sha1()
    for row_id, text in zip(ids, texts):
        digest.update(str(row_id).encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update(str(text).encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def cache_is_valid(cached: dict, df, model_name: str, texts=None) -> bool:
    """Whether a loaded embeddings cache matches ``df`` row-for-row.

    Accepts either metadata key spelling used by the writers in this repo
    (``model_name`` from the ML pipeline, ``sbert_model`` from the engines).
    A cache with no fingerprint is treated as invalid, so adding this check
    invalidates every pre-existing cache exactly once.

    Args:
        cached: The dict previously ``np.save``-d to the cache path.
        df: DataFrame the caller is about to index embeddings with.
        model_name: SBERT model the caller needs.
        texts: Payload that was encoded. Defaults to title+description.

    Returns:
        ``True`` only if model, row count and fingerprint all match.
    """
    if not isinstance(cached, dict):
        return False
    if cached.get("n_rows") != len(df):
        return False
    if model_name not in (cached.get("model_name"), cached.get("sbert_model")):
        return False
    if texts is None:
        texts = title_desc_texts(df)
    return cached.get("fingerprint") == embeddings_fingerprint(df["patch_id"], texts)


def assert_embeddings_aligned(df, emb: np.ndarray, label: str = "embeddings") -> None:
    """Raise if an embedding matrix is not row-aligned with ``df``.

    Cheap insurance for the failure mode that costs the most to debug: a
    plausible-looking ranking built from another patch's embedding. There is
    no way to detect this from shapes, so callers that did not build the
    matrix themselves in this exact order should say so out loud.
    """
    if len(emb) != len(df):
        raise ValueError(
            f"{label} has {len(emb)} rows but df has {len(df)} -- not row-aligned"
        )


TICKET_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d+|#\d{3,7})\b")
"""Matches bug/ticket references such as ``BUG-1234`` or ``#4821``."""


def safe_parse_json(x) -> list:
    """Parse a JSON-encoded list, tolerating nulls, NaN and malformed input."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if not isinstance(x, str):
        return []
    try:
        parsed = json.loads(x)
        return parsed if isinstance(parsed, list) else []
    except ValueError:  # includes json.JSONDecodeError
        return []


def extract_discussion_text(row) -> str:
    """Best-effort discussion blob from a row's change_log / comments entries.

    Shared by training and serving so ``discussion_text`` -- and therefore the
    discussion embeddings fingerprinted against it -- is built identically on
    both sides.
    """
    parts: list[str] = []
    for parsed_key in ("change_log_parsed", "comments_parsed"):
        for entry in (row.get(parsed_key) or []):
            if isinstance(entry, dict):
                for key in ("message", "text", "body", "comment", "description"):
                    val = entry.get(key)
                    if val:
                        parts.append(str(val))
                        break
            elif isinstance(entry, str):
                parts.append(entry)
    return " ".join(parts)


def extract_ticket_refs(text: str) -> set[str]:
    """Uppercased ticket references mentioned in ``text``.

    Lets two patches that name the same ticket be flagged as linked even when
    their title/description wording differs.
    """
    return set(m.upper() for m in TICKET_RE.findall(text or ""))
