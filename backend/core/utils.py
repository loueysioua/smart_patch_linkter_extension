"""
Path similarity utilities for SmartPatch.

Provides Jaccard set similarity and prefix/suffix path overlap metrics
used when scoring candidate patch relevance.
"""

from __future__ import annotations

import numpy as np


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
    * ``delta_files`` – Signed difference in list length (``len(A) - len(B)``).
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
    stats["delta_files"] = float(len(files_a) - len(files_b))

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