"""
candidate_retrieval.py

ANN-based candidate retrieval ("recall stage") for the patch-linking
pipeline in train_patch_ranker.py.

Replaces the pure time-window candidate generator with a hybrid:
  - all patches within +/- `window_days` of the anchor (existing behavior)
  - the top-`ann_k` nearest neighbors by embedding cosine similarity,
    restricted to a wider +/- `ann_max_days` bound

This directly targets the failure mode flagged by report_window_coverage():
linked pairs whose time delta exceeds `window_days` are currently
unrecoverable no matter how similar their text is. ANN retrieval recovers
those without blowing the window open for every anchor (which would
balloon group sizes and negative-sampling cost).

The FAISS index is built ONCE over the full embedding matrix. This is safe
to do before the train/test time split because it's unsupervised (no
ground-truth labels are used to build or query it) -- only the *labels*
used at group-construction time need to respect the time split, which
they already do in build_training_rows / positives_by_anchor.

Usage (see integration notes at the bottom of this file):

    from candidate_retrieval import CandidateIndex

    index = CandidateIndex.build(emb, df["created_time"].to_numpy())

    candidates = index.get_candidates(
        anchor_idx=42,
        anchor_time=df.loc[42, "created_time"],
        window_days=14,
        ann_k=50,
        ann_max_days=90,
    )
"""

from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "This module requires faiss. Install with `pip install faiss-cpu` "
        "(or faiss-gpu if you have CUDA available)."
    ) from e


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    mat = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@dataclass
class CandidateIndex:
    """Wraps a FAISS index over patch embeddings plus the created_time
    array needed to apply time-window filters on retrieved neighbors."""

    index: "faiss.Index"
    created_time_ns: np.ndarray  # int64 nanoseconds, for fast comparisons
    n: int

    @classmethod
    def build(cls, emb: np.ndarray, created_time: np.ndarray, use_hnsw: bool = True):
        """
        emb:            (n_patches, dim) float array of embeddings
                         (title+description SBERT embeddings, typically)
        created_time:   (n_patches,) array-like of pandas/np datetime64
        use_hnsw:       HNSW is fast + accurate enough for this scale and
                         needs no training step, unlike IVF. Switch to
                         IndexIVFFlat if you have >~1-2M patches and need
                         faster build/query at some recall cost.
        """
        emb_norm = _normalize(emb)
        dim = emb_norm.shape[1]

        if use_hnsw:
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 = graph connectivity (M)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
        else:
            index = faiss.IndexFlatIP(dim)  # exact search, fine for small datasets

        index.add(emb_norm)

        created_time_ns = np.asarray(created_time, dtype="datetime64[ns]").astype("int64")

        return cls(index=index, created_time_ns=created_time_ns, n=len(emb_norm))

    def _query_embedding(self, emb: np.ndarray, idx: int) -> np.ndarray:
        return _normalize(emb[idx : idx + 1])

    def get_candidates(
        self,
        anchor_idx: int,
        emb: np.ndarray,
        window_days: int,
        ann_k: int = 50,
        ann_max_days: int = 90,
        time_window_fn=None,
    ) -> list:
        """
        Returns a deduplicated list of candidate indices for `anchor_idx`,
        excluding the anchor itself.

        `time_window_fn` should be the existing get_candidate_indices from
        train_patch_ranker.py (passed in to avoid duplicating that logic /
        avoid a circular import). If None, only ANN candidates are returned.
        """
        query = self._query_embedding(emb, anchor_idx)

        # over-fetch a bit since we'll filter by time and drop the anchor itself
        fetch_k = min(ann_k * 3 + 5, self.n)
        sims, ann_idxs = self.index.search(query, fetch_k)
        ann_idxs = ann_idxs[0]
        sims = sims[0]

        anchor_time_ns = self.created_time_ns[anchor_idx]
        max_delta_ns = np.int64(ann_max_days) * np.int64(24 * 3600 * 1_000_000_000)

        ann_candidates = []
        for cand_idx, sim in zip(ann_idxs, sims):
            if cand_idx < 0 or cand_idx == anchor_idx:
                continue
            delta_ns = abs(int(self.created_time_ns[cand_idx]) - int(anchor_time_ns))
            if delta_ns > max_delta_ns:
                continue
            ann_candidates.append(int(cand_idx))
            if len(ann_candidates) >= ann_k:
                break

        time_window_candidates = time_window_fn(anchor_idx) if time_window_fn else []

        merged = list(dict.fromkeys(time_window_candidates + ann_candidates))  # preserve order, dedupe
        return merged


# --------------------------------------------------------------------------
# Integration notes for train_patch_ranker.py
# --------------------------------------------------------------------------
#
# 1. Build the index once, right after embeddings are computed:
#
#       from candidate_retrieval import CandidateIndex
#       cand_index = CandidateIndex.build(emb, df["created_time"].to_numpy())
#
# 2. Replace every call site of get_candidate_indices(df, idx, window_days)
#    with a hybrid wrapper, e.g.:
#
#       def get_hybrid_candidates(anchor_idx, window_days):
#           return cand_index.get_candidates(
#               anchor_idx=anchor_idx,
#               emb=emb,
#               window_days=window_days,
#               ann_k=args.ann_k,
#               ann_max_days=args.ann_max_days,
#               time_window_fn=lambda a: get_candidate_indices(df, a, window_days),
#           )
#
#    Call sites to update:
#       - report_window_coverage()      -> use it to re-measure coverage
#         with the hybrid candidates; should show fewer "unusable" pairs
#       - build_training_rows()         -> candidate_idxs = get_hybrid_candidates(...)
#       - evaluate()                    -> candidate_idxs = get_hybrid_candidates(...)
#
# 3. Add CLI args:
#       --ann-k            (default 50)   number of ANN neighbors to pull in
#       --ann-max-days     (default 90)   time bound applied to ANN candidates
#
# 4. Retrain from scratch after switching -- the candidate distribution the
#    model sees during training must match what it sees at inference time.
#    Don't reuse a model trained on pure time-window candidates against the
#    new hybrid candidate pool.
#
# 5. Re-run report_window_coverage-style diagnostics with hybrid candidates
#    to confirm the previously "unusable" (out-of-window) pairs are now
#    reachable. If usable_pct doesn't improve much, ann_max_days or ann_k
#    is probably too tight -- widen and re-check before touching the model.