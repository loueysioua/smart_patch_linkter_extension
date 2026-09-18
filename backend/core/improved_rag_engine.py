"""
Improved RAG Engine — Hybrid Retrieval for Patch Similarity Detection.

Enhances the base :class:`~core.engine.SmartPatchEngine` with:

1. **Hybrid retrieval** — FAISS semantic search combined with a lightweight
   BM25-style inverted-index keyword search.
2. **File-overlap boosting** — patches sharing files with the query receive a
   score boost.
3. **Multi-query retrieval** — multiple query formulations (title+body, title
   only, file-path keywords) are merged for higher recall.

Usage::

    engine = ImprovedRAGEngine()
    engine.load_project("onap", "datasets/onap/all_candidates.csv")
    results = engine.predict("onap", patch_ref, top_k=5)
"""

from __future__ import annotations

import ast
import logging
import os
import re
from collections import Counter
from datetime import timedelta
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from core.utils import get_path_similarity_stats

logger = logging.getLogger(__name__)

_SBERT_MODEL = "all-MiniLM-L6-v2"
_MIN_TOKEN_LEN = 2  # tokens shorter than this are discarded


class ImprovedRAGEngine:
    """Enhanced RAG engine with hybrid retrieval for improved recall.

    Args:
        sbert_model: HuggingFace model name used for sentence embeddings.
        use_hybrid: When ``True``, BM25-style keyword search is layered on top
                    of FAISS semantic search.
    """

    def __init__(
        self,
        sbert_model: str = _SBERT_MODEL,
        use_hybrid: bool = True,
    ) -> None:
        logger.info("Initialising Improved RAG Engine (SBERT: %s, hybrid=%s)…", sbert_model, use_hybrid)
        self._sbert_model_name = sbert_model
        self._sbert_encoder: SentenceTransformer | None = None  # lazy-loaded

        self.datasets: dict[str, pd.DataFrame] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.faiss_indices: dict[str, faiss.Index] = {}
        self.loaded_projects: list[str] = []
        self.use_hybrid = use_hybrid

        # BM25-style structures
        self.doc_tokens: dict[str, dict[int, Counter]] = {}
        self.inverted_index: dict[str, dict[str, list[int]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_project(
        self,
        project_key: str,
        csv_path: str,
        embeddings_path: str | None = None,
    ) -> None:
        """Load a project dataset and build retrieval indices.

        Args:
            project_key: Short identifier (e.g. ``"onap"``).
            csv_path: Path to the patches CSV file.
            embeddings_path: Optional path to a ``.npy`` embedding cache.
        """
        if not os.path.exists(csv_path):
            logger.warning("CSV not found: %s — skipping project %s.", csv_path, project_key)
            return

        logger.info("[%s] Loading dataset from %s…", project_key, csv_path)
        df = pd.read_csv(csv_path, parse_dates=["created_time"], dtype={"patch_id": str})
        df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
        df["files_parsed"] = df["files"].apply(self._safe_parse_list)
        df["title"] = df["title"].fillna("")
        df["description"] = df["description"].fillna("")
        df["combined_text"] = df["title"] + " " + df["description"]

        self.datasets[project_key] = df

        embeddings = self._load_or_compute_embeddings(project_key, df, embeddings_path)

        logger.info("[%s] Building FAISS index…", project_key)
        self._build_faiss_index(project_key, embeddings)

        if self.use_hybrid:
            logger.info("[%s] Building inverted index for keyword retrieval…", project_key)
            self._build_inverted_index(project_key, df)

        self.loaded_projects.append(project_key)
        logger.info("[%s] Ready — %d patches indexed.", project_key, len(df))

    def predict(
        self,
        project: str,
        patch_ref: dict[str, Any],
        top_k: int = 5,
        window_days: int = 14,
        strategy: str = "multi_query",
    ) -> list[dict[str, Any]]:
        """Return the top-*k* most similar patches for a query patch.

        Args:
            project: Project key (must already be loaded).
            patch_ref: Dictionary with ``patch_id``, ``title``,
                       ``description``, ``created_time``, ``files``.
            top_k: Maximum number of results.
            window_days: Time-window half-width in days.
            strategy: Retrieval strategy — one of ``"multi_query"``,
                      ``"hybrid"``, or ``"file_boost"``.

        Returns:
            Sorted list of candidate dicts (highest score first).
        """
        if strategy == "hybrid":
            candidates = self.retrieve_hybrid(
                project=project,
                query_text=f"{patch_ref.get('title', '')} {patch_ref.get('description', '')}",
                top_k=top_k * 2,
                time_window_days=window_days,
                ref_time=patch_ref.get("created_time"),
                exclude_id=patch_ref.get("patch_id"),
            )
            for cand in candidates:
                file_stats = get_path_similarity_stats(
                    cand.get("files", []), patch_ref.get("files", [])
                )
                cand["score"] = 0.7 * cand["score"] + 0.3 * file_stats.get("jaccard", 0.0)
            candidates.sort(key=lambda r: r["score"], reverse=True)
            return candidates[:top_k]

        if strategy == "file_boost":
            return self.retrieve_with_file_boost(
                project=project,
                patch_ref=patch_ref,
                top_k=top_k,
                time_window_days=window_days,
            )

        # Default — "multi_query"
        return self.retrieve_multi_query(
            project=project,
            patch_ref=patch_ref,
            top_k=top_k,
            time_window_days=window_days,
        )

    def retrieve_hybrid(
        self,
        project: str,
        query_text: str,
        top_k: int = 10,
        time_window_days: int | None = None,
        ref_time: pd.Timestamp | None = None,
        exclude_id: str | None = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        retrieval_factor: int = 5,
    ) -> list[dict[str, Any]]:
        """Merge FAISS semantic search with BM25 keyword search.

        Args:
            project: Project key.
            query_text: Raw text query.
            top_k: Number of final results.
            time_window_days: Optional time-window filter (days).
            ref_time: Reference timestamp for the time window.
            exclude_id: Patch ID to exclude from results.
            semantic_weight: Weight for the normalised semantic score.
            keyword_weight: Weight for the normalised keyword score.
            retrieval_factor: Expand candidate pool to ``top_k * retrieval_factor``
                              before filtering.

        Returns:
            Sorted list of candidate dicts.
        """
        if project not in self.faiss_indices:
            return []

        df = self.datasets[project]
        encoder = self._get_encoder()
        retrieval_k = top_k * retrieval_factor

        # --- Semantic search ---
        query_emb = encoder.encode([query_text], convert_to_numpy=True)
        query_emb_norm = self._normalize(query_emb)
        raw_scores, raw_indices = self.faiss_indices[project].search(
            query_emb_norm.astype("float32"), min(retrieval_k, len(df))
        )
        semantic_scores: dict[int, float] = {
            int(idx): float(score)
            for score, idx in zip(raw_scores[0], raw_indices[0])
            if idx >= 0
        }

        # --- Keyword search ---
        keyword_scores: dict[int, float] = {}
        if self.use_hybrid:
            kw_results = self._keyword_search(project, query_text, top_k=retrieval_k)
            if kw_results:
                max_kw = max(s for _, s in kw_results)
                keyword_scores = {idx: score / max_kw for idx, score in kw_results}

        # --- Merge and filter ---
        all_candidates = set(semantic_scores) | set(keyword_scores)
        results: list[dict[str, Any]] = []

        for idx in all_candidates:
            row = df.iloc[idx]

            if time_window_days and ref_time is not None:
                delta_days = abs((row["created_time"] - ref_time).total_seconds() / 86400)
                if delta_days > time_window_days:
                    continue

            if exclude_id and row["patch_id"] == exclude_id:
                continue

            sem_score = semantic_scores.get(idx, 0.0)
            kw_score = keyword_scores.get(idx, 0.0)
            combined = semantic_weight * sem_score + keyword_weight * kw_score

            results.append({
                "patch_id": row["patch_id"],
                "score": combined,
                "semantic_score": sem_score,
                "keyword_score": kw_score,
                "title": row["title"],
                "description": row["description"],
                "created_time": row["created_time"],
                "files": row["files_parsed"],
                "idx": idx,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def retrieve_with_file_boost(
        self,
        project: str,
        patch_ref: dict[str, Any],
        top_k: int = 10,
        time_window_days: int = 14,
        file_boost_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval followed by a file-overlap re-scoring pass.

        Patches that share files with the query receive a score boost
        proportional to their Jaccard file overlap.

        Args:
            project: Project key.
            patch_ref: Query patch dict.
            top_k: Number of results.
            time_window_days: Time-window filter.
            file_boost_weight: Weight given to file-overlap boosting
                               (``0.0`` = no boost, ``1.0`` = only file overlap).

        Returns:
            Re-scored and re-sorted candidate list.
        """
        query_text = f"{patch_ref.get('title', '')} {patch_ref.get('description', '')}"
        query_files = set(patch_ref.get("files", []))

        candidates = self.retrieve_hybrid(
            project=project,
            query_text=query_text,
            top_k=top_k * 3,
            time_window_days=time_window_days,
            ref_time=patch_ref.get("created_time"),
            exclude_id=patch_ref.get("patch_id"),
        )

        for cand in candidates:
            cand_files = set(cand.get("files", []))
            union = query_files | cand_files
            file_overlap = len(query_files & cand_files) / len(union) if union else 0.0
            cand["file_overlap"] = file_overlap
            cand["original_score"] = cand["score"]
            cand["score"] = (1 - file_boost_weight) * cand["score"] + file_boost_weight * file_overlap

        candidates.sort(key=lambda r: r["score"], reverse=True)
        return candidates[:top_k]

    def retrieve_multi_query(
        self,
        project: str,
        patch_ref: dict[str, Any],
        top_k: int = 10,
        time_window_days: int = 14,
    ) -> list[dict[str, Any]]:
        """Merge results from multiple query formulations for higher recall.

        Three query variants are tried in order:

        1. ``title + description`` (primary)
        2. ``title`` only
        3. Meaningful file-path segments (top 5 files, top 20 tokens)

        A candidate appearing in multiple queries receives a small score boost.
        Final scoring blends retrieval score (60 %) with Jaccard file overlap (40 %).

        Args:
            project: Project key.
            patch_ref: Query patch dict.
            top_k: Maximum results.
            time_window_days: Time-window filter.

        Returns:
            Sorted list of candidate dicts.
        """
        ref_time = patch_ref.get("created_time")
        exclude_id = patch_ref.get("patch_id")
        merged: dict[str, dict[str, Any]] = {}

        def _merge(results: list[dict[str, Any]]) -> None:
            for r in results:
                pid = r["patch_id"]
                if pid not in merged:
                    merged[pid] = {**r, "query_count": 1}
                else:
                    merged[pid]["score"] = max(merged[pid]["score"], r["score"])
                    merged[pid]["query_count"] += 1

        # --- Query 1: title + description ---
        q1 = f"{patch_ref.get('title', '')} {patch_ref.get('description', '')}"
        _merge(self.retrieve_hybrid(
            project, q1, top_k=top_k * 2,
            time_window_days=time_window_days, ref_time=ref_time, exclude_id=exclude_id,
        ))

        # --- Query 2: title only ---
        if patch_ref.get("title"):
            _merge(self.retrieve_hybrid(
                project, patch_ref["title"], top_k=top_k,
                time_window_days=time_window_days, ref_time=ref_time, exclude_id=exclude_id,
            ))

        # --- Query 3: file-path keywords ---
        if patch_ref.get("files"):
            file_tokens: list[str] = []
            for f in patch_ref["files"][:5]:
                file_tokens.extend(
                    part for part in f.split("/")
                    if len(part) > 3 and not part.startswith(".")
                )
            if file_tokens:
                q3 = " ".join(file_tokens[:20])
                _merge(self.retrieve_hybrid(
                    project, q3, top_k=top_k,
                    time_window_days=time_window_days, ref_time=ref_time, exclude_id=exclude_id,
                ))

        candidates = list(merged.values())

        # Boost score by query-count coverage
        for cand in candidates:
            cand["score"] *= 1 + 0.1 * cand.get("query_count", 1)

        # Final re-scoring: blend retrieval score with file similarity
        for cand in candidates:
            file_stats = get_path_similarity_stats(
                cand.get("files", []), patch_ref.get("files", [])
            )
            cand["file_stats"] = file_stats
            cand["score"] = 0.6 * cand["score"] + 0.4 * file_stats.get("jaccard", 0.0)

        candidates.sort(key=lambda r: r["score"], reverse=True)
        return candidates[:top_k]

    def get_patch_details(self, project: str, patch_id: str) -> dict[str, Any] | None:
        """Retrieve patch metadata from the loaded dataset.

        Args:
            project: Project key.
            patch_id: Target patch identifier.

        Returns:
            Patch dict or ``None`` if not found.
        """
        df = self.datasets.get(project)
        if df is None:
            return None
        row_df = df[df.patch_id == patch_id]
        if row_df.empty:
            return None
        row = row_df.iloc[0]
        return {
            "patch_id": row["patch_id"],
            "title": row["title"],
            "description": row["description"],
            "created_time": row["created_time"],
            "files": row["files_parsed"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_encoder(self) -> SentenceTransformer:
        """Lazy-load and return the SBERT encoder."""
        if self._sbert_encoder is None:
            logger.info("Loading SBERT model '%s'…", self._sbert_model_name)
            self._sbert_encoder = SentenceTransformer(self._sbert_model_name)
        return self._sbert_encoder

    @staticmethod
    def _safe_parse_list(value: Any) -> list[str]:
        """Safely deserialise a stringified Python list from a CSV cell."""
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split on non-word chars, and filter short tokens.

        Args:
            text: Raw text to tokenise.

        Returns:
            List of tokens with length > :data:`_MIN_TOKEN_LEN`.
        """
        if not isinstance(text, str):
            return []
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if len(t) > _MIN_TOKEN_LEN]

    def _build_inverted_index(self, project_key: str, df: pd.DataFrame) -> None:
        """Construct a token → [doc_idx, …] inverted index for *project_key*."""
        self.doc_tokens[project_key] = {}
        self.inverted_index[project_key] = {}

        for idx, row in df.iterrows():
            tokens = self._tokenize(f"{row['title']} {row['description']}")
            self.doc_tokens[project_key][idx] = Counter(tokens)
            for token in tokens:
                self.inverted_index[project_key].setdefault(token, []).append(idx)

    def _keyword_search(
        self, project: str, query_text: str, top_k: int = 100
    ) -> list[tuple[int, float]]:
        """IDF-weighted keyword search against the inverted index.

        Args:
            project: Project key.
            query_text: Raw query string.
            top_k: Maximum number of (doc_idx, score) pairs to return.

        Returns:
            List of ``(doc_idx, score)`` tuples, highest score first.
        """
        query_tokens = self._tokenize(query_text)
        if not query_tokens or project not in self.inverted_index:
            return []

        doc_scores: Counter = Counter()
        n_docs = len(self.datasets[project])
        inv_idx = self.inverted_index[project]

        for token in query_tokens:
            if token in inv_idx:
                idf = np.log(n_docs / (len(inv_idx[token]) + 1))
                for doc_idx in inv_idx[token]:
                    doc_scores[doc_idx] += idf

        return doc_scores.most_common(top_k)

    def _load_or_compute_embeddings(
        self,
        project_key: str,
        df: pd.DataFrame,
        cache_path: str | None,
    ) -> np.ndarray:
        """Load from cache or compute + save SBERT embeddings."""
        if cache_path and os.path.exists(cache_path):
            logger.info("[%s] Loading cached embeddings from %s…", project_key, cache_path)
            try:
                cached = np.load(cache_path, allow_pickle=True).item()
                if cached.get("n_rows") == len(df):
                    return cached["embeddings"]
                logger.warning("[%s] Cache row-count mismatch — recomputing.", project_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Failed to read cache (%s) — recomputing.", project_key, exc)

        logger.info("[%s] Computing embeddings for %d patches…", project_key, len(df))
        encoder = self._get_encoder()
        texts = (df["title"] + " " + df["description"]).tolist()
        embeddings: np.ndarray = encoder.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )

        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, {
                "embeddings": embeddings,
                "n_rows": len(df),
                "sbert_model": self._sbert_model_name,
            })
            logger.info("[%s] Cached embeddings → %s", project_key, cache_path)

        return embeddings

    def _build_faiss_index(self, project_key: str, embeddings: np.ndarray) -> None:
        """Normalise embeddings and load them into a FAISS ``IndexFlatIP``."""
        embeddings_norm = self._normalize(embeddings)
        dim = embeddings_norm.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_norm.astype("float32"))
        self.embeddings[project_key] = embeddings
        self.faiss_indices[project_key] = index

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """Return L2-normalised copy of *embeddings*."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
