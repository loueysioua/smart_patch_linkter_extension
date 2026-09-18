"""
SmartPatch Engine — RAG-based Patch Similarity Detection.

Uses FAISS semantic search (SBERT embeddings) for candidate retrieval,
optionally combined with file-overlap scoring for final ranking.

The engine also supports persisting and reloading pre-built FAISS indices
to avoid recomputing embeddings on every startup.
"""

from __future__ import annotations

import ast
import logging
import os
from datetime import timedelta
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from core.utils import get_path_similarity_stats

logger = logging.getLogger(__name__)

_SBERT_MODEL = "all-MiniLM-L6-v2"

# Weights for the combined relevance score
_SEMANTIC_WEIGHT = 0.7
_FILE_WEIGHT = 0.3
_RELEVANCE_THRESHOLD = 0.5


class SmartPatchEngine:
    """RAG-based patch similarity engine using FAISS + optional file-overlap scoring.

    Workflow:
    1. :meth:`load_project` — load a CSV, compute (or restore) SBERT embeddings,
       build a FAISS ``IndexFlatIP`` for fast cosine-similarity search.
    2. :meth:`predict` — encode a query patch, search the FAISS index, apply a
       time-window filter, and blend semantic + file-overlap scores.
    """

    def __init__(self, sbert_model: str = _SBERT_MODEL) -> None:
        logger.info("Initialising SmartPatch Engine (SBERT: %s)…", sbert_model)
        self.sbert = SentenceTransformer(sbert_model)
        self._sbert_model_name = sbert_model
        self.datasets: dict[str, pd.DataFrame] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.faiss_indices: dict[str, faiss.Index] = {}
        self.loaded_projects: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_project(
        self,
        project_key: str,
        csv_path: str,
        embeddings_path: str | None = None,
        *,
        model_path: str | None = None,  # kept for backward compat, ignored
    ) -> None:
        """Load project data and build a FAISS index.

        Args:
            project_key: Short project identifier (e.g. ``"onap"``).
            csv_path: Path to the patches CSV file
                      (columns: ``patch_id``, ``created_time``, ``title``,
                      ``description``, ``files``).
            embeddings_path: Optional path to a ``.npy`` cache file.  When
                             provided, cached embeddings are loaded if they
                             match the current row count; otherwise they are
                             recomputed and saved.
            model_path: **Deprecated** — ignored.  Kept for backward
                        compatibility with older call sites.
        """
        if not os.path.exists(csv_path):
            logger.warning("CSV file not found: %s — skipping project %s", csv_path, project_key)
            return

        logger.info("[%s] Loading dataset from %s…", project_key, csv_path)
        df = pd.read_csv(csv_path, parse_dates=["created_time"], dtype={"patch_id": str})
        df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
        df["files_parsed"] = df["files"].apply(self._safe_parse_list)
        df["title"] = df["title"].fillna("")
        df["description"] = df["description"].fillna("")

        self.datasets[project_key] = df

        embeddings = self._load_or_compute_embeddings(project_key, df, embeddings_path)
        logger.info("[%s] Building FAISS index…", project_key)
        self._build_faiss_index(project_key, embeddings)

        self.loaded_projects.append(project_key)
        logger.info("[%s] Ready — %d patches indexed.", project_key, len(df))

    def predict(
        self,
        project: str,
        patch_ref: dict[str, Any],
        top_k: int = 5,
        window_days: int = 14,
    ) -> list[dict[str, Any]]:
        """Return the top-*k* most similar patches for a given query patch.

        Args:
            project: Project key (must be loaded via :meth:`load_project`).
            patch_ref: Dictionary with keys ``patch_id``, ``title``,
                       ``description``, ``created_time``, ``files``.
            top_k: Maximum number of results to return.
            window_days: Only consider patches within ±*window_days* of the
                         query's creation date.

        Returns:
            List of result dicts sorted by combined score (descending).
            Each result contains: ``patch_id``, ``score``, ``is_related``,
            ``created_time``, ``title``, ``description``, ``files``.
        """
        if project not in self.faiss_indices:
            logger.warning("Project '%s' not loaded.", project)
            return []

        df = self.datasets[project]
        index = self.faiss_indices[project]

        query_text = f"{patch_ref.get('title', '')} {patch_ref.get('description', '')}"
        query_emb = self.sbert.encode([query_text], convert_to_numpy=True)
        query_emb = self._normalize(query_emb)

        # Fetch a larger pool to account for time-window filtering
        search_k = min(top_k * 3, len(df))
        scores, indices = index.search(query_emb.astype("float32"), search_k)

        ref_time = patch_ref.get("created_time")
        exclude_id = patch_ref.get("patch_id")
        results: list[dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS sentinel for empty slots
                continue

            row = df.iloc[idx]

            if ref_time and window_days:
                delta_days = abs((row["created_time"] - ref_time).total_seconds() / 86400)
                if delta_days > window_days:
                    continue

            if exclude_id and row["patch_id"] == exclude_id:
                continue

            file_stats = get_path_similarity_stats(
                row["files_parsed"], patch_ref.get("files", [])
            )
            combined_score = (
                _SEMANTIC_WEIGHT * float(score)
                + _FILE_WEIGHT * file_stats.get("jaccard", 0.0)
            )

            results.append({
                "patch_id": row["patch_id"],
                "score": combined_score,
                "is_related": combined_score > _RELEVANCE_THRESHOLD,
                "created_time": row["created_time"].strftime("%Y-%m-%d"),
                "title": row["title"],
                "description": row["description"],
                "files": row["files_parsed"],
            })

            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def get_candidates(
        self,
        project: str,
        ref_time: pd.Timestamp,
        exclude_id: str,
        days: int = 14,
    ) -> pd.DataFrame:
        """Return candidate patches within a time window (legacy helper).

        Args:
            project: Project key.
            ref_time: Reference creation timestamp.
            exclude_id: Patch ID to exclude.
            days: Half-width of the time window in days.

        Returns:
            Filtered DataFrame (may be empty).
        """
        df = self.datasets.get(project)
        if df is None:
            return pd.DataFrame()

        start = ref_time - timedelta(days=days)
        end = ref_time + timedelta(days=days)
        mask = (
            (df.created_time >= start)
            & (df.created_time <= end)
            & (df.patch_id != exclude_id)
        )
        return df[mask].copy()

    def save_index(self, project: str, output_dir: str) -> None:
        """Persist the FAISS index and embeddings for *project* to disk.

        Args:
            project: Project key whose index should be saved.
            output_dir: Directory where index files will be written.
        """
        if project not in self.faiss_indices:
            logger.warning("No index loaded for project '%s' — nothing to save.", project)
            return

        os.makedirs(output_dir, exist_ok=True)

        index_path = os.path.join(output_dir, f"{project}_faiss.index")
        faiss.write_index(self.faiss_indices[project], index_path)
        logger.info("Saved FAISS index → %s", index_path)

        emb_path = os.path.join(output_dir, f"{project}_embeddings.npy")
        np.save(emb_path, {
            "embeddings": self.embeddings[project],
            "n_rows": len(self.datasets[project]),
            "sbert_model": self._sbert_model_name,
        })
        logger.info("Saved embeddings → %s", emb_path)

    def load_index(self, project: str, csv_path: str, index_dir: str) -> None:
        """Load a pre-built FAISS index from disk instead of recomputing.

        Args:
            project: Project key to register.
            csv_path: Path to the patches CSV (still needed for the DataFrame).
            index_dir: Directory containing ``{project}_faiss.index`` and
                       optionally ``{project}_embeddings.npy``.
        """
        if not os.path.exists(csv_path):
            logger.warning("CSV file not found: %s", csv_path)
            return

        logger.info("[%s] Loading dataset from %s…", project, csv_path)
        df = pd.read_csv(csv_path, parse_dates=["created_time"], dtype={"patch_id": str})
        df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
        df["files_parsed"] = df["files"].apply(self._safe_parse_list)
        df["title"] = df["title"].fillna("")
        df["description"] = df["description"].fillna("")
        self.datasets[project] = df

        index_path = os.path.join(index_dir, f"{project}_faiss.index")
        if not os.path.exists(index_path):
            logger.warning("FAISS index not found: %s", index_path)
            return

        logger.info("[%s] Loading FAISS index from %s…", project, index_path)
        self.faiss_indices[project] = faiss.read_index(index_path)

        emb_path = os.path.join(index_dir, f"{project}_embeddings.npy")
        if os.path.exists(emb_path):
            cached = np.load(emb_path, allow_pickle=True).item()
            self.embeddings[project] = cached["embeddings"]

        self.loaded_projects.append(project)
        logger.info("[%s] Loaded from pre-built index — %d patches.", project, len(df))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_list(value: Any) -> list[str]:
        """Safely deserialise a stringified Python list.

        Args:
            value: Raw cell value from the CSV (may be a list, string, or NaN).

        Returns:
            Parsed list of strings, or an empty list on failure.
        """
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []

    def _load_or_compute_embeddings(
        self,
        project_key: str,
        df: pd.DataFrame,
        cache_path: str | None,
    ) -> np.ndarray:
        """Load embeddings from a ``.npy`` cache or compute + cache them.

        Args:
            project_key: Used for log messages.
            df: DataFrame whose ``title`` + ``description`` columns are encoded.
            cache_path: Path to the ``.npy`` cache file (``None`` → no caching).

        Returns:
            Float32 embedding matrix of shape ``(len(df), embedding_dim)``.
        """
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
        texts = (df["title"] + " " + df["description"]).tolist()
        embeddings: np.ndarray = self.sbert.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
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
        """L2-normalise *embeddings* and load them into a FAISS ``IndexFlatIP``.

        After normalisation, inner product == cosine similarity.

        Args:
            project_key: Used as the key in :attr:`faiss_indices`.
            embeddings: Raw (unnormalised) embedding matrix.
        """
        embeddings_norm = self._normalize(embeddings)
        dim = embeddings_norm.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_norm.astype("float32"))
        self.embeddings[project_key] = embeddings
        self.faiss_indices[project_key] = index

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """Return L2-normalised copy of *embeddings*.

        Zero-vectors are left unchanged to avoid division by zero.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
