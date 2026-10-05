"""
ML Model → RAG Pipeline Test

This script tests the reverse approach:
1. ML model (LightGBM) predicts candidate scores
2. RAG engine reranks/refines the top candidates

This contrasts with the RAG → ML approach where RAG retrieves candidates
and ML ranks them.

Usage:
    python ml_to_rag_pipeline.py --project onap --patch-id <patch_id>
    python ml_to_rag_pipeline.py --project onap --evaluate --window-days 14
"""

import argparse
import ast
import hashlib
import json
import sys
import os
import time
from datetime import timedelta
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from core.improved_rag_engine import ImprovedRAGEngine
from core.llm_rerank import rerank_candidates
from core.utils import (
    assert_embeddings_aligned,
    build_pair_features,
    cache_is_valid,
    embeddings_fingerprint,
    extract_discussion_text,
    extract_ticket_refs,
    get_path_similarity_stats,
    safe_parse_json,
    title_desc_texts,
)

class MLToRAGPipeline:
    """
    Pipeline that uses ML model first, then RAG for refinement.
    
    Workflow:
    1. Get time-window candidates
    2. ML model scores and ranks all candidates
    3. Take top-k*2 candidates from ML
    4. RAG engine reranks them with semantic + file overlap scores
    5. Return final top-k results
    """
    
    def __init__(
        self,
        model_path: str,
        project: str,
        csv_path: str,
        rag_engine: ImprovedRAGEngine = None,
        verbose: bool = True,
    ):
        """Initialize the pipeline.
        
        Args:
            model_path: Path to trained LightGBM model (.pkl)
            project: Project name
            csv_path: Path to candidate CSV
            rag_engine: Optional pre-loaded RAG engine
            verbose: Print per-query progress. Turn OFF for batch evaluation --
                a full ground-truth sweep calls predict() thousands of times and
                the per-query trace is ~17 lines each (~60k lines). Startup
                messages and warnings are always shown.
        """
        self.verbose = verbose
        # Blend weights for the RAG refine stage. rag_weight=0 makes the final
        # ranking pure ML, which is the baseline for measuring what the
        # reranker actually contributes on an identical query set.
        self.ml_weight = 0.5
        self.rag_weight = 0.5

        # --- Optional Stage 3: LLM rerank (off unless --llm-rerank is passed) ---
        self.llm_enabled = False
        self.llm_model = "gpt-4o-mini"
        # How many top candidates the LLM actually sees. Independent of --top-k
        # so the prompt size (and cost) stays fixed even when the retrieval
        # depth changes. 0 = send everything.
        self.llm_top_k = 10
        self.llm_base_url = None
        self.llm_timeout = 60.0
        self.llm_cache_path = None
        self.llm_cache: Dict[str, list] = {}
        self._llm_client = None
        self._llm_stats = {"calls": 0, "cache_hits": 0, "failures": 0, "reordered": 0}

        self._warned: set = set()
        self.project = project
        self.model_data = joblib.load(model_path)
        self.model = self.model_data["model"]
        self.feature_cols = self.model_data["feature_cols"]
        self.window_days = self.model_data.get("window_days", 30)
        self.eval_window_days = self.model_data.get("eval_window_days", 2)  # Bug B fix: Use eval window
        self.sbert_model_name = self.model_data.get("sbert_model_name", "all-MiniLM-L6-v2")
        
        print(f"📦 Loaded ML model from {model_path}")
        print(f"   Training window: ±{self.window_days} days")
        print(f"   Eval window: ±{self.eval_window_days} days (recommended for inference)")
        print(f"   Features: {len(self.feature_cols)}")
        
        # Load or use provided RAG engine
        if rag_engine:
            self.rag_engine = rag_engine
        else:
            print(f"\n� Initializing RAG engine...")
            self.rag_engine = ImprovedRAGEngine(use_hybrid=True)
            # Pass embeddings cache path to avoid recomputing
            embeddings_cache = f"data/{project}/embeddings_cache.npy"
            self.rag_engine.load_project(project, csv_path, embeddings_path=embeddings_cache)
        
        # BUG #1 FIX: Reuse rag_engine's dataset to avoid index misalignment
        # The RAG engine has already loaded the CSV and computed embeddings in a 
        # consistent order. Reusing its dataframe ensures self.df row indices 
        # align with self.rag_engine.embeddings indices.
        if project in self.rag_engine.datasets:
            print(f"\n� Reusing RAG engine's dataset for {project} (consistent ordering)...")
            self.df = self.rag_engine.datasets[project].copy()
            # Add the extra columns needed by ML pipeline that RAG engine doesn't set
            self.df["change_log_parsed"] = self.df["change_log"].apply(safe_parse_json) if "change_log" in self.df.columns else [[] for _ in range(len(self.df))]
            self.df["comments_parsed"] = self.df["comments"].apply(safe_parse_json) if "comments" in self.df.columns else [[] for _ in range(len(self.df))]
            self.df["discussion_text"] = self.df.apply(extract_discussion_text, axis=1)
            self.df["has_discussion"] = self.df["discussion_text"].str.strip().str.len() > 0
            self.df["ticket_refs"] = (self.df["title"] + " " + self.df["description"] + " " + self.df["discussion_text"]).apply(extract_ticket_refs)
            print(f"   Loaded {len(self.df)} patches")
        else:
            # Fallback: load independently (should not happen in normal flow)
            print(f"\n⚠️ RAG engine dataset not available, loading independently...")
            self.df = self._load_dataset(csv_path)
            print(f"   Loaded {len(self.df)} patches")
        
        # Compute embeddings for ML features (these are separate from RAG engine's embeddings)
        print(f"\n🧮 Computing embeddings for ML features...")
        self.sbert = SentenceTransformer(self.sbert_model_name)
        self.embeddings = self._compute_embeddings()
        self.discussion_embeddings = self._compute_discussion_embeddings()
        assert_embeddings_aligned(self.df, self.embeddings, "title/description embeddings")
        assert_embeddings_aligned(self.df, self.discussion_embeddings, "discussion embeddings")
        print(f"   Done!")
        
        # Build patch_id to index mapping
        self.id_to_idx = {str(pid): i for i, pid in enumerate(self.df["patch_id"])}
    
    def _log(self, msg: str = "") -> None:
        """Print only in verbose mode (per-query tracing)."""
        if self.verbose:
            print(msg)

    def _warn_once(self, key: str, msg: str) -> None:
        """Print a warning the first time it occurs, regardless of verbosity.

        Startup/config problems must stay visible when quiet, otherwise the run
        completes with plausible-looking but wrong numbers.
        """
        if key not in self._warned:
            self._warned.add(key)
            print(msg)

    def _load_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess dataset.
        
        NOTE: This method is now deprecated - the pipeline reuses rag_engine.datasets
        to avoid index misalignment bugs. The dataframe ordering must match the 
        embeddings array indices.
        """
        import ast
        
        def safe_parse_list(x):
            if isinstance(x, list):
                return x
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            try:
                parsed = ast.literal_eval(x)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                return []
        
        df = pd.read_csv(csv_path, parse_dates=["created_time"], dtype={"patch_id": str})
        df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
        df["title"] = df["title"].fillna("")
        df["description"] = df["description"].fillna("")
        df["files_parsed"] = df["files"].apply(safe_parse_list)
        
        # Parse change_log and comments for discussion features (Bug A fix)
        df["change_log_parsed"] = df["change_log"].apply(safe_parse_json) if "change_log" in df.columns else [[] for _ in range(len(df))]
        df["comments_parsed"] = df["comments"].apply(safe_parse_json) if "comments" in df.columns else [[] for _ in range(len(df))]
        
        # Extract discussion text and metadata
        df["discussion_text"] = df.apply(extract_discussion_text, axis=1)
        df["has_discussion"] = df["discussion_text"].str.strip().str.len() > 0
        
        # Extract ticket references from title + description + discussion
        df["ticket_refs"] = (df["title"] + " " + df["description"] + " " + df["discussion_text"]).apply(extract_ticket_refs)
        
        # BUG #1 FIX: Do NOT sort by created_time here!
        # The RAG engine loads the CSV without sorting, and embeddings are computed
        # in that order. Sorting here creates index misalignment between self.df
        # and self.rag_engine.embeddings, causing wrong embeddings to be retrieved.
        # The RAG engine's dataset ordering is the source of truth for embeddings.
        
        return df
    
    def _compute_embeddings(self) -> np.ndarray:
        """Compute or load cached embeddings."""
        cache_path = f"data/{self.project}/embeddings_cache.npy"
        texts = title_desc_texts(self.df)

        if os.path.exists(cache_path):
            print(f"   Loading cached embeddings from {cache_path}")
            cached = np.load(cache_path, allow_pickle=True).item()
            if cache_is_valid(cached, self.df, self.sbert_model_name, texts):
                print("   Cache valid (model/rows/order/text match)")
                return cached["embeddings"]
            print(f"   ⚠️ Cache stale (model/rows/order/text changed) - recomputing")

        embeddings = self.sbert.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

        # Cache for future use, fingerprinted so a reordered dataframe can never
        # silently reuse it (see core.utils.cache_is_valid).
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, {
            "embeddings": embeddings,
            "n_rows": len(self.df),
            "model_name": self.sbert_model_name,
            "fingerprint": embeddings_fingerprint(self.df["patch_id"], texts),
        })

        return embeddings
    
    def _compute_discussion_embeddings(self) -> np.ndarray:
        """Compute discussion text embeddings for discussion similarity features.
        
        Bug A fix: Now computes real discussion embeddings instead of reusing title+desc.
        """
        discussion_texts = self.df["discussion_text"].tolist()
        
        # Check cache for discussion embeddings
        cache_path = f"data/{self.project}/embeddings_cache.npy"
        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True).item()
            if ("discussion_embeddings" in cached
                    and cached.get("discussion_fingerprint")
                    == embeddings_fingerprint(self.df["patch_id"], discussion_texts)):
                print(f"   Loading cached discussion embeddings from {cache_path}")
                return cached["discussion_embeddings"]
            print(f"   ⚠️ Discussion embeddings not in cache or stale - computing")
        
        # Compute discussion embeddings
        print(f"   Computing discussion embeddings for {len(discussion_texts)} patches...")
        discussion_embeddings = self.sbert.encode(
            discussion_texts, 
            batch_size=64, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        # Update cache to include discussion embeddings
        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True).item()
            cached["discussion_embeddings"] = discussion_embeddings
            cached["discussion_fingerprint"] = embeddings_fingerprint(
                self.df["patch_id"], discussion_texts)
            np.save(cache_path, cached)
            print(f"   Cached discussion embeddings to {cache_path}")
        else:
            # Create new cache with both embeddings
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, {
                "embeddings": self.embeddings,
                "discussion_embeddings": discussion_embeddings,
                "n_rows": len(self.df),
                "model_name": self.sbert_model_name,
                "fingerprint": embeddings_fingerprint(
                    self.df["patch_id"], title_desc_texts(self.df)),
                "discussion_fingerprint": embeddings_fingerprint(
                    self.df["patch_id"], discussion_texts),
            })
            print(f"   Created new cache with discussion embeddings at {cache_path}")
        
        return discussion_embeddings
    
    def _get_time_window_candidates(self, anchor_idx: int, window_days: int) -> List[int]:
        """Get candidates within time window."""
        anchor_time = self.df.iloc[anchor_idx]["created_time"]
        start = anchor_time - timedelta(days=window_days)
        end = anchor_time + timedelta(days=window_days)
        
        mask = (self.df["created_time"] >= start) & (self.df["created_time"] <= end)
        candidates = self.df.index[mask].tolist()
        
        return [idx for idx in candidates if idx != anchor_idx]
    
    def _build_features(self, anchor_idx: int, candidate_idx: int) -> Dict[str, float]:
        """Build pairwise features for ML model.

        Delegates to core.utils.build_pair_features -- the SAME function the
        trainer used. Do not reimplement features here: a feature present in
        the saved model but missing here is silently zero-filled by
        reindex(fill_value=0), which is how 4 of 17 features died last time.
        """
        return build_pair_features(
            self.df, self.embeddings, self.discussion_embeddings,
            anchor_idx, candidate_idx,
        )

    def predict_ml_stage(
        self,
        patch_ref: Dict[str, Any],
        top_k: int = 20,
        window_days: int = None,
        return_all_candidates: bool = False,
    ) -> List[Dict[str, Any]]:
        """Stage 1: ML model prediction.
        
        Args:
            patch_ref: Query patch dict
            top_k: Number of candidates to retrieve
            window_days: Time window (uses model's eval_window_days by default)
            return_all_candidates: If True, return all candidates with scores (for union merging)
        
        Returns:
            List of candidates with ML scores
        """
        # Issue 1 fix: Explicit window handling - default to training window for fair comparison
        # The eval_window_days (±2) was causing ML→RAG to miss candidates that RAG-only could find
        window_days = window_days or self.window_days  # Use training window by default
        
        # Find anchor index
        patch_id = patch_ref["patch_id"]
        if patch_id not in self.id_to_idx:
            self._log(f"⚠️ Patch {patch_id} not found in dataset")
            return []
        
        anchor_idx = self.id_to_idx[patch_id]
        
        # Get time-window candidates
        candidate_idxs = self._get_time_window_candidates(anchor_idx, window_days)
        
        if not candidate_idxs:
            self._log(f"⚠️ No candidates within ±{window_days} days")
            return []
        
        self._log(f"   ML Stage: {len(candidate_idxs)} candidates in time window")
        
        # Build features for all candidates
        features_list = []
        for cand_idx in candidate_idxs:
            feats = self._build_features(anchor_idx, cand_idx)
            features_list.append(feats)
        
        X = pd.DataFrame(features_list)
        # reindex(fill_value=0) would silently zero-fill any feature the model
        # expects but this pipeline does not compute -- that is a train/serve
        # skew bug, not a missing value. Fail loudly instead.
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"Model expects {len(missing)} feature(s) this pipeline does not "
                f"compute: {missing}. Fix core.utils.build_pair_features (shared "
                f"with training) or retrain the model."
            )
        X = X.reindex(columns=self.feature_cols, fill_value=0)
        
        # ML model predictions
        scores = self.model.predict(X)
        
        # Rank by ML score
        ranked_indices = np.argsort(-scores)
        
        results = []
        # Store all scores for global normalization (Issue 4 fix)
        all_ml_scores = []
        for rank, idx in enumerate(ranked_indices, 1):
            cand_idx = candidate_idxs[idx]
            row = self.df.iloc[cand_idx]
            
            candidate = {
                "patch_id": row["patch_id"],
                "ml_score": float(scores[idx]),
                "ml_rank": rank,
                "title": row["title"],
                "description": row["description"],
                "created_time": row["created_time"],
                "files": row["files_parsed"],
                "idx": cand_idx,
            }
            all_ml_scores.append(candidate)
            if rank <= top_k:
                results.append(candidate)
        
        # Return all candidates if requested (for union merge in Issue 2 fix)
        if return_all_candidates:
            return all_ml_scores
        return results
    
    def predict_rag_refine(
        self,
        patch_ref: Dict[str, Any],
        ml_candidates: List[Dict[str, Any]],
        top_k: int = 5,
        ml_weight: float = 0.5,
        rag_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Stage 2: RAG reranking of ML candidates.
        
        RAG independently scores candidates, then blends with ML scores.
        
        BUG #2 FIX: Now blends ML and RAG scores instead of overriding ML ranking.
        This preserves the learned signals from the ML model (ticket refs, discussion
        similarity, time delta, author/reviewer overlap) while incorporating RAG's
        semantic and file-overlap signals.
        
        Args:
            patch_ref: Query patch dict
            ml_candidates: Candidates from ML stage
            top_k: Final number of results
            ml_weight: Weight for ML score in final ranking (default 0.5)
            rag_weight: Weight for RAG score in final ranking (default 0.5)
        
        Returns:
            Reranked candidates by combined ML+RAG score
        """
        if not ml_candidates:
            return []
        
        self._log(f"   RAG Stage: Reranking top {len(ml_candidates)} ML candidates")
        
        # Encode query once
        query_text = f"{patch_ref.get('title', '')} {patch_ref.get('description', '')}"
        encoder = self.rag_engine._get_encoder()
        query_emb = encoder.encode([query_text], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Get project embeddings
        project = self.project
        if project not in self.rag_engine.embeddings:
            self._warn_once(
                f"no_embeddings_{project}",
                f"   ⚠️ No pre-computed embeddings for {project}",
            )
            project_embeddings = None
        else:
            project_embeddings = self.rag_engine.embeddings[project]
        
        # BUG #1 FIX: Build patch_id → index mapping for RAG engine's embeddings
        # This ensures we look up embeddings by patch_id rather than assuming
        # indices align across independently-ordered dataframes.
        rag_id_to_idx = {}
        if project in self.rag_engine.datasets:
            rag_df = self.rag_engine.datasets[project]
            rag_id_to_idx = {str(pid): i for i, pid in enumerate(rag_df["patch_id"])}
        
        # RAG scores each candidate independently
        for cand in ml_candidates:
            # Get semantic score
            # BUG #1 FIX: Look up embedding by patch_id, not by index
            cand_patch_id = cand.get("patch_id")
            cand_idx = rag_id_to_idx.get(cand_patch_id) if rag_id_to_idx else None
            
            if project_embeddings is not None and cand_idx is not None and cand_idx < len(project_embeddings):
                cand_emb = project_embeddings[cand_idx]
                cand_emb_norm = cand_emb / np.linalg.norm(cand_emb)
                semantic_score = float(np.dot(query_emb.flatten(), cand_emb_norm))
            else:
                # Fallback: encode candidate text
                cand_text = f"{cand['title']} {cand['description']}"
                cand_emb = encoder.encode([cand_text], convert_to_numpy=True)
                semantic_score = float(cosine_similarity(query_emb, cand_emb)[0][0])
            
            # File overlap score
            file_stats = get_path_similarity_stats(
                cand.get("files", []),
                patch_ref.get("files", [])
            )
            file_score = file_stats.get("jaccard", 0.0)
            
            # RAG score (70% semantic + 30% file)
            cand["rag_semantic"] = semantic_score
            cand["rag_file"] = file_score
            cand["rag_score"] = 0.7 * semantic_score + 0.3 * file_score
        
        # BUG #2 FIX: Normalize ML scores for blending
        # ML scores can have arbitrary range, so normalize to [0, 1] before blending
        ml_scores = [c["ml_score"] for c in ml_candidates]
        ml_min, ml_max = min(ml_scores), max(ml_scores)
        ml_range = ml_max - ml_min if ml_max > ml_min else 1.0
        
        rag_scores = [c["rag_score"] for c in ml_candidates]
        rag_min, rag_max = min(rag_scores), max(rag_scores)
        rag_range = rag_max - rag_min if rag_max > rag_min else 1.0
        
        # Compute blended score
        for cand in ml_candidates:
            norm_ml = (cand["ml_score"] - ml_min) / ml_range
            norm_rag = (cand["rag_score"] - rag_min) / rag_range
            cand["blended_score"] = ml_weight * norm_ml + rag_weight * norm_rag
            cand["norm_ml_score"] = norm_ml
            cand["norm_rag_score"] = norm_rag
        
        # Rerank by blended score (BUG #2 FIX: was sorting by rag_score only)
        ml_candidates.sort(key=lambda x: x["blended_score"], reverse=True)
        
        # Add final ranks
        for i, cand in enumerate(ml_candidates[:top_k], 1):
            cand["final_rank"] = i
        
        return ml_candidates[:top_k]
    
    # ------------------------------------------------------------------
    # Stage 3 (optional): LLM rerank
    # ------------------------------------------------------------------

    def _llm_cache_key(self, patch_ref: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
        payload = json.dumps({
            "model": self.llm_model,
            "llm_top_k": self.llm_top_k,
            "target": str(patch_ref.get("patch_id")),
            "ids": [str(c.get("patch_id")) for c in candidates],
        }, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()

    def _llm_load_cache(self) -> None:
        if not self.llm_cache_path or not os.path.exists(self.llm_cache_path):
            return
        try:
            with open(self.llm_cache_path, encoding="utf-8") as fh:
                self.llm_cache = json.load(fh)
            self._log(f"   LLM cache: {len(self.llm_cache)} entries from {self.llm_cache_path}")
        except (OSError, json.JSONDecodeError) as exc:
            self._warn_once("llm_cache", f"   ⚠️ Could not read LLM cache ({exc}) - starting empty")

    def _llm_save_cache(self) -> None:
        if not self.llm_cache_path or not self.llm_cache:
            return
        out_dir = os.path.dirname(self.llm_cache_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.llm_cache_path, "w", encoding="utf-8") as fh:
            json.dump(self.llm_cache, fh)

    def _get_llm_client(self):
        """Build the OpenAI-compatible client lazily.

        Works against any OpenAI-compatible endpoint via ``--llm-base-url``
        (Ollama, llama.cpp, vLLM, LM Studio), so the stage is usable without
        internet access.
        """
        if self._llm_client is not None:
            return self._llm_client
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "--llm-rerank needs the 'openai' package: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not self.llm_base_url:
            raise RuntimeError(
                "--llm-rerank needs OPENAI_API_KEY, or --llm-base-url for a local endpoint"
            )

        kwargs: Dict[str, Any] = {"api_key": api_key or "not-needed"}
        if self.llm_base_url:
            kwargs["base_url"] = self.llm_base_url
        self._llm_client = OpenAI(**kwargs)
        return self._llm_client

    def predict_llm_rerank(
        self,
        patch_ref: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Optional Stage 3: rerank with an LLM.

        Only the first ``self.llm_top_k`` candidates are sent to the model;
        the rest keep their existing order and stay below the reranked block,
        so the full candidate list is preserved and Recall@top_k is untouched
        for every k.

        Returns a **permutation** of *candidates*. Any failure (no key, no
        network, bad JSON) returns the input order untouched, so the stage can
        only ever change ranking order -- never the retrieved set.

        That means Recall@top_k is mathematically invariant here, exactly as for
        the ML+RAG blend. Only MRR and Recall@1..top_k-1 can move. To measure
        what the LLM actually contributes, compare against ``--rag-weight 0``
        on the same query set.
        """
        if not self.llm_enabled or not candidates:
            return candidates

        head_n = len(candidates) if self.llm_top_k <= 0 else min(self.llm_top_k, len(candidates))
        head, tail = candidates[:head_n], candidates[head_n:]

        cache_key = self._llm_cache_key(patch_ref, head)
        cached_order = self.llm_cache.get(cache_key)
        if cached_order:
            by_id = {str(c["patch_id"]): c for c in candidates}
            if all(pid in by_id for pid in cached_order):
                self._llm_stats["cache_hits"] += 1
                return [by_id[pid] for pid in cached_order] + tail

        try:
            client = self._get_llm_client()
        except RuntimeError as exc:
            self._llm_stats["failures"] += 1
            self._warn_once("llm_client", f"   ⚠️ LLM stage disabled: {exc}")
            self.llm_enabled = False
            return candidates

        target = {
            "patch_id": patch_ref.get("patch_id"),
            "title": patch_ref.get("title"),
            "description": patch_ref.get("description"),
            "files": patch_ref.get("files") or [],
        }
        enriched = [
            {**c, "description": c.get("description", "")}
            for c in head
        ]

        try:
            reranked = rerank_candidates(
                client, target, enriched,
                model_name=self.llm_model,
                timeout=self.llm_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            self._llm_stats["failures"] += 1
            self._warn_once(f"llm_fail_{type(exc).__name__}",
                            f"   ⚠️ LLM rerank failed ({exc}) - keeping ML+RAG order")
            return candidates

        self._llm_stats["calls"] += 1
        before = [str(c["patch_id"]) for c in head]
        after = [str(c["patch_id"]) for c in reranked]
        if before != after:
            self._llm_stats["reordered"] += 1

        # cache only the id order -- the payloads change per run
        self.llm_cache[cache_key] = after
        return reranked + tail

    def llm_stats(self) -> Dict[str, int]:
        """Counters for the LLM stage (call/cache/failure/reorder counts)."""
        return dict(self._llm_stats)

    def predict(
        self,
        patch_ref: Dict[str, Any],
        top_k: int = 10,
        window_days: int = None,
    ) -> List[Dict[str, Any]]:
        """Full ML → RAG pipeline prediction.
        
        ML retrieves top-k candidates, then RAG reranks them using its own scoring.
        
        Args:
            patch_ref: Query patch dict
            top_k: Final number of results
            window_days: Time window for candidates (defaults to self.window_days)
        
        Returns:
            Final ranked candidates by RAG score
        """
        self._log(f"\n🔍 ML → RAG Pipeline for patch {patch_ref['patch_id']}")
        self._log(f"   Title: {patch_ref.get('title', '')[:60]}...")
        
        window_days = window_days or self.window_days
        
        # Stage 1: ML prediction
        self._log(f"\n📊 Stage 1: ML Model Prediction (window=±{window_days} days)...")
        ml_candidates_all = self.predict_ml_stage(
            patch_ref,
            top_k=len(self.df),
            window_days=window_days,
        )
        
        if not ml_candidates_all:
            return []
        
        # Compute global ML score range for normalization
        all_ml_scores = [cand["ml_score"] for cand in ml_candidates_all]
        ml_score_range = (min(all_ml_scores), max(all_ml_scores))
        
        # Take top-k for refinement
        ml_candidates = ml_candidates_all[:top_k]
        
        self._log("   Top 5 ML candidates:")
        for i, cand in enumerate(ml_candidates[:5], 1):
            self._log(f"      {i}. [{cand['patch_id']}] ML Score: {cand['ml_score']:.4f}")
            self._log(f"         {cand['title'][:60]}...")
        
        # Stage 2: RAG refinement
        self._log("\n🎯 Stage 2: RAG Reranking...")
        final_results = self.predict_rag_refine(
            patch_ref,
            ml_candidates,
            top_k=top_k,
            ml_weight=self.ml_weight,
            rag_weight=self.rag_weight,
        )
        
        # Stage 3 (optional): LLM rerank of the blended top-k
        if self.llm_enabled and final_results:
            final_results = self.predict_llm_rerank(
                patch_ref, final_results, top_k=top_k,
            )
            for rank, cand in enumerate(final_results, 1):
                cand["final_rank"] = rank
            self._log("\n🤖 LLM-Reranked Results:")

        self._log("\n✅ Final Results (ML → RAG blended):")
        for i, cand in enumerate(final_results, 1):
            self._log(f"   {i}. [{cand['patch_id']}]")
            self._log(f"      Blended: {cand['blended_score']:.4f} (ML: {cand['norm_ml_score']:.4f}, RAG: {cand['norm_rag_score']:.4f})")
            self._log(f"      RAG: {cand['rag_score']:.4f} (semantic: {cand['rag_semantic']:.4f}, file: {cand['rag_file']:.4f})")
            self._log(f"      ML: {cand['ml_score']:.4f} (original rank {cand['ml_rank']})")
            self._log(f"      {cand['title'][:60]}...")
        
        return final_results


def evaluate_pipeline(
    pipeline: MLToRAGPipeline,
    ground_truth_path: str,
    window_days: int = 14,
    top_k: int = 10,
    verbose: bool = False,
    sample_ratio: float = 1.0,
    random_seed: int = 42,
    details_out: str = None,
    progress_every: int = 100,
    llm_limit: int = 0,
):
    """Evaluate the ML → RAG pipeline on ground truth.

    Quiet by default: this sweeps every ground-truth source, so per-query
    tracing produces ~17 lines per query (tens of thousands of lines). You get
    a progress line with running MRR, throughput and ETA instead. Pass
    ``details_out`` to keep the per-query numbers as CSV for later analysis.

    Reachability: a ground-truth target that falls OUTSIDE ``±window_days`` of
    its source cannot be retrieved, so scoring such a query as a miss would
    measure the window, not the model. Targets are therefore filtered to those
    inside the window, and sources left with none are excluded from the
    population entirely. This matches the trainer's convention in
    ``core/train_patch_ranker.py``::

        positive_set = positives_by_anchor[anchor] & set(candidate_idxs)
        if not positive_set:
            continue
        recall = hits / len(positive_set)

    so the two Recall@k figures are directly comparable. On the onap dataset
    that is 1776 of 3512 sources at ±2 days -- without this filter the ±2d
    Recall@10 is capped at 0.5057 no matter how good the ranking is.

    Args:
        pipeline: MLToRAGPipeline instance
        ground_truth_path: Path to ground truth CSV
        window_days: Time window for candidates, and the reachability bound
        top_k: Number of results to evaluate
        verbose: Print one line per query (debugging small samples only)
        sample_ratio: Fraction of dataset to evaluate (0.0-1.0, default 1.0 = all)
        random_seed: Random seed for reproducible sampling
        details_out: Optional CSV path for per-query rows
        progress_every: Queries between progress lines
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 70)
    print("EVALUATION: ML → RAG Pipeline")
    print("=" * 70)
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path, dtype=str)
    
    # Handle column names
    if 'source_patch_id' in gt_df.columns:
        gt_pairs = list(zip(gt_df['source_patch_id'], gt_df['target_patch_id']))
    else:
        cols = list(gt_df.columns[:2])
        gt_pairs = [(str(r[cols[0]]), str(r[cols[1]])) for _, r in gt_df.iterrows()]
    
    print(f"\n📋 Loaded {len(gt_pairs)} ground truth pairs")
    
    # Build lookup: source -> [targets]
    from collections import defaultdict
    gt_by_source = defaultdict(set)
    for src, tgt in gt_pairs:
        gt_by_source[src].add(tgt)
    
    # Sample a portion of the dataset if sample_ratio < 1.0
    if sample_ratio < 1.0:
        import random
        random.seed(random_seed)
        all_sources = list(gt_by_source.keys())
        sample_size = max(1, int(len(all_sources) * sample_ratio))
        sampled_sources = set(random.sample(all_sources, sample_size))
        gt_by_source = {k: v for k, v in gt_by_source.items() if k in sampled_sources}
        print(f"📊 Sampled {len(gt_by_source)}/{len(all_sources)} sources ({sample_ratio*100:.1f}%) with seed={random_seed}")
    
    # patch_id -> created_time, for the reachability filter
    created_lookup = dict(zip(pipeline.df["patch_id"], pipeline.df["created_time"]))
    window_span = timedelta(days=window_days)

    # Evaluate with Recall@k for k = 1, 2, 4, 6, 8, 10
    reciprocal_ranks = []
    recalls_at_k = {k: [] for k in [1, 2, 4, 6, 8, 10]}
    detail_rows = []

    total_queries = len(gt_by_source)
    started_at = time.perf_counter()

    evaluated = 0
    skipped = 0                # source id not present in the dataset (data problem)
    skipped_out_of_window = 0  # no target reachable in ±window_days (expected filtering)
    llm_queries = 0            # queries actually sent to the LLM
    llm_applied = 0
    llm_limit = llm_limit or 0
    
    for source_id, target_ids in gt_by_source.items():
        # Get source patch details
        if source_id not in pipeline.id_to_idx:
            skipped += 1
            continue

        # Keep only targets inside the window; drop the query if none survive.
        source_time = created_lookup.get(source_id)
        reachable = {
            t for t in target_ids
            if t in created_lookup
            and abs((created_lookup[t] - source_time)) <= window_span
        }
        if not reachable:
            skipped_out_of_window += 1
            continue

        source_idx = pipeline.id_to_idx[source_id]
        row = pipeline.df.iloc[source_idx]
        
        patch_ref = {
            "patch_id": source_id,
            "title": row["title"],
            "description": row["description"],
            "created_time": row["created_time"],
            "files": row["files_parsed"],
        }
        
        # Run pipeline
        use_llm = pipeline.llm_enabled and (
            llm_limit <= 0 or llm_queries < llm_limit
        )
        was_enabled = pipeline.llm_enabled
        if not use_llm:
            pipeline.llm_enabled = False
        try:
            # BUG #3 FIX: Use the top_k parameter passed to the function
            # (was hardcoded to 10, ignoring the parameter)
            results = pipeline.predict(
                patch_ref,
                top_k=top_k,
                window_days=window_days,
            )
            llm_applied = int(pipeline.llm_enabled and was_enabled)
        except Exception as e:
            print(f"   ⚠️ Error processing {source_id}: {e}", file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            skipped += 1
            continue
        finally:
            pipeline.llm_enabled = was_enabled
            if llm_applied:
                llm_queries += 1
        
        # Compute metrics against the in-window targets only
        predicted_ids = [r["patch_id"] for r in results]
        
        # MRR
        rr = 0.0
        for rank, pid in enumerate(predicted_ids, 1):
            if pid in reachable:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        
        # Recall@k for k = 1, 2, 4, 6, 8, 10
        query_recalls = {}
        for k in recalls_at_k.keys():
            top_k_pred = set(r["patch_id"] for r in results[:k])
            recall = len(top_k_pred & reachable) / len(reachable)
            recalls_at_k[k].append(recall)
            query_recalls[k] = recall
        
        # Per-query detail: stdout only when asked, always to CSV when requested
        if verbose:
            recall_str = " | ".join([f"@{k}:{query_recalls[k]:.2f}" for k in [1, 2, 4, 6, 8, 10]])
            print(f"   [{source_id}] targets={len(target_ids)} | {recall_str}")

        detail_rows.append({
            "source_patch_id": source_id,
            "n_targets_total": len(target_ids),
            "n_targets_reachable": len(reachable),
            "reciprocal_rank": rr,
            "n_returned": len(results),
            "llm_applied": int(llm_applied),
            **{f"recall_at_{k}": query_recalls[k] for k in query_recalls},
        })

        evaluated += 1
        if evaluated % progress_every == 0 or evaluated == total_queries:
            elapsed = time.perf_counter() - started_at
            rate = evaluated / elapsed if elapsed > 0 else 0.0
            eta_min = (total_queries - evaluated - skipped_out_of_window) / rate / 60 if rate > 0 else float("nan")
            running_mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            print(f"   [{evaluated:>5}/{total_queries}]  MRR={running_mrr:.4f}  "
                  f"{rate:5.2f} q/s  eta={eta_min:6.1f} min  "
                  f"out-of-window={skipped_out_of_window}", flush=True)
    
    # Aggregate results
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_recalls = {k: np.mean(v) for k, v in recalls_at_k.items()}
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n📊 Coverage")
    print(f"   ground-truth pairs          : {len(gt_pairs)}")
    print(f"   unique source queries       : {total_queries}")
    print(f"   source not in dataset       : {skipped}")
    print(f"   no target within ±{window_days}d      : {skipped_out_of_window}")
    print(f"   → evaluated (reachable)     : {evaluated}")

    if evaluated < total_queries:
        print(f"\n   Recall is measured over the {evaluated} queries whose linked patch is")
        print(f"   actually retrievable within ±{window_days}d. The {skipped_out_of_window} excluded")
        print(f"   queries cannot be answered at this window regardless of model quality.")

    print(f"\n🎯 Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"\n📈 Recall@k  (over {evaluated} evaluated queries)")
    for k in [1, 2, 4, 6, 8, 10]:
        print(f"   @{k}: {mean_recalls[k]:.4f}")

    if pipeline.llm_enabled or pipeline.llm_stats()["calls"]:
        st = pipeline.llm_stats()
        print(f"\n🤖 LLM stage ({pipeline.llm_model})")
        print(f"   API calls: {st['calls']}  cache hits: {st['cache_hits']}  "
              f"failures: {st['failures']}  orderings changed: {st['reordered']}")
        print(f"   sent to LLM: {llm_queries} queries"
              + (f" (capped by --llm-limit {llm_limit})" if llm_limit else ""))

    elapsed = time.perf_counter() - started_at
    print(f"\n⏱️  {elapsed/60:.1f} min total ({evaluated/max(elapsed,1e-9):.2f} q/s)")

    pipeline._llm_save_cache()

    if details_out and detail_rows:
        out_dir = os.path.dirname(details_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(detail_rows).to_csv(details_out, index=False)
        print(f"📝 Per-query details -> {details_out} ({len(detail_rows)} rows)")

    return {
        "mrr": mrr,
        "recall": mean_recalls,
        "evaluated": evaluated,
        "skipped": skipped,
        "skipped_out_of_window": skipped_out_of_window,
        "total_queries": total_queries,
        "window_days": window_days,
        "elapsed_sec": elapsed,
        "details_out": details_out if detail_rows else None,
    }


def compare_approaches(
    pipeline: MLToRAGPipeline,
    patch_ref: Dict[str, Any],
    top_k: int = 10,
):
    """Compare ML-only, RAG-only, and ML→RAG approaches.
    
    Args:
        pipeline: MLToRAGPipeline instance
        patch_ref: Query patch dict
        top_k: Number of results
    """
    print("\n" + "=" * 70)
    print("COMPARISON: ML vs RAG vs ML→RAG")
    print("=" * 70)
    
    print(f"\n📌 Query Patch: {patch_ref['patch_id']}")
    print(f"   Title: {patch_ref['title'][:60]}...")
    
    # Issue 1 fix: Use consistent window_days for all approaches
    window_days = pipeline.window_days  # Use training window for fair comparison
    print(f"\n⏱️ Using consistent time window: ±{window_days} days for all approaches")
    
    # 1. ML-only approach
    print(f"\n1️⃣ ML-Only Approach:")
    ml_results = pipeline.predict_ml_stage(patch_ref, top_k=top_k, window_days=window_days)
    for i, r in enumerate(ml_results[:5], 1):
        print(f"   {i}. [{r['patch_id']}] Score: {r['ml_score']:.4f} - {r['title'][:50]}...")
    
    # 2. RAG-only approach (now using same window_days)
    print(f"\n2️⃣ RAG-Only Approach:")
    rag_results = pipeline.rag_engine.predict(
        pipeline.project,
        patch_ref,
        top_k=top_k,
        window_days=window_days,  # Issue 1 fix: Use consistent window
    )
    for i, r in enumerate(rag_results[:5], 1):
        print(f"   {i}. [{r['patch_id']}] Score: {r['score']:.4f} - {r['title'][:50]}...")
    
    # 3. ML → RAG approach
    print(f"\n3️⃣ ML → RAG Approach:")
    combined_results = pipeline.predict(
        patch_ref,
        top_k=top_k,
        window_days=window_days,
    )
    
    # Compare overlap
    ml_ids = set(r['patch_id'] for r in ml_results)
    rag_ids = set(r['patch_id'] for r in rag_results)
    combined_ids = set(r['patch_id'] for r in combined_results)
    
    print(f"\n📊 Overlap Analysis:")
    print(f"   ML ∩ RAG: {len(ml_ids & rag_ids)} patches")
    print(f"   ML ∩ Combined: {len(ml_ids & combined_ids)} patches")
    print(f"   RAG ∩ Combined: {len(rag_ids & combined_ids)} patches")
    
    print(f"\n   Unique to ML: {len(ml_ids - rag_ids - combined_ids)}")
    print(f"   Unique to RAG: {len(rag_ids - ml_ids - combined_ids)}")
    print(f"   RAG rescued (in combined but not ML top-{top_k}): {len(combined_ids - ml_ids)}")


def main():
    parser = argparse.ArgumentParser(description="Test ML → RAG Pipeline")
    parser.add_argument("--project", default="onap", help="Project name")
    parser.add_argument("--model", default=None, help="Path to ML model")
    parser.add_argument("--csv", default=None, help="Path to candidates CSV")
    parser.add_argument("--patch-id", help="Specific patch ID to test")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on ground truth")
    parser.add_argument("--ground-truth", default=None, help="Path to ground truth CSV")
    parser.add_argument("--compare", action="store_true", help="Compare ML vs RAG vs ML→RAG")
    parser.add_argument("--window-days", type=int, default=14, help="Time window in days")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of dataset to evaluate (0.0-1.0, default 1.0 = all)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the full per-query trace (debugging small samples only; a full sweep is ~60k lines)")
    parser.add_argument("--quiet", action="store_true",
                        help="Silence per-query tracing (default for --evaluate; progress + summary still shown)")
    parser.add_argument("--details-out", default=None,
                        help="CSV path for per-query rows (source, RR, recall@k) from --evaluate")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Queries between progress lines during --evaluate")
    parser.add_argument("--ml-weight", type=float, default=0.5,
                        help="Weight of the ML score in the blended rerank (default 0.5)")
    parser.add_argument("--rag-weight", type=float, default=0.5,
                        help="Weight of the RAG score in the blended rerank; 0 = pure ML baseline")

    llm_grp = parser.add_argument_group(
        "LLM rerank (optional Stage 3)",
        "Permutes the ML+RAG top-k. Cannot change which patches are retrieved, "
        "so Recall@top_k is invariant; only ordering (MRR, Recall@1..k-1) can improve.",
    )
    llm_grp.add_argument("--llm-rerank", action="store_true",
                         help="Enable the LLM rerank stage (off by default)")
    llm_grp.add_argument("--llm-top-k", type=int, default=10,
                         help="How many top candidates the LLM sees (default 10). "
                              "Remaining candidates keep their order below the reranked block. 0 = all")
    llm_grp.add_argument("--llm-model", default="gpt-4o-mini",
                         help="Chat model id, or a local model name with --llm-base-url")
    llm_grp.add_argument("--llm-base-url", default=None,
                         help="OpenAI-compatible base URL (Ollama, llama.cpp, vLLM) - no API key needed")
    llm_grp.add_argument("--llm-timeout", type=float, default=60.0,
                         help="Per-request timeout in seconds (default: 60)")
    llm_grp.add_argument("--llm-cache", default=None,
                         help="JSON file caching LLM orderings so re-runs cost nothing")
    llm_grp.add_argument("--llm-limit", type=int, default=0,
                         help="Only send the first N evaluated queries to the LLM (0 = no limit). "
                              "Use a small value to smoke-test before a full paid sweep.")
    
    args = parser.parse_args()
    
    # Determine paths
    csv_path = args.csv or f"data/{args.project}/all_candidates.csv"
    model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"   Please train a model first or specify --model path")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        sys.exit(1)
    
    # Initialize pipeline
    # Batch mode is quiet by default; single-patch debugging stays verbose.
    verbose = not args.quiet and (args.verbose or not args.evaluate)
    pipeline = MLToRAGPipeline(
        model_path=model_path,
        project=args.project,
        csv_path=csv_path,
        verbose=verbose,
    )
    pipeline.ml_weight = args.ml_weight
    pipeline.rag_weight = args.rag_weight
    if args.ml_weight + args.rag_weight <= 0:
        sys.exit("❌ --ml-weight and --rag-weight cannot both be 0")

    pipeline.llm_enabled = args.llm_rerank
    pipeline.llm_model = args.llm_model
    pipeline.llm_top_k = args.llm_top_k
    pipeline.llm_base_url = args.llm_base_url
    pipeline.llm_timeout = args.llm_timeout
    pipeline.llm_cache_path = args.llm_cache
    if args.llm_rerank:
        pipeline._llm_load_cache()
        key_state = "OPENAI_API_KEY set" if os.environ.get("OPENAI_API_KEY") else "no API key"
        view = "all" if args.llm_top_k <= 0 else str(args.llm_top_k)
        print(f"   LLM rerank: ON ({args.llm_model}, top-{view}, {key_state}"
              f"{', ' + args.llm_base_url if args.llm_base_url else ''})")
    if args.evaluate:
        print(f"   per-query tracing: {'ON' if args.verbose else 'OFF'}")
    
    # Run requested test
    if args.patch_id:
        # Get patch details
        if args.patch_id not in pipeline.id_to_idx:
            print(f"❌ Patch {args.patch_id} not found in dataset")
            sys.exit(1)
        
        idx = pipeline.id_to_idx[args.patch_id]
        row = pipeline.df.iloc[idx]
        
        patch_ref = {
            "patch_id": args.patch_id,
            "title": row["title"],
            "description": row["description"],
            "created_time": row["created_time"],
            "files": row["files_parsed"],
        }
        
        if args.compare:
            compare_approaches(pipeline, patch_ref, top_k=args.top_k)
        else:
            results = pipeline.predict(
                patch_ref,
                top_k=args.top_k,
                window_days=args.window_days,
            )
    
    elif args.evaluate:
        gt_path = args.ground_truth or f"data/{args.project}/ground_truth.csv"
        
        if not os.path.exists(gt_path):
            print(f"❌ Ground truth not found: {gt_path}")
            sys.exit(1)
        
        evaluate_pipeline(
            pipeline,
            gt_path,
            window_days=args.window_days,
            top_k=args.top_k,
            verbose=args.verbose,
            sample_ratio=args.sample_ratio,
            random_seed=args.random_seed,
            details_out=args.details_out,
            progress_every=args.progress_every,
            llm_limit=args.llm_limit,
        )
    
    else:
        # Default: show usage examples
        print("\n" + "=" * 70)
        print("ML → RAG Pipeline Test")
        print("=" * 70)
        print("\nUsage examples:")
        print("\n  # Test specific patch:")
        print(f"  python {sys.argv[0]} --project onap --patch-id <patch_id>")
        print("\n  # Compare approaches:")
        print(f"  python {sys.argv[0]} --project onap --patch-id <patch_id> --compare")
        print("\n  # Evaluate on ground truth (quiet; progress + summary only):")
        print(f"  python {sys.argv[0]} --project onap --evaluate")
        print("\n  # Evaluate, keeping per-query numbers as CSV:")
        print(f"  python {sys.argv[0]} --project onap --evaluate --details-out /kaggle/working/eval_details.csv")
        print("\n  # Evaluate small sample WITH the full per-query trace:")
        print(f"  python {sys.argv[0]} --project onap --evaluate --sample-ratio 0.01 --verbose")
        print("\n  # Evaluate on 30%% of dataset (random sample):")
        print(f"  python {sys.argv[0]} --project onap --evaluate --sample-ratio 0.3")
        print("\n  # Custom parameters:")
        print(f"  python {sys.argv[0]} --project onap --patch-id <id> --top-k 20")
        
        # Show first patch as example
        print("\n" + "-" * 70)
        print("Running example with first patch in dataset...")
        first_idx = 0
        row = pipeline.df.iloc[first_idx]
        
        patch_ref = {
            "patch_id": row["patch_id"],
            "title": row["title"],
            "description": row["description"],
            "created_time": row["created_time"],
            "files": row["files_parsed"],
        }
        
        compare_approaches(pipeline, patch_ref, top_k=args.top_k)


if __name__ == "__main__":
    main()
