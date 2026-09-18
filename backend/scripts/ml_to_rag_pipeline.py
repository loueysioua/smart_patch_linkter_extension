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
import sys
import os
from datetime import timedelta
from typing import Any, Dict, List
import json

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from core.improved_rag_engine import ImprovedRAGEngine
from core.utils import get_path_similarity_stats


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
    ):
        """Initialize the pipeline.
        
        Args:
            model_path: Path to trained LightGBM model (.pkl)
            project: Project name
            csv_path: Path to candidate CSV
            rag_engine: Optional pre-loaded RAG engine
        """
        self.project = project
        self.model_data = joblib.load(model_path)
        self.model = self.model_data["model"]
        self.feature_cols = self.model_data["feature_cols"]
        self.window_days = self.model_data.get("window_days", 14)
        self.sbert_model_name = self.model_data.get("sbert_model_name", "all-MiniLM-L6-v2")
        
        print(f"📦 Loaded ML model from {model_path}")
        print(f"   Window: ±{self.window_days} days")
        print(f"   Features: {len(self.feature_cols)}")
        
        # Load dataset
        print(f"\n📂 Loading dataset for {project}...")
        self.df = self._load_dataset(csv_path)
        print(f"   Loaded {len(self.df)} patches")
        
        # Load or use provided RAG engine
        if rag_engine:
            self.rag_engine = rag_engine
        else:
            print(f"\n🔧 Initializing RAG engine...")
            self.rag_engine = ImprovedRAGEngine(use_hybrid=True)
            self.rag_engine.load_project(project, csv_path)
        
        # Compute embeddings for ML features
        print(f"\n🧮 Computing embeddings for ML features...")
        self.sbert = SentenceTransformer(self.sbert_model_name)
        self.embeddings = self._compute_embeddings()
        self.discussion_embeddings = self._compute_discussion_embeddings()
        print(f"   Done!")
        
        # Build patch_id to index mapping
        self.id_to_idx = {str(pid): i for i, pid in enumerate(self.df["patch_id"])}
    
    def _load_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess dataset."""
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
        df = df.sort_values("created_time").reset_index(drop=True)
        
        return df
    
    def _compute_embeddings(self) -> np.ndarray:
        """Compute or load cached embeddings."""
        cache_path = f"data/{self.project}/embeddings_cache.npy"
        
        if os.path.exists(cache_path):
            print(f"   Loading cached embeddings from {cache_path}")
            cached = np.load(cache_path, allow_pickle=True).item()
            if cached.get("n_rows") == len(self.df):
                return cached["embeddings"]
        
        texts = (self.df["title"] + " " + self.df["description"]).tolist()
        embeddings = self.sbert.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        # Cache for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, {"embeddings": embeddings, "n_rows": len(self.df)})
        
        return embeddings
    
    def _compute_discussion_embeddings(self) -> np.ndarray:
        """Compute discussion text embeddings (placeholder - uses title+desc)."""
        # For simplicity, use title+description embeddings
        # In full implementation, would extract discussion from change_log/comments
        return self.embeddings
    
    def _get_time_window_candidates(self, anchor_idx: int, window_days: int) -> List[int]:
        """Get candidates within time window."""
        anchor_time = self.df.iloc[anchor_idx]["created_time"]
        start = anchor_time - timedelta(days=window_days)
        end = anchor_time + timedelta(days=window_days)
        
        mask = (self.df["created_time"] >= start) & (self.df["created_time"] <= end)
        candidates = self.df.index[mask].tolist()
        
        return [idx for idx in candidates if idx != anchor_idx]
    
    def _build_features(self, anchor_idx: int, candidate_idx: int) -> Dict[str, float]:
        """Build pairwise features for ML model."""
        row_i = self.df.iloc[anchor_idx]
        row_j = self.df.iloc[candidate_idx]
        
        # Semantic similarity
        sim_cosine = float(cosine_similarity(
            self.embeddings[anchor_idx].reshape(1, -1),
            self.embeddings[candidate_idx].reshape(1, -1)
        )[0][0])
        
        # File similarity
        file_stats = get_path_similarity_stats(
            row_i["files_parsed"],
            row_j["files_parsed"]
        )
        
        # Text similarity
        text_i = f"{row_i['title']} {row_i['description']}"
        text_j = f"{row_j['title']} {row_j['description']}"
        token_jaccard = self._token_jaccard(text_i, text_j)
        
        # Time delta
        delta_time_hours = abs(
            (row_i["created_time"] - row_j["created_time"]).total_seconds() / 3600
        )
        
        # Discussion similarity
        sim_cosine_discussion = float(cosine_similarity(
            self.discussion_embeddings[anchor_idx].reshape(1, -1),
            self.discussion_embeddings[candidate_idx].reshape(1, -1)
        )[0][0])
        
        features = {
            **file_stats,
            "sim_cosine": sim_cosine,
            "token_jaccard": token_jaccard,
            "delta_time_hours": delta_time_hours,
            "sim_cosine_discussion": sim_cosine_discussion,
            "token_jaccard_discussion": 0.0,  # Placeholder
            "has_discussion_both": 0.0,  # Placeholder
            "shares_ticket_ref": 0.0,  # Placeholder
        }
        
        return features
    
    def _token_jaccard(self, text_a: str, text_b: str) -> float:
        """Compute token Jaccard similarity."""
        import re
        TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
        
        tokens_a = set(w.lower() for w in TOKEN_RE.findall(text_a or ""))
        tokens_b = set(w.lower() for w in TOKEN_RE.findall(text_b or ""))
        
        union = tokens_a | tokens_b
        if not union:
            return 0.0
        
        return len(tokens_a & tokens_b) / len(union)
    
    def predict_ml_stage(
        self,
        patch_ref: Dict[str, Any],
        top_k: int = 20,
        window_days: int = None,
    ) -> List[Dict[str, Any]]:
        """Stage 1: ML model prediction.
        
        Args:
            patch_ref: Query patch dict
            top_k: Number of candidates to retrieve
            window_days: Time window (uses model default if None)
        
        Returns:
            List of candidates with ML scores
        """
        window_days = window_days or self.window_days
        
        # Find anchor index
        patch_id = patch_ref["patch_id"]
        if patch_id not in self.id_to_idx:
            print(f"⚠️ Patch {patch_id} not found in dataset")
            return []
        
        anchor_idx = self.id_to_idx[patch_id]
        
        # Get time-window candidates
        candidate_idxs = self._get_time_window_candidates(anchor_idx, window_days)
        
        if not candidate_idxs:
            print(f"⚠️ No candidates within ±{window_days} days")
            return []
        
        print(f"   ML Stage: {len(candidate_idxs)} candidates in time window")
        
        # Build features for all candidates
        features_list = []
        for cand_idx in candidate_idxs:
            feats = self._build_features(anchor_idx, cand_idx)
            features_list.append(feats)
        
        X = pd.DataFrame(features_list)
        X = X.reindex(columns=self.feature_cols, fill_value=0)
        
        # ML model predictions
        scores = self.model.predict(X)
        
        # Rank by ML score
        ranked_indices = np.argsort(-scores)
        
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            cand_idx = candidate_idxs[idx]
            row = self.df.iloc[cand_idx]
            
            results.append({
                "patch_id": row["patch_id"],
                "ml_score": float(scores[idx]),
                "ml_rank": rank,
                "title": row["title"],
                "description": row["description"],
                "created_time": row["created_time"],
                "files": row["files_parsed"],
                "idx": cand_idx,
            })
        
        return results
    
    def predict_rag_refine(
        self,
        patch_ref: Dict[str, Any],
        ml_candidates: List[Dict[str, Any]],
        top_k: int = 5,
        rag_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Stage 2: RAG refinement of ML candidates.
        
        Args:
            patch_ref: Query patch dict
            ml_candidates: Candidates from ML stage
            top_k: Final number of results
            rag_weight: Weight for RAG score (0.0-1.0)
        
        Returns:
            Reranked candidates with combined scores
        """
        if not ml_candidates:
            return []
        
        print(f"   RAG Stage: Refining top {len(ml_candidates)} ML candidates")
        
        # Get RAG scores for each candidate
        for cand in ml_candidates:
            # Compute RAG-style scores
            query_text = f"{patch_ref.get('title', '')} {patch_ref.get('description', '')}"
            cand_text = f"{cand['title']} {cand['description']}"
            
            # Semantic similarity (via RAG engine's FAISS)
            query_emb = self.rag_engine._get_encoder().encode([query_text], convert_to_numpy=True)
            cand_emb = self.rag_engine._get_encoder().encode([cand_text], convert_to_numpy=True)
            
            semantic_score = float(cosine_similarity(query_emb, cand_emb)[0][0])
            
            # File overlap score
            file_stats = get_path_similarity_stats(
                cand.get("files", []),
                patch_ref.get("files", [])
            )
            file_score = file_stats.get("jaccard", 0.0)
            
            # Combined RAG score (70% semantic + 30% file)
            rag_score = 0.7 * semantic_score + 0.3 * file_score
            
            cand["rag_semantic"] = semantic_score
            cand["rag_file"] = file_score
            cand["rag_score"] = rag_score
            
            # Combined score: ML + RAG
            cand["combined_score"] = (
                (1 - rag_weight) * cand["ml_score"] +
                rag_weight * rag_score
            )
        
        # Rerank by combined score
        ml_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Add final ranks
        for i, cand in enumerate(ml_candidates[:top_k], 1):
            cand["final_rank"] = i
        
        return ml_candidates[:top_k]
    
    def predict(
        self,
        patch_ref: Dict[str, Any],
        top_k: int = 5,
        window_days: int = None,
        rag_weight: float = 0.3,
        ml_candidates_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full ML → RAG pipeline prediction.
        
        Args:
            patch_ref: Query patch dict
            top_k: Final number of results
            window_days: Time window for candidates
            rag_weight: Weight for RAG refinement (0.0 = pure ML, 1.0 = pure RAG)
            ml_candidates_k: Number of ML candidates to refine
        
        Returns:
            Final ranked candidates
        """
        print(f"\n🔍 ML → RAG Pipeline for patch {patch_ref['patch_id']}")
        print(f"   Title: {patch_ref.get('title', '')[:60]}...")
        
        # Stage 1: ML prediction
        print(f"\n📊 Stage 1: ML Model Prediction...")
        ml_candidates = self.predict_ml_stage(
            patch_ref,
            top_k=ml_candidates_k,
            window_days=window_days,
        )
        
        if not ml_candidates:
            return []
        
        print(f"   Top 5 ML candidates:")
        for i, cand in enumerate(ml_candidates[:5], 1):
            print(f"      {i}. [{cand['patch_id']}] ML Score: {cand['ml_score']:.4f}")
            print(f"         {cand['title'][:60]}...")
        
        # Stage 2: RAG refinement
        print(f"\n🎯 Stage 2: RAG Refinement (weight={rag_weight})...")
        final_results = self.predict_rag_refine(
            patch_ref,
            ml_candidates,
            top_k=top_k,
            rag_weight=rag_weight,
        )
        
        print(f"\n✅ Final Results (ML → RAG):")
        for i, cand in enumerate(final_results, 1):
            print(f"   {i}. [{cand['patch_id']}]")
            print(f"      Combined: {cand['combined_score']:.4f}")
            print(f"      ML: {cand['ml_score']:.4f} (rank {cand['ml_rank']}) | RAG: {cand['rag_score']:.4f}")
            print(f"      {cand['title'][:60]}...")
        
        return final_results


def evaluate_pipeline(
    pipeline: MLToRAGPipeline,
    ground_truth_path: str,
    window_days: int = 14,
    top_k: int = 10,
    rag_weight: float = 0.3,
    verbose: bool = True,
):
    """Evaluate the ML → RAG pipeline on ground truth.
    
    Args:
        pipeline: MLToRAGPipeline instance
        ground_truth_path: Path to ground truth CSV
        window_days: Time window for candidates
        top_k: Number of results to evaluate
        rag_weight: Weight for RAG refinement
        verbose: Print detailed results
    
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
    
    # Evaluate
    reciprocal_ranks = []
    recalls_at_k = {k: [] for k in [1, 2, 5, 10]}
    
    evaluated = 0
    skipped = 0
    
    for source_id, target_ids in gt_by_source.items():
        # Get source patch details
        if source_id not in pipeline.id_to_idx:
            skipped += 1
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
        try:
            results = pipeline.predict(
                patch_ref,
                top_k=top_k,
                window_days=window_days,
                rag_weight=rag_weight,
            )
        except Exception as e:
            skipped += 1
            continue
        
        # Compute metrics
        predicted_ids = [r["patch_id"] for r in results]
        
        # MRR
        rr = 0.0
        for rank, pid in enumerate(predicted_ids, 1):
            if pid in target_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        
        # Recall@k
        for k in recalls_at_k.keys():
            top_k_pred = set(predicted_ids[:k])
            hits = len(top_k_pred & target_ids)
            recall = hits / len(target_ids) if target_ids else 0.0
            recalls_at_k[k].append(recall)
        
        evaluated += 1
        
        if verbose and evaluated % 50 == 0:
            print(f"   Evaluated {evaluated}/{len(gt_by_source)} queries...")
    
    # Aggregate results
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_recalls = {k: np.mean(v) for k, v in recalls_at_k.items()}
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n📊 Evaluated: {evaluated} queries, skipped: {skipped}")
    print(f"\n🎯 Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"\n📈 Recall@k:")
    for k, recall in sorted(mean_recalls.items()):
        print(f"   @{k}: {recall:.4f}")
    
    return {
        "mrr": mrr,
        "recall": mean_recalls,
        "evaluated": evaluated,
        "skipped": skipped,
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
    
    # 1. ML-only approach
    print(f"\n1️⃣ ML-Only Approach:")
    ml_results = pipeline.predict_ml_stage(patch_ref, top_k=top_k)
    for i, r in enumerate(ml_results[:5], 1):
        print(f"   {i}. [{r['patch_id']}] Score: {r['ml_score']:.4f} - {r['title'][:50]}...")
    
    # 2. RAG-only approach
    print(f"\n2️⃣ RAG-Only Approach:")
    rag_results = pipeline.rag_engine.predict(
        pipeline.project,
        patch_ref,
        top_k=top_k,
        window_days=pipeline.window_days,
    )
    for i, r in enumerate(rag_results[:5], 1):
        print(f"   {i}. [{r['patch_id']}] Score: {r['score']:.4f} - {r['title'][:50]}...")
    
    # 3. ML → RAG approach
    print(f"\n3️⃣ ML → RAG Approach:")
    combined_results = pipeline.predict(patch_ref, top_k=top_k, rag_weight=0.3)
    
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


def main():
    parser = argparse.ArgumentParser(description="Test ML → RAG Pipeline")
    parser.add_argument("--project", default="onap", help="Project name")
    parser.add_argument("--model", default="train3/onap/model_onap30.pkl", help="Path to ML model")
    parser.add_argument("--csv", default=None, help="Path to candidates CSV")
    parser.add_argument("--patch-id", help="Specific patch ID to test")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on ground truth")
    parser.add_argument("--ground-truth", default=None, help="Path to ground truth CSV")
    parser.add_argument("--compare", action="store_true", help="Compare ML vs RAG vs ML→RAG")
    parser.add_argument("--window-days", type=int, default=14, help="Time window in days")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--rag-weight", type=float, default=0.3, help="RAG weight in combined score")
    parser.add_argument("--ml-candidates-k", type=int, default=20, help="ML candidates to refine")
    
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
    pipeline = MLToRAGPipeline(
        model_path=model_path,
        project=args.project,
        csv_path=csv_path,
    )
    
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
                rag_weight=args.rag_weight,
                ml_candidates_k=args.ml_candidates_k,
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
            rag_weight=args.rag_weight,
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
        print("\n  # Evaluate on ground truth:")
        print(f"  python {sys.argv[0]} --project onap --evaluate")
        print("\n  # Custom parameters:")
        print(f"  python {sys.argv[0]} --project onap --patch-id <id> --rag-weight 0.5 --top-k 20")
        
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
