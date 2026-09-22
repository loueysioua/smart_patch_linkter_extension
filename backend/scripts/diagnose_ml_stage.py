"""
ML Stage Recall Diagnostics

This script diagnoses the ML stage recall issue identified in the architecture review:
- Issue 1: Window mismatch between ML and RAG stages
- Issue 2: Hard filter-then-rerank ceiling
- Issue 3: Per-candidate embedding recomputation inefficiency
- Issue 4: Narrow min-max normalization

Usage:
    python scripts/diagnose_ml_stage.py --project onap --model train3/onap/model_onap_30.pkl
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from scripts.ml_to_rag_pipeline import MLToRAGPipeline


def diagnose_ml_stage_recall(
    pipeline: MLToRAGPipeline,
    ground_truth_path: str,
    window_days: int = 14,
    ml_candidates_k: int = 20,
):
    """Diagnose ML stage recall to determine if it's the bottleneck.
    
    This answers: "Is the ML model's recall@k the ceiling for the combined pipeline?"
    """
    print("\n" + "=" * 70)
    print("ML STAGE RECALL DIAGNOSTICS")
    print("=" * 70)
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path, dtype=str)
    
    if 'source_patch_id' in gt_df.columns:
        gt_pairs = list(zip(gt_df['source_patch_id'], gt_df['target_patch_id']))
    else:
        cols = list(gt_df.columns[:2])
        gt_pairs = [(str(r[cols[0]]), str(r[cols[1]])) for _, r in gt_df.iterrows()]
    
    print(f"\n📋 Loaded {len(gt_pairs)} ground truth pairs")
    
    # Build lookup: source -> [targets]
    gt_by_source = defaultdict(set)
    for src, tgt in gt_pairs:
        gt_by_source[src].add(tgt)
    
    # Test different window sizes
    windows_to_test = [2, 7, 14, 30]
    
    print(f"\n🔬 Testing ML recall at different time windows:")
    print(f"   (Using top-{ml_candidates_k} candidates)")
    
    results_by_window = {}
    
    for test_window in windows_to_test:
        recalls_at_k = []
        mrr_scores = []
        
        evaluated = 0
        skipped = 0
        
        for source_id, target_ids in gt_by_source.items():
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
            
            try:
                # Get ML candidates with specific window
                ml_candidates = pipeline.predict_ml_stage(
                    patch_ref,
                    top_k=ml_candidates_k,
                    window_days=test_window,
                )
            except Exception as e:
                skipped += 1
                continue
            
            predicted_ids = [c["patch_id"] for c in ml_candidates]
            
            # Recall@k
            predicted_set = set(predicted_ids[:ml_candidates_k])
            hits = len(predicted_set & target_ids)
            recall = hits / len(target_ids) if target_ids else 0.0
            recalls_at_k.append(recall)
            
            # MRR
            rr = 0.0
            for rank, pid in enumerate(predicted_ids, 1):
                if pid in target_ids:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)
            
            evaluated += 1
        
        mean_recall = np.mean(recalls_at_k) if recalls_at_k else 0.0
        mean_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        
        results_by_window[test_window] = {
            "mean_recall": mean_recall,
            "mean_mrr": mean_mrr,
            "evaluated": evaluated,
            "skipped": skipped,
        }
        
        print(f"\n   Window ±{test_window} days:")
        print(f"      Recall@{ml_candidates_k}: {mean_recall:.4f}")
        print(f"      MRR: {mean_mrr:.4f}")
        print(f"      Evaluated: {evaluated}, Skipped: {skipped}")
    
    # Compare with RAG-only
    print(f"\n📊 Comparing with RAG-only at window ±{window_days} days:")
    
    rag_recalls = []
    rag_mrrs = []
    
    for source_id, target_ids in gt_by_source.items():
        if source_id not in pipeline.id_to_idx:
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
        
        try:
            rag_results = pipeline.rag_engine.retrieve_multi_query(
                project=pipeline.project,
                patch_ref=patch_ref,
                top_k=ml_candidates_k,
                time_window_days=window_days,
            )
        except Exception:
            continue
        
        predicted_ids = [c["patch_id"] for c in rag_results]
        
        # Recall@k
        predicted_set = set(predicted_ids[:ml_candidates_k])
        hits = len(predicted_set & target_ids)
        recall = hits / len(target_ids) if target_ids else 0.0
        rag_recalls.append(recall)
        
        # MRR
        rr = 0.0
        for rank, pid in enumerate(predicted_ids, 1):
            if pid in target_ids:
                rr = 1.0 / rank
                break
        rag_mrrs.append(rr)
    
    mean_rag_recall = np.mean(rag_recalls) if rag_recalls else 0.0
    mean_rag_mrr = np.mean(rag_mrrs) if rag_mrrs else 0.0
    
    print(f"   RAG-only Recall@{ml_candidates_k}: {mean_rag_recall:.4f}")
    print(f"   RAG-only MRR: {mean_rag_mrr:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    ml_recall_14 = results_by_window.get(14, {}).get("mean_recall", 0.0)
    
    print(f"\n📌 Key Finding:")
    print(f"   ML Stage Recall@{ml_candidates_k} at ±14 days: {ml_recall_14:.4f}")
    print(f"   RAG-only Recall@{ml_candidates_k} at ±14 days: {mean_rag_recall:.4f}")
    
    if ml_recall_14 < mean_rag_recall:
        print(f"\n⚠️ ML stage is the bottleneck!")
        print(f"   The ML model's recall ({ml_recall_14:.2%}) is lower than RAG ({mean_rag_recall:.2%}).")
        print(f"   This confirms Issue #2: hard filter-then-rerank caps you at ML's ceiling.")
        print(f"\n✅ Recommendation: Use union retrieval to let RAG reintroduce candidates ML missed.")
    else:
        print(f"\n✅ ML stage recall is sufficient ({ml_recall_14:.2%}).")
        print(f"   The bottleneck is likely elsewhere (RAG refinement weights, normalization).")
    
    # Check window mismatch
    ml_recall_2 = results_by_window.get(2, {}).get("mean_recall", 0.0)
    
    if ml_recall_2 < ml_recall_14 * 0.8:
        print(f"\n⚠️ Window mismatch detected!")
        print(f"   ML Recall@{ml_candidates_k} at ±2 days: {ml_recall_2:.4f}")
        print(f"   ML Recall@{ml_candidates_k} at ±14 days: {ml_recall_14:.4f}")
        print(f"   Using eval_window_days (±2) instead of training window (±14) loses significant recall.")
        print(f"\n✅ Recommendation: Always use consistent window_days across all pipeline stages.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose ML stage recall")
    parser.add_argument("--project", default="onap", help="Project name")
    parser.add_argument("--model", default="train3/onap/model_onap_30.pkl", help="Path to ML model")
    parser.add_argument("--csv", default=None, help="Path to candidates CSV")
    parser.add_argument("--ground-truth", default="data/onap/ground_truth.csv", help="Path to ground truth")
    parser.add_argument("--window-days", type=int, default=14, help="Reference time window")
    parser.add_argument("--ml-candidates-k", type=int, default=20, help="Number of ML candidates")
    
    args = parser.parse_args()
    
    csv_path = args.csv or f"data/{args.project}/all_candidates.csv"
    model_path = args.model
    gt_path = args.ground_truth
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        sys.exit(1)
    
    if not os.path.exists(gt_path):
        print(f"❌ Ground truth not found: {gt_path}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = MLToRAGPipeline(
        model_path=model_path,
        project=args.project,
        csv_path=csv_path,
    )
    
    diagnose_ml_stage_recall(
        pipeline,
        gt_path,
        window_days=args.window_days,
        ml_candidates_k=args.ml_candidates_k,
    )


if __name__ == "__main__":
    main()
