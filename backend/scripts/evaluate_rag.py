"""
Evaluation script for RAG pipeline with MRR and Recall@k metrics.

Usage:
    python evaluate_rag.py --project onap --window-days 14 --top-k 10
    
    # With custom time window
    python evaluate_rag.py --project onap --window-days 7 --top-k 10
    
    # Evaluate specific ground truth file
    python evaluate_rag.py --project onap --ground-truth datasets/onap/ground_truth.csv
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.engine import SmartPatchEngine


class RAGEvaluator:
    """Evaluator for RAG-based patch similarity detection."""
    
    def __init__(self, engine: SmartPatchEngine, project: str):
        """
        Initialize evaluator.
        
        Args:
            engine: SmartPatchEngine instance with project loaded
            project: Project name to evaluate
        """
        self.engine = engine
        self.project = project
        self.dataset = engine.datasets.get(project)
        
        if self.dataset is None:
            raise ValueError(f"Project {project} not loaded in engine")
    
    def load_ground_truth(self, gt_path: str) -> pd.DataFrame:
        """
        Load ground truth file.
        
        Args:
            gt_path: Path to ground truth CSV file
            
        Returns:
            DataFrame with source_patch_id and target_patch_id columns
        """
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        gt_df = pd.read_csv(gt_path, dtype={"source_patch_id": str, "target_patch_id": str})
        
        # Handle both possible column naming conventions
        if 'source_patch_id' not in gt_df.columns:
            if 'patch_id' in gt_df.columns:
                # Assume format: patch_id, related_patch_id
                gt_df = gt_df.rename(columns={
                    'patch_id': 'source_patch_id',
                    'related_patch_id': 'target_patch_id'
                })
        
        print(f"   -> Loaded {len(gt_df)} ground truth pairs")
        return gt_df
    
    def filter_by_time_window(self, gt_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
        """
        Filter ground truth pairs to only include pairs within time window.
        
        Args:
            gt_df: Ground truth DataFrame
            window_days: Time window in days
            
        Returns:
            Filtered DataFrame
        """
        filtered_pairs = []
        
        for _, row in gt_df.iterrows():
            source_id = str(row['source_patch_id'])
            target_id = str(row['target_patch_id'])
            
            # Get timestamps for both patches
            source_row = self.dataset[self.dataset.patch_id == source_id]
            target_row = self.dataset[self.dataset.patch_id == target_id]
            
            if source_row.empty or target_row.empty:
                continue
            
            source_time = source_row.iloc[0]['created_time']
            target_time = target_row.iloc[0]['created_time']
            
            # Check if within window
            delta_days = abs((source_time - target_time).total_seconds() / 86400)
            
            if delta_days <= window_days:
                filtered_pairs.append({
                    'source_patch_id': source_id,
                    'target_patch_id': target_id,
                    'delta_days': delta_days
                })
        
        filtered_df = pd.DataFrame(filtered_pairs)
        print(f"   -> {len(filtered_df)} pairs within {window_days}-day window (from {len(gt_df)} total)")
        
        return filtered_df
    
    def evaluate_single_query(
        self, 
        source_patch_id: str, 
        target_patch_ids: List[str],
        top_k: int = 10,
        window_days: int = 14
    ) -> Dict:
        """
        Evaluate a single query and compute metrics.
        
        Args:
            source_patch_id: Source patch ID
            target_patch_ids: List of relevant target patch IDs
            top_k: Maximum number of results to consider
            window_days: Time window for candidate retrieval
            
        Returns:
            Dictionary with metrics for this query
        """
        # Get source patch details
        source_row = self.dataset[self.dataset.patch_id == source_patch_id]
        
        if source_row.empty:
            return None
        
        row = source_row.iloc[0]
        patch_ref = {
            "patch_id": row['patch_id'],
            "title": row['title'],
            "description": row['description'],
            "created_time": row['created_time'],
            "files": row['files_parsed']
        }
        
        # Get predictions from RAG engine
        try:
            # Check if using improved engine with strategy
            if hasattr(self.engine, 'predict') and hasattr(self.engine, '__class__'):
                if self.engine.__class__.__name__ == 'ImprovedRAGEngine' and hasattr(self, 'strategy'):
                    predictions = self.engine.predict(
                        self.project, 
                        patch_ref, 
                        top_k=top_k, 
                        window_days=window_days,
                        strategy=self.strategy
                    )
                else:
                    predictions = self.engine.predict(
                        self.project, 
                        patch_ref, 
                        top_k=top_k, 
                        window_days=window_days
                    )
            else:
                predictions = self.engine.predict(
                    self.project, 
                    patch_ref, 
                    top_k=top_k, 
                    window_days=window_days
                )
        except Exception as e:
            print(f"⚠️ Error predicting for patch {source_patch_id}: {e}")
            return None
        
        # Extract predicted patch IDs
        predicted_ids = [p['patch_id'] for p in predictions]
        
        # Compute metrics
        metrics = {
            'source_patch_id': source_patch_id,
            'target_patch_ids': target_patch_ids,
            'predicted_ids': predicted_ids,
            'num_targets': len(target_patch_ids),
            'num_predictions': len(predictions)
        }
        
        # Compute Reciprocal Rank (for first relevant item found)
        rr = 0.0
        for rank, pred_id in enumerate(predicted_ids, start=1):
            if pred_id in target_patch_ids:
                rr = 1.0 / rank
                break
        
        metrics['reciprocal_rank'] = rr
        
        # Compute Recall@k for different k values
        k_values = [1, 2, 4, 6, 8, 10]
        
        for k in k_values:
            top_k_preds = set(predicted_ids[:k])
            hits = len(top_k_preds.intersection(set(target_patch_ids)))
            recall_k = hits / len(target_patch_ids) if target_patch_ids else 0.0
            metrics[f'recall@{k}'] = recall_k
        
        return metrics
    
    def evaluate_dataset(
        self, 
        gt_df: pd.DataFrame, 
        top_k: int = 10,
        window_days: int = 14,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate entire dataset and compute aggregate metrics.
        
        Args:
            gt_df: Ground truth DataFrame
            top_k: Maximum number of results to consider
            window_days: Time window for candidate retrieval
            verbose: Print progress
            
        Returns:
            Dictionary with aggregate metrics
        """
        # Group by source patch to handle multiple targets
        gt_grouped = gt_df.groupby('source_patch_id')['target_patch_id'].apply(list).to_dict()
        
        all_metrics = []
        skipped = 0
        
        total_queries = len(gt_grouped)
        
        for i, (source_id, target_ids) in enumerate(gt_grouped.items()):
            if verbose and (i + 1) % 100 == 0:
                print(f"   -> Processing query {i+1}/{total_queries}...")
            
            metrics = self.evaluate_single_query(
                source_id, 
                target_ids, 
                top_k=top_k,
                window_days=window_days
            )
            
            if metrics is None:
                skipped += 1
                continue
            
            all_metrics.append(metrics)
        
        if verbose:
            print(f"   -> Evaluated {len(all_metrics)} queries, skipped {skipped}")
        
        # Compute aggregate metrics
        aggregate = self._compute_aggregate_metrics(all_metrics)
        
        return {
            'aggregate_metrics': aggregate,
            'per_query_metrics': all_metrics,
            'num_queries': len(all_metrics),
            'num_skipped': skipped,
            'window_days': window_days,
            'top_k': top_k
        }
    
    def _compute_aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Compute aggregate metrics from per-query results."""
        if not all_metrics:
            return {}
        
        aggregate = {}
        
        # Mean Reciprocal Rank
        rr_values = [m['reciprocal_rank'] for m in all_metrics]
        aggregate['MRR'] = np.mean(rr_values)
        aggregate['MRR_std'] = np.std(rr_values)
        
        # Mean Recall@k
        k_values = [1, 2, 4, 6, 8, 10]
        for k in k_values:
            recall_values = [m[f'recall@{k}'] for m in all_metrics]
            aggregate[f'Mean_Recall@{k}'] = np.mean(recall_values)
            aggregate[f'Recall@{k}_std'] = np.std(recall_values)
        
        
        # Additional statistics
        aggregate['num_queries'] = len(all_metrics)
        
        # Hit rate (percentage of queries with at least one relevant result in top-k)
        for k in k_values:
            hits = sum(1 for m in all_metrics if m[f'recall@{k}'] > 0)
            aggregate[f'Hit_Rate@{k}'] = hits / len(all_metrics) if all_metrics else 0.0
        
        return aggregate
    
    def analyze_distribution(self, all_metrics: List[Dict]) -> Dict:
        """Analyze the distribution of metrics to understand variance."""
        rr_values = [m['reciprocal_rank'] for m in all_metrics]
        
        # MRR distribution
        rr_perfect = sum(1 for rr in rr_values if rr == 1.0)
        rr_zero = sum(1 for rr in rr_values if rr == 0.0)
        rr_middle = len(rr_values) - rr_perfect - rr_zero
        
        # Recall@10 distribution
        recall_values = [m['recall@10'] for m in all_metrics]
        recall_perfect = sum(1 for r in recall_values if r == 1.0)
        recall_zero = sum(1 for r in recall_values if r == 0.0)
        recall_partial = len(recall_values) - recall_perfect - recall_zero
        
        # Number of targets per query
        num_targets = [m['num_targets'] for m in all_metrics]
        
        return {
            'mrr_distribution': {
                'perfect_rank_1': rr_perfect,
                'not_found': rr_zero,
                'middle_ranks': rr_middle,
                'pct_perfect': rr_perfect / len(rr_values) if rr_values else 0,
                'pct_zero': rr_zero / len(rr_values) if rr_values else 0,
            },
            'recall_distribution': {
                'perfect_1.0': recall_perfect,
                'zero': recall_zero,
                'partial': recall_partial,
                'pct_perfect': recall_perfect / len(recall_values) if recall_values else 0,
                'pct_zero': recall_zero / len(recall_values) if recall_values else 0,
            },
            'targets_per_query': {
                'mean': np.mean(num_targets),
                'median': np.median(num_targets),
                'max': max(num_targets) if num_targets else 0,
                'single_target_pct': sum(1 for n in num_targets if n == 1) / len(num_targets) if num_targets else 0,
            }
        }
    
    def print_results(self, results: Dict):
        """Print evaluation results in a formatted table."""
        agg = results['aggregate_metrics']
        all_metrics = results.get('per_query_metrics', [])
        
        print("\n" + "=" * 70)
        print("RAG EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n📊 Configuration:")
        print(f"   Time Window: {results['window_days']} days")
        print(f"   Top-K: {results['top_k']}")
        print(f"   Queries Evaluated: {results['num_queries']}")
        print(f"   Queries Skipped: {results['num_skipped']}")
        
        # Distribution analysis
        if all_metrics:
            dist = self.analyze_distribution(all_metrics)
            
            print(f"\n📉 Distribution Analysis (explains high variance):")
            print(f"\n   MRR Distribution:")
            print(f"      Perfect (rank 1):  {dist['mrr_distribution']['perfect_rank_1']:>6} ({dist['mrr_distribution']['pct_perfect']*100:>5.1f}%)")
            print(f"      Not found (rank ∞): {dist['mrr_distribution']['not_found']:>6} ({dist['mrr_distribution']['pct_zero']*100:>5.1f}%)")
            print(f"      Middle ranks:      {dist['mrr_distribution']['middle_ranks']:>6} ({(1-dist['mrr_distribution']['pct_perfect']-dist['mrr_distribution']['pct_zero'])*100:>5.1f}%)")
            
            print(f"\n   Recall@10 Distribution:")
            print(f"      Perfect (1.0):     {dist['recall_distribution']['perfect_1.0']:>6} ({dist['recall_distribution']['pct_perfect']*100:>5.1f}%)")
            print(f"      Zero (0.0):        {dist['recall_distribution']['zero']:>6} ({dist['recall_distribution']['pct_zero']*100:>5.1f}%)")
            print(f"      Partial:           {dist['recall_distribution']['partial']:>6}")
            
            print(f"\n   Targets per Query:")
            print(f"      Mean:   {dist['targets_per_query']['mean']:.2f}")
            print(f"      Median: {dist['targets_per_query']['median']:.1f}")
            print(f"      Single-target queries: {dist['targets_per_query']['single_target_pct']*100:.1f}%")
        
        print(f"\n📈 Core Metrics:")
        print(f"\n   Mean Reciprocal Rank (MRR):")
        print(f"      {agg['MRR']:.4f} ± {agg['MRR_std']:.4f}")
        
        print(f"\n   Recall@k:")
        print(f"   {'k':<5} {'Recall':>10} {'Std':>10} {'Hit Rate':>10}")
        print(f"   {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
        
        k_values = [1, 2, 4, 6, 8, 10]
        for k in k_values:
            recall = agg[f'Mean_Recall@{k}']
            std = agg[f'Recall@{k}_std']
            hit_rate = agg[f'Hit_Rate@{k}']
            print(f"   {k:<5} {recall:>10.4f} {std:>10.4f} {hit_rate:>10.4f}")
        
        print("\n" + "=" * 70)
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        # Clean results for JSON
        clean_results = {
            'aggregate_metrics': {
                k: convert_to_serializable(v) 
                for k, v in results['aggregate_metrics'].items()
            },
            'num_queries': results['num_queries'],
            'num_skipped': results['num_skipped'],
            'window_days': results['window_days'],
            'top_k': results['top_k']
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline")
    parser.add_argument("--project", default="onap", help="Project name")
    parser.add_argument("--ground-truth", default=None, help="Path to ground truth CSV")
    parser.add_argument("--window-days", type=int, default=14, help="Time window in days")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum results to retrieve")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--engine", default="default", choices=["default", "improved"], help="Engine to use")
    parser.add_argument("--strategy", default="multi_query", choices=["hybrid", "file_boost", "multi_query"], help="Retrieval strategy for improved engine")
    
    args = parser.parse_args()
    
    # Initialize engine
    print("=" * 70)
    print("RAG Pipeline Evaluation")
    print("=" * 70)
    
    if args.engine == "improved":
        from core.improved_rag_engine import ImprovedRAGEngine
        engine = ImprovedRAGEngine(use_hybrid=True)
        print(f"\n🔧 Using Improved RAG Engine with strategy: {args.strategy}")
    else:
        engine = SmartPatchEngine()
        print("\n🔧 Using Default RAG Engine")
    
    # Load project
    csv_path = f"datasets/{args.project}/all_candidates.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        sys.exit(1)
    
    print(f"\n📁 Loading project: {args.project}")
    engine.load_project(args.project, csv_path)
    
    # Load ground truth
    if args.ground_truth:
        gt_path = args.ground_truth
    else:
        gt_path = f"datasets/{args.project}/ground_truth.csv"
    
    if not os.path.exists(gt_path):
        print(f"❌ Ground truth not found: {gt_path}")
        sys.exit(1)
    
    print(f"\n📋 Loading ground truth...")
    evaluator = RAGEvaluator(engine, args.project)
    
    # Pass strategy to evaluator if using improved engine
    if args.engine == "improved":
        evaluator.strategy = args.strategy
    
    gt_df = evaluator.load_ground_truth(gt_path)
    
    # Filter by time window
    print(f"\n⏱️ Filtering by time window ({args.window_days} days)...")
    gt_filtered = evaluator.filter_by_time_window(gt_df, args.window_days)
    
    if len(gt_filtered) == 0:
        print("❌ No pairs within time window")
        sys.exit(1)
    
    # Run evaluation
    print(f"\n🔍 Running evaluation...")
    results = evaluator.evaluate_dataset(
        gt_filtered, 
        top_k=args.top_k,
        window_days=args.window_days,
        verbose=True
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results if output path provided
    if args.output:
        evaluator.save_results(results, args.output)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
