"""
Diagnostic tool to understand why recall is low.
Analyzes the "hard" queries where relevant patches aren't found in top-k.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.engine import SmartPatchEngine


def diagnose_hard_queries(engine, project, gt_df, window_days=30, sample_size=50):
    """
    Analyze queries where relevant patches weren't found in top-10.
    """
    df = engine.datasets[project]
    
    # Group by source
    gt_grouped = gt_df.groupby('source_patch_id')['target_patch_id'].apply(list).to_dict()
    
    hard_cases = []  # No match in top-10
    easy_cases = []  # Match at rank 1
    medium_cases = []  # Match but not at rank 1
    
    print(f"\n🔍 Diagnosing {len(gt_grouped)} queries...")
    
    for i, (source_id, target_ids) in enumerate(gt_grouped.items()):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(gt_grouped)}...")
        
        # Get source patch
        source_row = df[df.patch_id == source_id]
        if source_row.empty:
            continue
        
        row = source_row.iloc[0]
        patch_ref = {
            "patch_id": row['patch_id'],
            "title": row['title'],
            "description": row['description'],
            "created_time": row['created_time'],
            "files": row['files_parsed']
        }
        
        # Get predictions
        predictions = engine.predict(project, patch_ref, top_k=10, window_days=window_days)
        predicted_ids = [p['patch_id'] for p in predictions]
        
        # Check if targets found
        found_ranks = []
        for target_id in target_ids:
            if target_id in predicted_ids:
                found_ranks.append(predicted_ids.index(target_id) + 1)
        
        min_rank = min(found_ranks) if found_ranks else None
        
        case_data = {
            'source_id': source_id,
            'target_ids': target_ids,
            'predicted_ids': predicted_ids,
            'found_ranks': found_ranks,
            'min_rank': min_rank,
            'source_title': row['title'][:100],
            'source_files': len(row['files_parsed']),
            'source_desc_len': len(row['description']) if pd.notna(row['description']) else 0
        }
        
        if min_rank is None:
            hard_cases.append(case_data)
        elif min_rank == 1:
            easy_cases.append(case_data)
        else:
            medium_cases.append(case_data)
    
    print(f"\n📊 Case Distribution:")
    print(f"   Easy (rank 1):    {len(easy_cases)} ({len(easy_cases)/len(gt_grouped)*100:.1f}%)")
    print(f"   Medium (rank >1): {len(medium_cases)} ({len(medium_cases)/len(gt_grouped)*100:.1f}%)")
    print(f"   Hard (not found): {len(hard_cases)} ({len(hard_cases)/len(gt_grouped)*100:.1f}%)")
    
    return hard_cases, medium_cases, easy_cases


def analyze_hard_cases(hard_cases, df, sample_size=20):
    """Deep dive into hard cases."""
    if not hard_cases:
        return
    
    print(f"\n🔬 Analyzing {min(len(hard_cases), sample_size)} Hard Cases:")
    print("=" * 80)
    
    np.random.seed(42)
    sample = np.random.choice(len(hard_cases), min(sample_size, len(hard_cases)), replace=False)
    
    for idx in sample:
        case = hard_cases[idx]
        source_id = case['source_id']
        target_ids = case['target_ids']
        predicted_ids = case['predicted_ids'][:5]  # Top 5
        
        source_row = df[df.patch_id == source_id].iloc[0]
        
        print(f"\n📌 Query: {source_id}")
        print(f"   Title: {source_row['title'][:70]}...")
        print(f"   Files: {case['source_files']}")
        print(f"   Desc Length: {case['source_desc_len']}")
        
        print(f"\n   Target patches (not found in top-10):")
        for tid in target_ids[:3]:
            target_row = df[df.patch_id == tid]
            if not target_row.empty:
                tr = target_row.iloc[0]
                print(f"      - [{tid}] {tr['title'][:60]}...")
                print(f"        Files overlap: {len(set(source_row['files_parsed']) & set(tr['files_parsed']))}")
                print(f"        Time delta: {abs((tr['created_time'] - source_row['created_time']).days)} days")
        
        print(f"\n   Predicted patches (top-5):")
        for i, pid in enumerate(predicted_ids, 1):
            pred_row = df[df.patch_id == pid]
            if not pred_row.empty:
                pr = pred_row.iloc[0]
                print(f"      {i}. [{pid}] {pr['title'][:60]}...")
                print(f"         Files overlap: {len(set(source_row['files_parsed']) & set(pr['files_parsed']))}")
        
        print("-" * 80)


def compute_retrieval_stats(hard_cases, medium_cases, easy_cases, df):
    """Compute statistics about retrieval quality."""
    print(f"\n📈 Retrieval Quality Analysis:")
    print("=" * 80)
    
    def get_stats(cases, label):
        if not cases:
            return
        
        # Text length
        desc_lens = []
        title_lens = []
        file_counts = []
        
        for case in cases:
            source_id = case['source_id']
            row = df[df.patch_id == source_id]
            if not row.empty:
                r = row.iloc[0]
                desc_lens.append(len(r['description']) if pd.notna(r['description']) else 0)
                title_lens.append(len(r['title']) if pd.notna(r['title']) else 0)
                file_counts.append(len(r['files_parsed']))
        
        print(f"\n   {label}:")
        print(f"      Avg Description Length: {np.mean(desc_lens):.0f} chars")
        print(f"      Avg Title Length: {np.mean(title_lens):.0f} chars")
        print(f"      Avg File Count: {np.mean(file_counts):.1f}")
        print(f"      Empty Descriptions: {sum(1 for d in desc_lens if d == 0)} ({sum(1 for d in desc_lens if d == 0)/len(desc_lens)*100:.1f}%)")
    
    get_stats(easy_cases, "Easy Cases (rank 1)")
    get_stats(medium_cases, "Medium Cases (rank >1)")
    get_stats(hard_cases, "Hard Cases (not found)")
    
    # Compare text similarity between source and target
    print(f"\n   Text Similarity (Source vs Target):")
    
    for label, cases in [("Easy", easy_cases[:100]), ("Hard", hard_cases[:100])]:
        if not cases:
            continue
        
        title_sim = []
        desc_sim = []
        
        for case in cases:
            source_id = case['source_id']
            target_ids = case['target_ids']
            
            source_row = df[df.patch_id == source_id]
            if source_row.empty:
                continue
            
            source_title = source_row.iloc[0]['title']
            source_desc = source_row.iloc[0]['description']
            
            for tid in target_ids[:1]:  # First target only
                target_row = df[df.patch_id == tid]
                if target_row.empty:
                    continue
                
                target_title = target_row.iloc[0]['title']
                target_desc = target_row.iloc[0]['description']
                
                # Simple word overlap
                if pd.notna(source_title) and pd.notna(target_title):
                    s_words = set(source_title.lower().split())
                    t_words = set(target_title.lower().split())
                    if s_words and t_words:
                        title_sim.append(len(s_words & t_words) / len(s_words | t_words))
                
                if pd.notna(source_desc) and pd.notna(target_desc):
                    s_words = set(source_desc.lower().split())
                    t_words = set(target_desc.lower().split())
                    if s_words and t_words:
                        desc_sim.append(len(s_words & t_words) / len(s_words | t_words))
        
        if title_sim:
            print(f"      {label} - Title Jaccard: {np.mean(title_sim):.4f}")
        if desc_sim:
            print(f"      {label} - Desc Jaccard: {np.mean(desc_sim):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose low recall in RAG pipeline")
    parser.add_argument("--project", default="onap", help="Project name")
    parser.add_argument("--window-days", type=int, default=30, help="Time window")
    parser.add_argument("--sample-size", type=int, default=20, help="Sample size for deep dive")
    
    args = parser.parse_args()
    
    # Load engine
    print("⚡ Loading RAG Engine...")
    engine = SmartPatchEngine()
    
    csv_path = f"datasets/{args.project}/all_candidates.csv"
    engine.load_project(args.project, csv_path)
    
    # Load ground truth
    gt_path = f"datasets/{args.project}/ground_truth.csv"
    gt_df = pd.read_csv(gt_path, dtype={"source_patch_id": str, "target_patch_id": str})
    
    # Filter by time window
    print(f"\n⏱️ Filtering by {args.window_days}-day window...")
    
    df = engine.datasets[args.project]
    filtered_pairs = []
    
    for _, row in gt_df.iterrows():
        source_id = str(row['source_patch_id'])
        target_id = str(row['target_patch_id'])
        
        source_row = df[df.patch_id == source_id]
        target_row = df[df.patch_id == target_id]
        
        if source_row.empty or target_row.empty:
            continue
        
        delta_days = abs((source_row.iloc[0]['created_time'] - target_row.iloc[0]['created_time']).total_seconds() / 86400)
        
        if delta_days <= args.window_days:
            filtered_pairs.append({
                'source_patch_id': source_id,
                'target_patch_id': target_id
            })
    
    gt_filtered = pd.DataFrame(filtered_pairs)
    print(f"   {len(gt_filtered)} pairs within time window")
    
    # Diagnose
    hard, medium, easy = diagnose_hard_queries(
        engine, args.project, gt_filtered, 
        window_days=args.window_days
    )
    
    # Analyze
    analyze_hard_cases(hard, df, sample_size=args.sample_size)
    compute_retrieval_stats(hard, medium, easy, df)
    
    # Recommendations
    print(f"\n💡 Recommendations to Improve Recall:")
    print("=" * 80)
    
    # Check empty descriptions
    empty_desc_count = sum(1 for case in hard 
                          if df[df.patch_id == case['source_id']].iloc[0]['description'] == '' 
                          or pd.isna(df[df.patch_id == case['source_id']].iloc[0]['description']))
    
    if empty_desc_count > len(hard) * 0.3:
        print(f"\n⚠️  {empty_desc_count}/{len(hard)} hard cases have empty descriptions")
        print("   → Consider enriching metadata (author, project, component)")
    
    print("\n✅ Diagnosis complete")


if __name__ == "__main__":
    main()
