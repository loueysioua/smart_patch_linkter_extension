"""
Test script for RAG pipeline with LLM reranking.

Usage:
    python test_rag_pipeline.py --project onap --patch-id <patch_id>
"""

import argparse
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.engine import SmartPatchEngine


def test_rag_engine():
    """Test basic RAG engine functionality."""
    print("=" * 60)
    print("Testing RAG Engine")
    print("=" * 60)
    
    # Initialize engine
    engine = SmartPatchEngine()
    
    # Test project loading
    print("\n1. Testing project loading...")
    test_csv = "datasets/onap/all_candidates.csv"
    
    if not os.path.exists(test_csv):
        print(f"⚠️ Test CSV not found: {test_csv}")
        print("Skipping project loading test.")
        return False
    
    engine.load_project("onap", test_csv)
    
    if "onap" not in engine.loaded_projects:
        print("❌ Failed to load project")
        return False
    
    print("✅ Project loaded successfully")
    
    # Test prediction
    print("\n2. Testing RAG prediction...")
    df = engine.datasets["onap"]
    
    if len(df) == 0:
        print("⚠️ No patches in dataset")
        return False
    
    # Get first patch as test
    test_patch = df.iloc[0]
    patch_ref = {
        "patch_id": test_patch['patch_id'],
        "title": test_patch['title'],
        "description": test_patch['description'],
        "created_time": test_patch['created_time'],
        "files": engine._safe_parse_list(test_patch['files'])
    }
    
    print(f"   Querying for patch: {test_patch['patch_id']}")
    print(f"   Title: {test_patch['title'][:80]}...")
    
    # Run prediction
    results = engine.predict("onap", patch_ref, top_k=5, window_days=14)
    
    if not results:
        print("⚠️ No results returned (this might be expected if no similar patches)")
        return True
    
    print(f"\n✅ Found {len(results)} similar patches:")
    for i, res in enumerate(results):
        print(f"\n   {i+1}. [{res['patch_id']}] (Score: {res['score']:.3f})")
        print(f"      Title: {res['title'][:70]}...")
        print(f"      Date: {res['created_time']}")
    
    return True


def test_full_pipeline(project, patch_id, use_llm=False, llm_model="gpt-4o-mini"):
    """Test full RAG + LLM reranking pipeline."""
    print("=" * 60)
    print("Testing Full Pipeline (RAG + LLM Reranking)")
    print("=" * 60)
    
    # Initialize engine
    engine = SmartPatchEngine()
    
    # Load project
    csv_path = f"datasets/{project}/all_candidates.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return False
    
    engine.load_project(project, csv_path)
    
    # Get patch details
    df = engine.datasets[project]
    patch_row = df[df.patch_id == patch_id]
    
    if patch_row.empty:
        print(f"❌ Patch {patch_id} not found in dataset")
        return False
    
    row = patch_row.iloc[0]
    patch_ref = {
        "patch_id": row['patch_id'],
        "title": row['title'],
        "description": row['description'],
        "created_time": row['created_time'],
        "files": engine._safe_parse_list(row['files'])
    }
    
    print(f"\n📌 Target Patch: {patch_id}")
    print(f"   Title: {row['title'][:80]}...")
    
    # Step 1: RAG Retrieval
    print("\n🔍 Step 1: RAG Semantic Search...")
    results = engine.predict(project, patch_ref, top_k=5, window_days=14)
    
    if not results:
        print("⚠️ No candidates found")
        return False
    
    print(f"\n   Found {len(results)} candidates:")
    for i, res in enumerate(results):
        print(f"   {i+1}. [{res['patch_id']}] (RAG Score: {res['score']:.3f})")
        print(f"      {res['title'][:70]}...")
    
    # Step 2: LLM Reranking (optional)
    if use_llm:
        print("\n🧠 Step 2: LLM Reranking...")
        
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                print("⚠️ OPENAI_API_KEY not set, skipping LLM reranking")
                return True
            
            client = OpenAI(api_key=api_key)
            
            # Import reranking function
            from llm_rerank import rerank_candidates
            
            reranked = rerank_candidates(client, patch_ref, results, model_name=llm_model)
            
            print(f"\n   Reranked results:")
            for i, res in enumerate(reranked):
                print(f"   {i+1}. [{res['patch_id']}] (RAG Score: {res['score']:.3f})")
                print(f"      {res['title'][:70]}...")
        
        except Exception as e:
            print(f"⚠️ LLM reranking failed: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test RAG Pipeline")
    parser.add_argument("--project", default="onap", help="Project name")
    parser.add_argument("--patch-id", help="Specific patch ID to test")
    parser.add_argument("--test-basic", action="store_true", help="Test basic RAG engine")
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM reranking")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model to use")
    
    args = parser.parse_args()
    
    if args.test_basic:
        # Basic RAG test
        success = test_rag_engine()
        sys.exit(0 if success else 1)
    
    if args.patch_id:
        # Full pipeline test with specific patch
        success = test_full_pipeline(
            args.project, 
            args.patch_id, 
            use_llm=args.with_llm,
            llm_model=args.llm_model
        )
        sys.exit(0 if success else 1)
    
    # Default: run basic test
    success = test_rag_engine()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
