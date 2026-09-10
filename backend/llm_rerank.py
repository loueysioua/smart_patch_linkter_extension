import argparse
import os
import json
import re
from typing import List, Dict, Any
from core.engine import SmartPatchEngine
from core.gerrit import GerritClient

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

def get_patch_details(engine: SmartPatchEngine, project: str, patch_id: str) -> Dict[str, Any]:
    df = engine.datasets.get(project)
    if df is not None:
        existing_row = df[df.patch_id == patch_id]
        if not existing_row.empty:
            row = existing_row.iloc[0]
            return {
                "patch_id": row.patch_id,
                "title": row.title,
                "description": row.description,
                "created_time": row.created_time,
                "files": engine._safe_parse_list(row.files)
            }
    
    # Fallback to Gerrit API if not in dataset
    patch_ref = GerritClient.get_patch_details(project, patch_id)
    return patch_ref

def rerank_candidates(client: OpenAI, target: Dict[str, Any], candidates: List[Dict[str, Any]], model_name: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    if not candidates:
        return []

    # Construct prompt
    prompt = f"""You are an expert software developer. Your task is to rerank a list of candidate code changes (patches) based on their relevance to a target patch.
The candidate patches are currently ranked by a machine learning model, but you need to re-evaluate their similarity to the target patch and rank them from closest (most relevant/similar) to furthest (least relevant).

### Target Patch
- **ID:** {target.get('patch_id')}
- **Title:** {target.get('title')}
- **Description:** 
- **Files:** {', '.join(target.get('files', []))}
{target.get('description')}

### Candidate Patches to Rerank:
"""
    for idx, cand in enumerate(candidates):
        prompt += f"""
--- Candidate {idx + 1} ---
- **ID:** {cand.get('patch_id')}
- **Title:** {cand.get('title')}
- **Original Model Score:** {cand.get('score')}
- **Description:** 
- **Files:** {', '.join(cand.get('files', []))}
{cand.get('description', 'No description available')}
"""

    prompt += """
---
Please rerank these candidates from most relevant (1) to least relevant.
Output your final reranked list in JSON format, as a list of candidate IDs ordered from most to least relevant. 
Ensure you ONLY output the valid JSON array of strings (the patch IDs) and **nothing else**. Example: ["id_2", "id_1", "id_3"]
**NOTE**: the patch IDs that you will output must exactly match the patch IDs in the candidate patches list. DO NOT add any extra characters or formatting around the JSON array.
"""

    print("🧠 Asking LLM to rerank candidates...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0
    )

    response_text = response.choices[0].message.content.strip()
    
    # Try parsing JSON
    try:
        # Simple extraction if there are markdown code blocks
        json_str = response_text
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
            
        json_str = json_str.strip()
        reranked_ids = json.loads(json_str)
        
        if not isinstance(reranked_ids, list):
            raise ValueError("LLM response is not a list")
            
        # Reorder candidates based on LLM response
        id_to_cand = {c['patch_id']: c for c in candidates}
        reranked_candidates = []
        
        for pid in reranked_ids:
            if pid in id_to_cand:
                reranked_candidates.append(id_to_cand[pid])
                del id_to_cand[pid]
                
        # Append any remaining candidates that the LLM might have missed
        for cand in id_to_cand.values():
            reranked_candidates.append(cand)
            
        return reranked_candidates

    except Exception as e:
        print(f"⚠️ Failed to parse LLM response: {e}")
        print("Raw response:", response_text)
        print("Returning original ranking.")
        return candidates


def main():
    parser = argparse.ArgumentParser(description="Rerank top-k patches using Gemini LLM")
    parser.add_argument("--project", required=True, help="Project key (e.g., openstack)")
    parser.add_argument("--csv", required=True, help="Path to all_candidates.csv")
    parser.add_argument("--model", required=True, help="Path to the trained model_<project>.pkl")
    parser.add_argument("--target-id", required=True, help="Patch ID of the target query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates from ML model")
    parser.add_argument("--window-days", type=int, default=14, help="Time window for candidates")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="GPT model to use")
    
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is not set.")
        exit(1)

    client = OpenAI(api_key=api_key)

    engine = SmartPatchEngine()
    engine.load_project(args.project, args.csv, args.model)

    print(f"\n🔍 Fetching details for target patch {args.target_id}...")
    target_patch = get_patch_details(engine, args.project, args.target_id)
    if not target_patch:
        print(f"❌ Could not find target patch {args.target_id}")
        exit(1)

    print(f"\n🤖 Running ML Model prediction (top_k={args.top_k})...")
    # Using engine to get top-K predictions
    initial_results = engine.predict(args.project, target_patch, top_k=args.top_k, window_days=args.window_days)
    
    if not initial_results:
        print("No candidates found in the time window.")
        exit(0)

    # Enhance initial results with description for the LLM
    df = engine.datasets.get(args.project)
    enhanced_candidates = []
    for res in initial_results:
        cand_id = res['patch_id']
        cand_row = df[df.patch_id == cand_id] if df is not None else None
        
        description = "No description available"
        if cand_row is not None and not cand_row.empty:
            description = cand_row.iloc[0].description
            
        enhanced_candidates.append({
            **res,
            "description": description
        })

    print("\n--- Original ML Ranking ---")
    for i, c in enumerate(enhanced_candidates):
        print(f"{i+1}. [{c['patch_id']}] (Score: {c['score']:.3f}) {c['title']}")

    print("\n⏳ Reranking with LLM...")
    reranked_results = rerank_candidates(client, target_patch, enhanced_candidates, model_name=args.llm_model)

    print("\n--- LLM Reranked Results ---")
    for i, c in enumerate(reranked_results):
        print(f"{i+1}. [{c['patch_id']}] (Orig Score: {c['score']:.3f}) {c['title']}")

if __name__ == "__main__":
    main()
