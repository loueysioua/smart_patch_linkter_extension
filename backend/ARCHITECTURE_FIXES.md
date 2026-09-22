# ML→RAG Pipeline Architecture Fixes

This document describes the architectural issues identified in the ML→RAG pipeline and the fixes applied.

## Issues Identified

### Issue 1: Window Mismatch Between ML and RAG Stages

**Problem:** The two arms of the comparison weren't playing on the same field.

- `predict_ml_stage` defaulted to `self.eval_window_days` (±2 days) when `window_days` wasn't explicitly passed
- `compare_approaches` called RAG-only with `pipeline.window_days` (±14 days)
- If the true duplicate/related patch fell between 2 and 14 days out, ML→RAG could never find it — not because RAG refinement was bad, but because ML stage threw the answer away before RAG ever got a look

**Fix:**
- Changed default in `predict_ml_stage` to use `self.window_days` (training window) instead of `self.eval_window_days`
- Updated `compare_approaches` to explicitly use consistent `window_days` for all approaches
- Added clear logging of which window is being used

**Files Changed:** `scripts/ml_to_rag_pipeline.py`

### Issue 2: Hard Filter-Then-Rerank Caps Recall at ML Model's Ceiling

**Problem:** Even with matched windows, the structure was: ML scores everything → keep top-20 → RAG can only reorder within those 20. RAG never gets to pull in a candidate the ML model ranked #25.

- Overall Recall@k can never exceed the ML model's Recall@ml_candidates_k
- If the LightGBM model's recall@20 is mediocre, chaining a great RAG reranker on top literally cannot rescue it
- This is very different from RAG→ML (RAG retrieves broadly, ML just reranks a rich set)

**Fix:**
- Implemented **union retrieval**: take top-K from ML stage AND top-K from RAG, merge by ID, then blend scores
- RAG can now reintroduce candidates ML missed
- RAG-only candidates get assigned a neutral ML score (minimum from global distribution)
- Added `use_union_retrieval` parameter to `predict()` method

**Files Changed:** `scripts/ml_to_rag_pipeline.py`

### Issue 3: Per-Candidate Embedding Recomputation (Efficiency)

**Problem:** The code was re-encoding the same query text once per candidate:
```python
for cand in ml_candidates:
    query_emb = self.rag_engine._get_encoder().encode([query_text], ...)
    cand_emb = self.rag_engine._get_encoder().encode([cand_text], ...)
```

The embeddings are already pre-computed in `self.rag_engine.embeddings` with matching indices.

**Fix:**
- Encode query once before the loop
- Look up candidate embeddings by index from pre-computed cache
- Fall back to encoding only if cache is unavailable

**Files Changed:** `scripts/ml_to_rag_pipeline.py`

### Issue 4: Narrow Min-Max Normalization on Top-20 Only

**Problem:** ML score normalization was done on the narrow, already-filtered slice:
```python
ml_min = min(ml_scores)  # only over the top 20!
ml_max = max(ml_scores)
normalized_ml_score = (cand["ml_score"] - ml_min) / ml_range
```

Since these 20 are already the model's most confident picks, their score range is compressed. Stretching to [0,1] makes `rag_weight=0.3` behave very differently run-to-run.

**Fix:**
- Compute global ML score range from all candidates in the time window
- Pass this range to `predict_rag_refine()` for stable normalization
- Now `rag_weight` has consistent meaning across queries

**Files Changed:** `scripts/ml_to_rag_pipeline.py`

## New Diagnostic Tool

Created `scripts/diagnose_ml_stage.py` to:
1. Measure ML stage recall at different window sizes (2, 7, 14, 30 days)
2. Compare with RAG-only recall at the same window
3. Identify if ML stage is the bottleneck
4. Recommend whether union retrieval will help

Usage:
```bash
python scripts/diagnose_ml_stage.py --project onap --model train3/onap/model_onap_30.pkl
```

## Verification

After applying fixes, re-run evaluation:
```bash
python scripts/ml_to_rag_pipeline.py \
    --project onap \
    --model train3/onap/model_onap_30.pkl \
    --evaluate \
    --window-days 14 \
    --rag-weight 0.3
```

Compare results with previous run. You should see:
- Higher recall when union retrieval is enabled
- More stable behavior across different queries
- Faster execution due to embedding cache usage

## Key Metrics to Watch

1. **ML Stage Recall@20** - If this is low (< RAG recall), union retrieval will help
2. **Window consistency** - All approaches should use same window for fair comparison
3. **RAG rescue rate** - How many candidates did RAG pull in that ML missed?
4. **Execution time** - Should be faster with embedding cache

## Future Improvements (Not Implemented)

1. **Calibrated ML scores** - Instead of min-max, fit a sigmoid to ML score distribution for better calibration
2. **Percentile-based normalization** - Use percentile rank against full-window score distribution
3. **Weighted union** - Instead of minimum ML score for RAG-only candidates, use a learned weight
4. **Adaptive rag_weight** - Tune per-query based on ML confidence distribution
