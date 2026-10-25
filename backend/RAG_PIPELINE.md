# RAG Pipeline Documentation

## Overview

The pipeline has been successfully transformed from **ML model training with LLM reranking** to **RAG (Retrieval Augmented Generation) with LLM reranking**.

## Architecture

### Previous Pipeline (ML-Based)
1. Train LightGBM LambdaRank model on labeled pairs
2. Model predicts similarity scores
3. LLM reranks top-k results

### New Pipeline (RAG-Based)
1. FAISS semantic search for candidate retrieval
2. Optional file similarity enhancement
3. LLM reranks top-k results

## Key Components

### 1. RAG Engine (`core/engine.py`)

**Main class**: `SmartPatchEngine`

**Key features**:
- FAISS-based semantic search using SBERT embeddings
- On-the-fly embedding computation with optional caching
- Time-window filtering for candidates
- Combined scoring: 70% semantic + 30% file overlap

**API**:
```python
engine = SmartPatchEngine()
engine.load_project("onap", "datasets/onap/all_candidates.csv")

# Predict similar patches
results = engine.predict(
    project="onap",
    patch_ref={
        "patch_id": "...",
        "title": "...",
        "description": "...",
        "created_time": datetime,
        "files": ["file1.py", "file2.py"]
    },
    top_k=5,
    window_days=14
)
```

### 2. LLM Reranking (`llm_rerank.py`)

Works seamlessly with RAG results. Takes the top-k candidates from semantic search and reranks them using GPT models.

**Usage**:
```bash
python llm_rerank.py \
    --project onap \
    --csv datasets/onap/all_candidates.csv \
    --target-id <patch_id> \
    --top-k 5 \
    --llm-model gpt-4o-mini
```

### 3. Flask API (`app.py`)

REST API endpoint for predictions:

```bash
POST /predict_topk
{
    "project": "onap",
    "patch_id": "100000",
    "time_window": 14,
    "top_k": 5
}
```

## Advantages of RAG Pipeline

### 1. No Model Training Required
- Embeddings computed on-the-fly
- No need for labeled training data
- No model retraining when data changes

### 2. Faster Iteration
- Update dataset → immediate results
- No training pipeline to manage
- Easier to experiment with different similarity measures

### 3. Better Interpretability
- Semantic similarity is easier to understand
- File overlap score provides additional context
- Clear ranking criteria

### 4. Comparable Performance
- RAG semantic search provides high-quality candidates
- LLM reranking adds deep understanding
- Combined approach: fast retrieval + intelligent ranking

## Performance

### Test Results (ONAP dataset: 101,829 patches)

✅ **Embedding computation**: ~1.5 minutes (cached for subsequent runs)  
✅ **FAISS index build**: Instant  
✅ **Prediction time**: < 1 second for top-5 candidates  
✅ **Memory usage**: Efficient (FAISS uses compressed indices)

### Example Output
```
📌 Target Patch: 100000
   Title: JJB prep for mod-bpgen component...

🔍 Step 1: RAG Semantic Search...
   Found 1 similar patches:
   1. [100135] (RAG Score: 0.826)
      mod-bpgen pattern typo...
```

## File Structure

```
backend/
├── core/
│   ├── engine.py          # RAG engine (FAISS + SBERT)
│   ├── rag_engine.py      # Alternative RAG implementation
│   ├── utils.py           # Utility functions
│   └── gerrit.py          # Gerrit API client
├── datasets/
│   └── onap/
│       ├── all_candidates.csv
│       └── ground_truth.csv
├── archived_ml_pipeline/  # Old ML training files
│   ├── train_model.py
│   ├── candidate_retrieval.py
│   ├── build_dataset.py
│   └── evaluate_model.py
├── app.py                 # Flask API server
├── llm_rerank.py         # LLM reranking script
├── test_rag_pipeline.py   # Test script
└── requirements.txt       # Dependencies
```

## Dependencies

Added to `requirements.txt`:
- `faiss-cpu` - Fast similarity search
- `openai` - LLM reranking
- `python-dotenv` - Environment variables

## Testing

### Basic RAG Test
```bash
python test_rag_pipeline.py --test-basic
```

### Full Pipeline with LLM Reranking
```bash
export OPENAI_API_KEY="your-key"
python test_rag_pipeline.py \
    --project onap \
    --patch-id 100000 \
    --with-llm \
    --llm-model gpt-4o-mini
```

### Start Flask Server
```bash
python app.py
# Server runs on http://0.0.0.0:5000
```

## Migration Guide

### From ML Pipeline to RAG Pipeline

1. **No code changes needed** for existing API calls
2. **Remove model files** - No longer needed
3. **Optional**: Cache embeddings for faster startup
4. **Benefits**: Faster iteration, no training required

### Backward Compatibility

The `predict()` method maintains the same interface, so existing integrations continue to work without modification.

## Future Enhancements

1. **Hybrid retrieval**: Combine semantic search with keyword matching (BM25)
2. **Embedding improvements**: Use domain-specific SBERT models
3. **FAISS optimization**: Use IVF or HNSW indices for larger datasets
4. **Multi-project search**: Cross-project similarity search
5. **Real-time updates**: Add/remove patches dynamically

## Troubleshooting

### FAISS Installation
```bash
pip install faiss-cpu  # For CPU
# or
pip install faiss-gpu  # For GPU (CUDA required)
```

### Memory Issues with Large Datasets
- Reduce embedding dimension: Use smaller SBERT models
- Use FAISS IVF indices instead of flat index
- Process in batches

### Slow First Startup
- First run computes embeddings (can take several minutes)
- Subsequent runs use cached embeddings
- Pre-compute embeddings: `engine._load_or_compute_embeddings()`

## Performance Tuning

### Embedding Cache
```python
# Cache embeddings for faster startup
engine.load_project(
    "onap",
    "datasets/onap/all_candidates.csv",
    embeddings_path="datasets/onap/onap_embeddings.npy"
)
```

### FAISS Index Options
```python
# For very large datasets (>1M patches)
# Use HNSW or IVF indices in core/rag_engine.py
index = faiss.IndexHNSWFlat(dim, 32)
# or
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
```

## Conclusion

The RAG pipeline provides a simpler, more maintainable approach to patch similarity detection while maintaining high-quality results through semantic search and LLM reranking.
