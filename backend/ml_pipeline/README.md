# Archived ML Pipeline

These files are no longer used in the current RAG-based pipeline.

## Archived Files

- `train_model.py` - LightGBM LambdaRank model training (replaced by FAISS semantic search)
- `condidate_retrieval.py` - ANN candidate retrieval for ML training (now integrated in RAG engine)
- `build_dataset.py` - Dataset building utilities
- `evaluate_model.py` - ML model evaluation metrics

## Current Pipeline

The new pipeline uses:
1. **RAG Engine** (`core/engine.py`) - FAISS-based semantic search for candidate retrieval
2. **LLM Reranking** (`llm_rerank.py`) - GPT-based reranking of RAG results

This provides:
- No model training required - embeddings are computed on-the-fly
- Faster iteration - no retraining when data changes
- Better interpretability - semantic similarity is easier to understand
- LLM-enhanced ranking - combines embedding similarity with LLM understanding
