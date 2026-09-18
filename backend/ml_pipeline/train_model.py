"""
train_patch_ranker.py

Trains a learning-to-rank model (LightGBM LambdaRank) to detect linked /
similar Gerrit patches, and evaluates it with MRR and Recall@k.

Usage:
    python train_patch_ranker.py \
        --dataset patches.csv \
        --ground-truth linked_pairs.csv \
        --model-out models/patch_ranker.pkl \
        --window-days 14 \
        --top-k 5

Dataset CSV is expected to have (at least) these columns:
    patch_id, title, description, created_time, files,
    author_name, author_email, author_username, change_log, comments

`patch_id` is treated as the sole unique identifier for a patch (both in
the dataset and in the ground-truth pairs file).

`files` must be a stringified Python list (e.g. "['a/b.py', 'c/d.py']").
`change_log` / `comments`, if present, are used for two things:
  1. deriving a reviewer/commenter list per patch (see `extract_reviewers`)
  2. building a "discussion text" blob per patch (see `extract_discussion_text`)
     which is SBERT-embedded and token-compared the same way title/description
     are, plus scanned for shared ticket/bug references.
If your schema stores these differently, adjust those two functions — the
key names they look for are just best-effort guesses.

Ground truth file is a list of linked patch_id pairs. Either:
  - a CSV with (at least) 2 columns, the first two of which are read as
    (patch_id_a, patch_id_b), or
  - a JSON file: a list of [patch_id_a, patch_id_b] pairs, or a list of
    dicts (the first two values of each dict are used).

Design notes / assumptions (see conversation for rationale):
  - All pairwise features are symmetric (order of A/B never matters), so
    each unordered ground-truth pair is assigned to exactly ONE canonical
    anchor (the earlier-created patch of the two). This avoids duplicate,
    identical feature rows and avoids the same pair leaking across the
    train/test split.
  - Candidates for an anchor are pulled from a hybrid pool: everything in
    the full dataset within a +/- `window_days` window, UNION the top
    `--ann-k` nearest neighbors by embedding cosine similarity (via FAISS,
    see candidate_retrieval.py), bounded to +/- `--ann-max-days`, UNION
    (optionally) RAG-based candidates from multi-query or file-boost
    retrieval (see --rag-candidates). This recovers ground-truth pairs
    that are textually/semantically linked but fall outside the plain time
    window -- pass --disable-ann to fall back to the pure time-window pool
    for comparison.
  - Train/test split is time-based (chronological), not random: anchors
    with created_time before the split point go to train, the rest to
    test. This matches "70% train / 30% test" via a time quantile.
  - Negative sampling mixes hard negatives (in-window, unlinked patches
    with high SBERT similarity to the anchor) with random negatives, to
    match the harder distribution the model sees at inference time.
"""

import sys
import os
# Add project root to sys.path so 'backend' can be resolved when running this script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend'))

import argparse
import ast
import json
import re
from datetime import timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.improved_rag_engine import ImprovedRAGEngine
from condidate_retrieval import CandidateIndex

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

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


def safe_parse_json(x):
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if not isinstance(x, str):
        return []
    try:
        parsed = json.loads(x)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


# def extract_reviewers(row):
#     """Best-effort reviewer extraction from change_log / comments entries.

#     Adjust the candidate key names below if your change_log/comments
#     records use a different schema for identifying the acting user.
#     """
#     reviewers = set()
#     for entry in (row.get("change_log_parsed") or []):
#         if isinstance(entry, dict):
#             for key in ("reviewer_username", "reviewer", "author_username", "user", "username"):
#                 val = entry.get(key)
#                 if val:
#                     reviewers.add(str(val).lower())
#     for entry in (row.get("comments_parsed") or []):
#         if isinstance(entry, dict):
#             for key in ("author_username", "reviewer_username", "user", "username", "author"):
#                 val = entry.get(key)
#                 if val:
#                     reviewers.add(str(val).lower())
#     own = str(row.get("author_username") or "").lower()
#     reviewers.discard(own)
#     return reviewers


def extract_discussion_text(row):
    """Best-effort text blob from change_log / comments entries, for SBERT
    + token comparison the same way title/description are compared.

    Adjust the candidate key names below if your change_log/comments
    records store the message text under a different field.
    """
    parts = []
    for entry in (row.get("change_log_parsed") or []):
        if isinstance(entry, dict):
            for key in ("message", "text", "body", "comment", "description"):
                val = entry.get(key)
                if val:
                    parts.append(str(val))
                    break
        elif isinstance(entry, str):
            parts.append(entry)
    for entry in (row.get("comments_parsed") or []):
        if isinstance(entry, dict):
            for key in ("message", "text", "body", "comment"):
                val = entry.get(key)
                if val:
                    parts.append(str(val))
                    break
        elif isinstance(entry, str):
            parts.append(entry)
    return " ".join(parts)


TICKET_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d+|#\d{3,7})\b")


def extract_ticket_refs(text):
    """Pulls bug/ticket-style references (e.g. 'BUG-1234', '#4821') out of
    text so two patches mentioning the same ticket can be flagged as linked
    even if their title/description wording differs."""
    return set(m.upper() for m in TICKET_RE.findall(text or ""))


def path2list(p):
    return p.strip("/").split("/") if isinstance(p, str) and p.strip("/") else []


def top_level_component(p):
    parts = path2list(p)
    return parts[0] if parts else None


def tokenize(text):
    return set(w.lower() for w in TOKEN_RE.findall(text or ""))


# --------------------------------------------------------------------------
# Feature engineering (symmetric: features(A, B) == features(B, A))
# --------------------------------------------------------------------------

def get_path_similarity_stats(files_A, files_B):
    stats = {}
    set_A, set_B = set(files_A), set(files_B)
    union = set_A | set_B
    stats["jaccard"] = len(set_A & set_B) / len(union) if union else 0.0
    stats["nb_shared"] = len(set_A & set_B)
    stats["abs_delta_files"] = abs(len(files_A) - len(files_B))
    stats["min_len_files"] = min(len(files_A), len(files_B))
    stats["max_len_files"] = max(len(files_A), len(files_B))

    comp_A = {top_level_component(f) for f in files_A if top_level_component(f)}
    comp_B = {top_level_component(f) for f in files_B if top_level_component(f)}
    stats["same_component"] = float(bool(comp_A & comp_B))

    if not files_A or not files_B:
        stats.update({"LCP_mean": 0.0, "LCP_max": 0.0, "LCSuff_mean": 0.0, "LCSuff_max": 0.0})
        return stats

    def lcp(f1, f2):
        return sum(1 for a, b in zip(path2list(f1), path2list(f2)) if a == b)

    def lcsuff(f1, f2):
        return sum(1 for a, b in zip(reversed(path2list(f1)), reversed(path2list(f2))) if a == b)

    lcp_scores = [lcp(fa, fb) / max(len(path2list(fa)), len(path2list(fb)), 1) for fa in files_A for fb in files_B]
    lcsuff_scores = [lcsuff(fa, fb) / max(len(path2list(fa)), len(path2list(fb)), 1) for fa in files_A for fb in files_B]
    stats["LCP_mean"] = float(np.mean(lcp_scores))
    stats["LCP_max"] = float(np.max(lcp_scores))
    stats["LCSuff_mean"] = float(np.mean(lcsuff_scores))
    stats["LCSuff_max"] = float(np.max(lcsuff_scores))
    return stats


def token_jaccard(text_a, text_b):
    a, b = tokenize(text_a), tokenize(text_b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def build_features(df, emb, emb_disc, i, j):
    row_i, row_j = df.iloc[i], df.iloc[j]
    sim_cosine = float(cosine_similarity(emb[i].reshape(1, -1), emb[j].reshape(1, -1))[0][0])
    file_stats = get_path_similarity_stats(row_i["files_parsed"], row_j["files_parsed"])
    text_i = f"{row_i['title']} {row_i['description']}"
    text_j = f"{row_j['title']} {row_j['description']}"

    # author_i = str(row_i.get("author_username") or "")
    # author_j = str(row_j.get("author_username") or "")

    # reviewers_i = row_i.get("reviewers", set())
    # reviewers_j = row_j.get("reviewers", set())

    has_disc_i = bool(row_i.get("has_discussion"))
    has_disc_j = bool(row_j.get("has_discussion"))
    if has_disc_i and has_disc_j:
        sim_cosine_discussion = float(cosine_similarity(emb_disc[i].reshape(1, -1), emb_disc[j].reshape(1, -1))[0][0])
        token_jaccard_discussion = token_jaccard(row_i["discussion_text"], row_j["discussion_text"])
    else:
        sim_cosine_discussion = 0.0
        token_jaccard_discussion = 0.0

    ticket_refs_i = row_i.get("ticket_refs", set())
    ticket_refs_j = row_j.get("ticket_refs", set())

    feats = {
        **file_stats,
        "sim_cosine": sim_cosine,
        "token_jaccard": token_jaccard(text_i, text_j),
        "delta_time_hours": abs((row_i["created_time"] - row_j["created_time"]).total_seconds() / 3600),
        # "same_author": float(bool(author_i) and author_i == author_j),
        # "same_reviewer": float(bool(reviewers_i & reviewers_j)),
        # "nb_shared_reviewers": len(reviewers_i & reviewers_j),
        "sim_cosine_discussion": sim_cosine_discussion,
        "token_jaccard_discussion": token_jaccard_discussion,
        "has_discussion_both": float(has_disc_i and has_disc_j),
        "shares_ticket_ref": float(bool(ticket_refs_i & ticket_refs_j)),
    }
    return feats


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["created_time"], dtype={"patch_id": str})
    df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["author_username"] = df.get("author_username", "").fillna("") if "author_username" in df.columns else ""
    df["files_parsed"] = df["files"].apply(safe_parse_list)
    df["change_log_parsed"] = df["change_log"].apply(safe_parse_json) if "change_log" in df.columns else [[] for _ in range(len(df))] 
    print("change_log_parsed",df["change_log_parsed"].head())
    df["comments_parsed"] = df["comments"].apply(safe_parse_json) if "comments" in df.columns else [[] for _ in range(len(df))]
    print("comments_parsed",df["comments_parsed"].count())
    # df["reviewers"] = df.apply(extract_reviewers, axis=1)
    df["discussion_text"] = df.apply(extract_discussion_text, axis=1)
    df["has_discussion"] = df["discussion_text"].str.strip().str.len() > 0
    df["ticket_refs"] = (df["title"] + " " + df["description"] + " " + df["discussion_text"]).apply(extract_ticket_refs)
    df = df.sort_values("created_time").reset_index(drop=True)
    return df


def load_or_compute_embeddings(df, sbert_model_name, cache_path):
    texts = (df["title"] + " " + df["description"]).tolist()
    discussion_texts = df["discussion_text"].tolist()

    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True).item()
            if cached.get("model_name") == sbert_model_name and cached.get("n_rows") == len(df):
                print(f"  loaded cached embeddings from {cache_path}")
                return cached["embeddings"], cached["discussion_embeddings"]
            print("  cache found but stale (model or row count changed) \u2014 recomputing")

    sbert = SentenceTransformer(sbert_model_name)
    emb = sbert.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    emb_disc = sbert.encode(discussion_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, {"model_name": sbert_model_name, "n_rows": len(df),
                              "embeddings": emb, "discussion_embeddings": emb_disc})
        print(f"  cached embeddings to {cache_path}")

    return emb, emb_disc


def load_ground_truth(path, df):
    id_to_idx = {str(pid): i for i, pid in enumerate(df["patch_id"])}
    raw_pairs = []

    if str(path).lower().endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_pairs.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict) and len(item) >= 2:
                vals = list(item.values())
                raw_pairs.append((str(vals[0]), str(vals[1])))
    else:
        gt = pd.read_csv(path, dtype=str)
        cols = list(gt.columns[:2])
        for _, r in gt.iterrows():
            raw_pairs.append((str(r[cols[0]]), str(r[cols[1]])))

    pairs, missing, seen = [], 0, set()
    for a, b in raw_pairs:
        if a not in id_to_idx or b not in id_to_idx or a == b:
            missing += 1
            continue
        i, j = id_to_idx[a], id_to_idx[b]
        key = tuple(sorted((i, j)))
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    if missing:
        print(f"  \u26a0\ufe0f  skipped {missing} ground-truth pairs referencing unknown patch_ids")
    return pairs


# --------------------------------------------------------------------------
# Candidate windowing / negative sampling / group construction
# --------------------------------------------------------------------------

def report_window_coverage(df, pairs, window_days, candidate_fn=None, ann_enabled=False, rag_enabled=False):
    """Diagnostic: how many ground-truth pairs are actually usable at this
    window size, before we spend time building groups / training.
    
    If candidate_fn is provided, also reports how many pairs are recovered
    by the hybrid candidate pool (ANN + RAG) vs pure time-window.
    """
    deltas_hours = []
    unusable_by_window = 0
    for i, j in pairs:
        delta_h = abs((df.loc[i, "created_time"] - df.loc[j, "created_time"]).total_seconds() / 3600)
        deltas_hours.append(delta_h)
        if delta_h > window_days * 24:
            unusable_by_window += 1

    deltas_days = np.array(deltas_hours) / 24
    print(f"  pair time-deltas (days): median={np.median(deltas_days):.1f}  "
          f"p75={np.percentile(deltas_days, 75):.1f}  p90={np.percentile(deltas_days, 90):.1f}  "
          f"max={np.max(deltas_days):.1f}")
    usable_pct = 100 * (len(pairs) - unusable_by_window) / len(pairs) if pairs else 0
    print(f"  {len(pairs) - unusable_by_window}/{len(pairs)} pairs ({usable_pct:.1f}%) fall within "
          f"+/-{window_days} days and will be usable for training")
    
    # If hybrid candidate pool is enabled, check how many additional pairs are recoverable
    # Note: This can be slow with RAG enabled, so we show progress
    if candidate_fn is not None and unusable_by_window > 0:
        print(f"  Checking recovery for {unusable_by_window} out-of-window pairs (this may take a moment)...")
        recovered_by_hybrid = 0
        checked = 0
        for i, j in pairs:
            delta_h = abs((df.loc[i, "created_time"] - df.loc[j, "created_time"]).total_seconds() / 3600)
            if delta_h > window_days * 24:
                # This pair is outside the time window - check if it's recovered by hybrid pool
                # Use the earlier patch as anchor
                anchor = i if df.loc[i, "created_time"] <= df.loc[j, "created_time"] else j
                candidate = j if anchor == i else i
                try:
                    candidates = candidate_fn(anchor)
                    if candidate in candidates:
                        recovered_by_hybrid += 1
                except Exception as e:
                    pass  # Skip if retrieval fails
                checked += 1
                if checked % 100 == 0:
                    print(f"    ... checked {checked}/{unusable_by_window} pairs")
        
        if recovered_by_hybrid > 0:
            sources = []
            if ann_enabled:
                sources.append("ANN")
            if rag_enabled:
                sources.append("RAG")
            source_str = " + ".join(sources) if sources else "hybrid"
            print(f"  \U0001F504 {recovered_by_hybrid}/{unusable_by_window} out-of-window pairs RECOVERED by {source_str} retrieval")
            total_usable = len(pairs) - unusable_by_window + recovered_by_hybrid
            print(f"  \u2705 Total usable pairs: {total_usable}/{len(pairs)} ({100*total_usable/len(pairs):.1f}%)")
        else:
            if unusable_by_window > 0:
                print(f"  \u26a0\ufe0f  No additional pairs recovered by hybrid retrieval (consider increasing ann_k, rag_k, or ann_max_days)")
    
    if usable_pct < 70 and candidate_fn is None:
        print(f"  \u26a0\ufe0f  over {100 - usable_pct:.0f}% of your labeled pairs are OUTSIDE the "
              f"+/-{window_days}-day window and will be silently dropped. Consider raising --window-days, "
              f"or enable --ann / --rag-candidates to recover them, "
              f"or confirm this loss is expected (e.g. distant links genuinely shouldn't be retrievable).")


def get_candidate_indices(df, anchor_idx, window_days):
    anchor_time = df.iloc[anchor_idx]["created_time"]
    start, end = anchor_time - timedelta(days=window_days), anchor_time + timedelta(days=window_days)
    mask = (df["created_time"] >= start) & (df["created_time"] <= end)
    idxs = df.index[mask].tolist()
    return [k for k in idxs if k != anchor_idx]


def sample_negatives(df, emb, anchor_idx, candidate_idxs, positive_idxs, max_negatives, hard_ratio, rng):
    neg_pool = [k for k in candidate_idxs if k not in positive_idxs]
    if not neg_pool:
        return []
    if len(neg_pool) <= max_negatives:
        return neg_pool

    anchor_emb = emb[anchor_idx].reshape(1, -1)
    sims = cosine_similarity(anchor_emb, emb[neg_pool])[0]
    order = np.argsort(-sims)

    n_hard = min(int(round(max_negatives * hard_ratio)), len(neg_pool))
    hard_negs = [neg_pool[idx] for idx in order[:n_hard]]

    remaining_pool = [neg_pool[idx] for idx in order[n_hard:]]
    n_easy = max_negatives - len(hard_negs)
    if n_easy > 0 and remaining_pool:
        easy_negs = list(rng.choice(remaining_pool, size=min(n_easy, len(remaining_pool)), replace=False))
    else:
        easy_negs = []
    return hard_negs + easy_negs


def build_training_rows(df, emb, emb_disc, anchor_positive_map, candidate_fn, max_negatives, hard_ratio, seed):
    """`candidate_fn(anchor_idx) -> list[int]` supplies the candidate pool for
    an anchor. Pass a plain time-window function or the hybrid ANN+window
    wrapper (get_hybrid_candidates) built in main()."""
    rng = np.random.default_rng(seed)
    groups = []
    for anchor_idx, positive_idxs in anchor_positive_map.items():
        candidate_idxs = candidate_fn(anchor_idx)
        in_window_positives = positive_idxs & set(candidate_idxs)
        if not in_window_positives:
            continue  # the linked patch fell outside the candidate pool; not learnable for this anchor

        negatives = sample_negatives(df, emb, anchor_idx, candidate_idxs, positive_idxs,
                                      max_negatives, hard_ratio, rng)

        group_candidates = list(in_window_positives) + negatives
        group_labels = [1] * len(in_window_positives) + [0] * len(negatives)

        order = rng.permutation(len(group_candidates))
        group_candidates = [group_candidates[k] for k in order]
        group_labels = [group_labels[k] for k in order]

        groups.append((anchor_idx, group_candidates, group_labels))
    return groups


def groups_to_frame(df, emb, emb_disc, groups):
    feat_rows, labels, group_sizes = [], [], []
    for anchor_idx, cand_idxs, cand_labels in groups:
        if not cand_idxs:
            continue
        for cidx, lbl in zip(cand_idxs, cand_labels):
            feat_rows.append(build_features(df, emb, emb_disc, anchor_idx, cidx))
            labels.append(lbl)
        group_sizes.append(len(cand_idxs))
    X = pd.DataFrame(feat_rows)
    y = np.array(labels)
    return X, y, group_sizes


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

def train_model(X_train, y_train, group_train, X_test, y_test, group_test, feature_cols, lgb_params, num_boost_round, stopping_rounds):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "verbosity": -1,
    }
    params.update(lgb_params)
    train_set = lgb.Dataset(X_train[feature_cols], label=y_train, group=group_train)
    valid_set = lgb.Dataset(X_test[feature_cols], label=y_test, group=group_test, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds), lgb.log_evaluation(period=50)],
    )
    return model


def evaluate(model, df, emb, emb_disc, anchors, candidate_fn, feature_cols, top_k, positives_by_anchor):
    reciprocal_ranks, recalls = [], []
    for anchor_idx in anchors:
        candidate_idxs = candidate_fn(anchor_idx)
        if not candidate_idxs:
            continue
        positive_set = positives_by_anchor.get(anchor_idx, set()) & set(candidate_idxs)
        if not positive_set:
            continue

        feats = pd.DataFrame([build_features(df, emb, emb_disc, anchor_idx, c) for c in candidate_idxs])
        feats = feats.reindex(columns=feature_cols, fill_value=0)
        scores = model.predict(feats)
        order = np.argsort(-scores)
        ranked = [candidate_idxs[k] for k in order]

        rank = next((r for r, cand in enumerate(ranked, start=1) if cand in positive_set), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        recalled = sum(1 for c in ranked[:top_k] if c in positive_set)
        recalls.append(recalled / len(positive_set))

    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    recall_k = float(np.mean(recalls)) if recalls else 0.0
    return mrr, recall_k, len(reciprocal_ranks)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a LambdaRank model to detect linked Gerrit patches.")
    parser.add_argument("--dataset", required=True, help="Path to the patches CSV.")
    parser.add_argument("--ground-truth", required=True, help="Path to linked-pairs CSV or JSON.")
    parser.add_argument("--model-out", required=True, help="Path to save the trained model (.pkl).")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ann-k", type=int, default=50, help="Number of ANN (embedding-similarity) neighbors to add to the candidate pool per anchor.")
    parser.add_argument("--ann-max-days", type=int, default=90, help="Time bound applied to ANN candidates (wider than --window-days).")
    parser.add_argument("--disable-ann", action="store_true", help="Fall back to pure time-window candidates (for A/B comparison against the hybrid pool).")
    parser.add_argument("--ann-exact", action="store_true", help="Use exact (brute-force) FAISS search instead of HNSW. Fine for small datasets; slower to build/query at scale.")
    parser.add_argument("--rag-candidates", action="store_true", help="Enable RAG-based candidate retrieval (multi-query/file-boost) as an additional candidate source.")
    parser.add_argument("--rag-k", type=int, default=20, help="Number of RAG candidates to retrieve per anchor.")
    parser.add_argument("--rag-strategy", choices=["multi_query", "file_boost"], default="multi_query", help="RAG retrieval strategy: 'multi_query' or 'file_boost'.")
    parser.add_argument("--rag-window-days", type=int, default=30, help="Time window for RAG retrieval (typically wider than --window-days).")
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction of (time-ordered) anchors held out for testing.")
    parser.add_argument("--max-negatives", type=int, default=20, help="Max negatives sampled per training/eval group.")
    parser.add_argument("--hard-negative-ratio", type=float, default=0.6, help="Fraction of sampled negatives that are 'hard' (high SBERT similarity, unlinked).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-cache", default=None, help="Optional .npy path to cache/reuse SBERT embeddings across runs.")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15, help="Lower = less overfitting. Keep small relative to feature count (14 features here).")
    parser.add_argument("--min-data-in-leaf", type=int, default=30, help="Higher = less overfitting on small group counts.")
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=5)
    parser.add_argument("--lambda-l1", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=0.1)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.dataset)
    print(f"  {len(df)} patches loaded")

    print("Loading ground truth...")
    pairs = load_ground_truth(args.ground_truth, df)
    print(f"  {len(pairs)} unique linked pairs resolved against dataset")
    if not pairs:
        raise SystemExit("No ground-truth pairs could be matched to the dataset. Aborting.")

    print("Encoding titles + descriptions + discussion text with SBERT...")
    emb, emb_disc = load_or_compute_embeddings(df, args.sbert_model, args.embedding_cache)
    n_with_discussion = int(df["has_discussion"].sum())
    print(f"  {n_with_discussion}/{len(df)} patches have usable change_log/comments text")

    print("Checking ground-truth pair coverage at this window size...")
    report_window_coverage(df, pairs, args.window_days)

    # Build RAG engine if requested
    rag_engine = None
    if args.rag_candidates:
        print(f"Building RAG engine for candidate retrieval (strategy={args.rag_strategy}, k={args.rag_k})...")
        rag_engine = ImprovedRAGEngine(use_hybrid=True)
        # Load the same dataset into RAG engine (use "train" as project key)
        rag_engine.load_project("train", args.dataset)
        print(f"  RAG engine initialized with {len(rag_engine.datasets.get('train', []))} patches")

    if args.disable_ann and not args.rag_candidates:
        print("ANN and RAG retrieval disabled — using pure time-window candidates.")
        cand_index = None
    else:
        print(f"Building FAISS candidate index (ann_k={args.ann_k}, ann_max_days={args.ann_max_days})...")
        cand_index = CandidateIndex.build(
            emb, df["created_time"].to_numpy(),
            use_hnsw=not args.ann_exact,
            df=df,
            rag_engine=rag_engine,
        )

    def get_hybrid_candidates(anchor_idx, window_days):
        """Time-window candidates unioned with ANN (embedding-similarity)
        neighbors and optionally RAG-based candidates, so ground-truth pairs
        outside the window are still reachable at train and eval time.
        Falls back to pure time-window candidates if both ANN and RAG are disabled."""
        time_window_fn = lambda a: get_candidate_indices(df, a, window_days)
        if cand_index is None:
            return time_window_fn(anchor_idx)
        return cand_index.get_candidates(
            anchor_idx=anchor_idx,
            emb=emb,
            window_days=window_days,
            ann_k=args.ann_k,
            ann_max_days=args.ann_max_days,
            time_window_fn=time_window_fn,
            rag_k=args.rag_k if args.rag_candidates else 0,
            rag_strategy=args.rag_strategy,
            rag_window_days=args.rag_window_days,
        )

    # Re-run coverage diagnostic with hybrid candidate pool to show recovery
    if cand_index is not None:
        print("\nRe-checking coverage with hybrid candidate pool...")
        report_window_coverage(
            df, pairs, args.window_days,
            candidate_fn=lambda a: get_hybrid_candidates(a, args.window_days),
            ann_enabled=not args.disable_ann,
            rag_enabled=args.rag_candidates,
        )

    print("Assigning canonical anchor per pair (earlier-created patch)...")
    positives_by_anchor = {}
    for i, j in pairs:
        anchor, cand = (i, j) if df.loc[i, "created_time"] <= df.loc[j, "created_time"] else (j, i)
        positives_by_anchor.setdefault(anchor, set()).add(cand)

    split_time = df["created_time"].quantile(1 - args.test_size)
    print(f"Time-based split point: {split_time}")
    train_anchors = [a for a in positives_by_anchor if df.loc[a, "created_time"] < split_time]
    test_anchors = [a for a in positives_by_anchor if df.loc[a, "created_time"] >= split_time]
    print(f"  {len(train_anchors)} train anchors / {len(test_anchors)} test anchors")

    print("Building training groups (with hard-negative sampling)...")
    train_map = {a: positives_by_anchor[a] for a in train_anchors}
    test_map = {a: positives_by_anchor[a] for a in test_anchors}
    train_candidate_fn = lambda a: get_hybrid_candidates(a, args.window_days)
    train_groups = build_training_rows(df, emb, emb_disc, train_map, train_candidate_fn, args.max_negatives, args.hard_negative_ratio, args.seed)
    test_groups = build_training_rows(df, emb, emb_disc, test_map, train_candidate_fn, args.max_negatives, args.hard_negative_ratio, args.seed + 1)

    X_train, y_train, group_sizes_train = groups_to_frame(df, emb, emb_disc, train_groups)
    X_test, y_test, group_sizes_test = groups_to_frame(df, emb, emb_disc, test_groups)

    if X_train.empty or X_test.empty:
        raise SystemExit("Train or test set ended up empty (no anchors had an in-window positive). "
                          "Try increasing --window-days or check your ground-truth file.")

    feature_cols = sorted(set(X_train.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    print(f"  train rows: {len(X_train)} across {len(group_sizes_train)} groups (positives: {int(y_train.sum())})")
    print(f"  test rows:  {len(X_test)} across {len(group_sizes_test)} groups (positives: {int(y_test.sum())})")

    print("Training LightGBM LambdaRank model...")
    lgb_params = {
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
    }
    model = train_model(X_train, y_train, group_sizes_train, X_test, y_test, group_sizes_test,
                         feature_cols, lgb_params, args.num_boost_round, args.early_stopping_rounds)

    eval_windows = [2, 7, 14, 30]
    eval_top_ks  = [1, 2, 4, 6, 8, 10]

    print("\n" + "=" * 60)
    print("EVALUATION  —  all day-windows × all top-k values")
    print("=" * 60)

    for win in eval_windows:
        print(f"\n── Window: ±{win} days ──────────────────────────────────────")
        header = f"  {'k':>4}  {'MRR':>8}  {'Recall@k':>10}  {'#anchors':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        eval_candidate_fn = lambda a, win=win: get_hybrid_candidates(a, win)
        for k in eval_top_ks:
            mrr, recall_k, n_eval = evaluate(
                model, df, emb, emb_disc, test_anchors,
                eval_candidate_fn, feature_cols, k, positives_by_anchor
            )
            print(f"  {k:>4}  {mrr:>8.4f}  {recall_k:>10.4f}  {n_eval:>9}")

    importances = sorted(zip(feature_cols, model.feature_importance()), key=lambda x: -x[1])
    print("Top feature importances:")
    for name, score in importances[:10]:
        print(f"    {name}: {score}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "window_days": args.window_days,
        "top_k": args.top_k,
        "sbert_model_name": args.sbert_model,
        "ann_enabled": not args.disable_ann,
        "ann_k": args.ann_k,
        "ann_max_days": args.ann_max_days,
    }, out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()