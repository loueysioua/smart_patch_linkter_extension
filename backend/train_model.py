"""
train_model.py

Trains model_<project>.pkl for SmartPatchEngine (core/engine.py).

Why this needs the RAW dump, not just the CSV
-----------------------------------------------
build_dataset.py only carries metadata.json into the CSV (patch_id, title,
description, created_time, files). Labels for "is patch A related to patch
B" live in linkages.json, which build_dataset.py never reads. So this script
walks the raw dump a second time, just for labels, and joins them against
the already-built CSV for features.

Layout expected (same as build_dataset.py):
    <root>/<project>/<change_id>/metadata.json
    <root>/<project>/<change_id>/linkages.json

Label rule (per your spec)
---------------------------
- linkages.json is a JSON array. Every object in it is a positive link
  between the source change (the directory it lives in) and a target
  change. The target is identified by the object's `changeNumber`
  (nonzero — the schema uses 0 to mean "no number, only a Change-Id hash
  is known") or, when that's unusable, by its `changeId`.
- IMPORTANT: as of your updated build_dataset.py, patch_id is NOT
  "<project>~<number>" anymore — extract_ids() there now derives it as
  just the bare `_number` (str), falling back to the raw `change_id`
  hash, falling back to the directory name. This script imports that
  same extract_ids() so the ids it computes while walking linkages.json
  are guaranteed to match what ended up in the CSV. The link object's
  `project` field is NOT used for matching — it names the Gerrit repo
  (e.g. "core"), which has no fixed relationship to whatever directory
  name you passed via --project, and isn't needed anyway since Gerrit
  change numbers are unique per host, not per repo.
- An empty array ([]) means that change has no links.
- isExternal / confidence / detectionMethod are NOT used to filter —
  every entry counts, per your instruction. Entries whose target isn't in
  this project's dataset (e.g. a genuinely cross-repo link, or a target
  that got dropped upstream for missing created_time) are skipped, since
  the model can only be trained on pairs it can compute features for.

Feature parity with engine.py
------------------------------
predict() in engine.py, for a query patch `patch_ref` and a candidate row
from the dataset, computes:

    file_stats = get_path_similarity_stats(row.files_parsed, patch_ref.files)
    sim_cosine  = cosine_similarity(sbert(query), sbert(candidate))
    delta_time_hours = abs(row.created_time - patch_ref.created_time) in hours
    len_A = len(row.files_parsed)          # candidate
    len_B = len(patch_ref.files)           # query

This script reproduces that exact construction (files_A = candidate,
files_B = query) for every training pair, so train/serve stay consistent.
Since "query" vs "candidate" is an arbitrary role assignment for a link
that is really symmetric, positive pairs are added in BOTH directions
(A-as-query/B-as-candidate and B-as-query/A-as-candidate) so the model
isn't biased toward one ordering.

Usage
-----
    python train_model.py \\
        --root /path/to/gerrit_dump \\
        --csv data/openstack/all_candidates.csv \\
        --project openstack \\
        --out data/openstack/model_openstack.pkl

    # tune negative sampling / window / forest size
    python train_model.py --root ... --csv ... --project openstack --out ... \\
        --window-days 14 --neg-ratio 3 --min-neg-per-patch 2 \\
        --n-estimators 300 --test-size 0.2 --random-state 42
"""

from sklearn.model_selection import RandomizedSearchCV
import argparse
import ast
import json
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Make `from core.utils import get_path_similarity_stats` work whether this
# script lives at the repo root (next to app.py / the core/ package) or
# somewhere else — add the parent of core/ to sys.path if needed.
try:
    from core.utils import get_path_similarity_stats
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.utils import get_path_similarity_stats

# Reuse build_dataset.py's own patch_id derivation so the ids we compute
# while walking linkages.json are guaranteed to match what ended up in
# the CSV — do NOT reimplement this logic separately, it WILL drift.
try:
    from build_dataset import extract_ids
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_dataset import extract_ids


# ----------------------------------------------------------------------
# Label extraction
# ----------------------------------------------------------------------

def load_positive_pairs(gt_csv, known_ids):
    """
    Load positive links from the ground_truth.csv dataset,
    filtering for patches that exist in `known_ids`.
    """
    df_gt = pd.read_csv(gt_csv, dtype=str)
    pairs = set()
    for _, row in df_gt.iterrows():
        a = row["source_patch_id"]
        b = row["target_patch_id"]
        if a in known_ids and b in known_ids:
            if a != b:
                pairs.add((a, b))
    return list(pairs)


# ----------------------------------------------------------------------
# Candidate windowing (mirrors SmartPatchEngine.get_candidates, but fast)
# ----------------------------------------------------------------------
#
# The original approach — a full boolean mask over the whole dataframe,
# called once per patch — is O(n) per call and O(n^2) overall. At 200k
# patches that's ~4*10^10 element comparisons, which is what "gets stuck".
#
# Instead: sort all patches by created_time once, then for each patch use
# binary search (np.searchsorted) to jump straight to its window's index
# range. That's O(log n) per patch, O(n log n) total.

class TimeIndex:
    def __init__(self, df):
        order = np.argsort(df["created_time"].values)
        self.sorted_times = df["created_time"].values[order]  # datetime64[ns], ascending
        self.sorted_ids = df["patch_id"].values[order]

    def window_ids(self, ref_time, days, exclude_id):
        ref_time = np.datetime64(ref_time)
        start = ref_time - np.timedelta64(days, "D")
        end = ref_time + np.timedelta64(days, "D")
        lo = np.searchsorted(self.sorted_times, start, side="left")
        hi = np.searchsorted(self.sorted_times, end, side="right")
        ids = self.sorted_ids[lo:hi]
        return [i for i in ids if i != exclude_id]


# ----------------------------------------------------------------------
# Feature construction (mirrors SmartPatchEngine.predict exactly)
# ----------------------------------------------------------------------

def build_feature_row(query_row, cand_row, embeddings):
    """
    query_row  ~= patch_ref in engine.py's predict()
    cand_row   ~= a row being scored against patch_ref

    Matches engine.py: files_A = candidate, files_B = query.
    """
    file_stats = get_path_similarity_stats(cand_row["files_parsed"], query_row["files_parsed"])

    emb_q = embeddings[query_row["patch_id"]]
    emb_c = embeddings[cand_row["patch_id"]]
    sim_cosine = cosine_similarity(emb_q.reshape(1, -1), emb_c.reshape(1, -1))[0][0]

    delta_time_hours = abs((cand_row["created_time"] - query_row["created_time"]).total_seconds() / 3600)

    files_q = query_row["files_parsed"]
    files_c = cand_row["files_parsed"]

    # Base features (must match engine.py)
    feats = {
        **file_stats,
        "sim_cosine": float(sim_cosine),
        "delta_time_hours": delta_time_hours,
        "len_A": len(files_c),
        "len_B": len(files_q),
    }

    # --- Extra file-path structural features ---
    set_q = set(files_q)
    set_c = set(files_c)
    intersection = set_q & set_c
    union = set_q | set_c
    feats["file_set_jaccard"] = len(intersection) / max(len(union), 1)
    feats["has_shared_directory"] = int(any(
        os.path.dirname(a) == os.path.dirname(b)
        for a in files_c for b in files_q
    ))

    # --- Log-scaled and day-level time features ---
    feats["delta_time_days"] = delta_time_hours / 24
    feats["log_delta_time_hours"] = np.log1p(delta_time_hours)

    # --- File count ratio ---
    len_a, len_b = len(files_c), len(files_q)
    feats["file_count_ratio"] = min(len_a, len_b) / max(len_a, len_b, 1)

    return feats


def compute_embeddings(df, sbert, all_ids=None):
    """
    Encode (title + " " + description) for every patch in `df` (indexed by
    patch_id) with SBERT. Shared by training and evaluation so both use the
    exact same text construction.

    Returns:
        embeddings: dict patch_id -> vector
        vecs: the raw (n, dim) array, aligned to all_ids
        all_ids: the id order used
    """
    if all_ids is None:
        all_ids = list(df.index)
    print(f"   -> Encoding {len(all_ids)} unique patches with SBERT...")
    texts = [
        (df.loc[pid, "title"] or "") + " " + (df.loc[pid, "description"] or "")
        for pid in all_ids
    ]
    vecs = sbert.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = {pid: vecs[i] for i, pid in enumerate(all_ids)}
    return embeddings, vecs, all_ids


import hashlib, pickle

def compute_embeddings_cached(df, sbert, all_ids=None, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"   -> Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    result = compute_embeddings(df, sbert, all_ids)
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"   -> Cached embeddings to {cache_path}")
    return result



# ----------------------------------------------------------------------
# Dataset assembly
# ----------------------------------------------------------------------

def split_linked_patches_temporal(df, positive_pairs, test_size=0.3):
    """
    Target-patch ("B") temporal split.

    Collects every unique TARGET patch (the "Bs" in A->B links).
    Sorts them by created_time (ascending) and designates:

        first (1-test_size) fraction  →  train_bs   (used during model.fit)
        last  test_size    fraction   →  test_bs    (held out for MRR / Recall@k)

    Returns:
        train_bs : set of patch_ids in the training split
        test_bs  : set of patch_ids in the test split (newest target patches)
    """
    target_ids = set(target_id for _, target_id in positive_pairs)

    df_linked = (
        df[df["patch_id"].isin(target_ids)]
        .sort_values("created_time")
        .reset_index(drop=True)
    )

    if df_linked.empty:
        return set(), set()

    n_test = max(1, int(len(df_linked) * test_size))
    n_train = len(df_linked) - n_test
    train_bs = set(df_linked["patch_id"].iloc[:n_train])
    test_bs  = set(df_linked["patch_id"].iloc[n_train:])
    return train_bs, test_bs


# Keep the old name as an alias so external callers don't break.
split_patches_temporal = split_linked_patches_temporal


def assemble_training_set(df, active_pairs, window_days, neg_ratio, min_neg_per_patch,
                           sbert, seed, all_positive_pairs=None, test_bs=None, embeddings_cache_path=None):
    """
    Build the (X, y) feature matrix for training or testing.

    active_pairs : List of positive (source, target) tuples to use as positive examples.
    all_positive_pairs : Used to build adjacency matrix to avoid sampling true links as negatives.
    test_bs : Set of test target IDs. If a candidate is in test_bs, it is skipped.
    """
    rng = random.Random(seed)

    df = df.set_index("patch_id", drop=False)
    all_ids = list(df.index)

    if all_positive_pairs is None:
        all_positive_pairs = active_pairs

    # Map each id to the set of positively-linked ids (undirected adjacency)
    pos_adjacency = {pid: set() for pid in all_ids}
    for a, b in all_positive_pairs:
        if a in pos_adjacency:
            pos_adjacency[a].add(b)
        if b in pos_adjacency:
            pos_adjacency[b].add(a)

    embeddings, _vecs, _ids = compute_embeddings_cached(df, sbert, all_ids ,cache_path=embeddings_cache_path)

    rows = []
    labels = []
    used_ordered_pairs = set()  # (query_id, cand_id) already emitted, avoid exact dupes

    def add_pair(query_id, cand_id, label):
        # Enforce that no test target is used as a candidate during training
        if test_bs is not None and cand_id in test_bs:
            return
        key = (query_id, cand_id)
        if key in used_ordered_pairs:
            return
        used_ordered_pairs.add(key)
        feats = build_feature_row(df.loc[query_id], df.loc[cand_id], embeddings)
        rows.append(feats)
        labels.append(label)

    n_pos_added = 0
    for a, b in tqdm(active_pairs, desc="   Positive pairs"):
        if a not in df.index or b not in df.index:
            continue
        add_pair(a, b, 1)
        n_pos_added += 1

    print("   -> Building time index for fast window lookups...")
    time_index = TimeIndex(df)

    n_neg_added = 0
    # Collect queries that we want to sample negatives for.
    # To keep it balanced, we'll sample for patches that are present in active_pairs
    # + a random sample of other patches if we want (the original code did all_ids).
    # Doing all_ids is fine, add_pair will filter out test_bs candidates.
    for pid in tqdm(all_ids, desc="   Negative sampling"):
        row = df.loc[pid]
        neg_pool_all = time_index.window_ids(row["created_time"], window_days, pid)
        if not neg_pool_all:
            continue

        linked = pos_adjacency.get(pid, set())
        # Negatives must not be true links, and must not be test targets (if test_bs provided)
        neg_pool = [cid for cid in neg_pool_all if cid not in linked]
        if test_bs is not None:
            neg_pool = [cid for cid in neg_pool if cid not in test_bs]
            
        if not neg_pool:
            continue

        n_links_for_pid = len(linked)
        target_negs = max(min_neg_per_patch, neg_ratio * n_links_for_pid) if n_links_for_pid else min_neg_per_patch
        target_negs = min(target_negs, len(neg_pool))

        sampled = rng.sample(neg_pool, target_negs) if target_negs > 0 else []
        for cid in sampled:
            add_pair(pid, cid, 0)
            n_neg_added += 1

    print(f"   -> Assembled {n_pos_added} positive rows, {n_neg_added} negative rows "
          f"({len(rows)} total after de-dup)")

    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


# ----------------------------------------------------------------------
# Ranking evaluation (Recall@k / MRR)
# ----------------------------------------------------------------------

RECALL_K_VALUES = [1, 2, 4, 6, 8, 10]
EVAL_WINDOWS    = [2, 7, 14, 30]  # days


def compute_ranking_metrics(model, df, test_pos_pairs, positive_pairs, embeddings, k_values=None, window_days_list=None):
    """
    For each evaluation window W and each test query A:
      - candidates = all patches within ±W days of A (except A itself)
      - score every candidate with the trained model
      - rank by descending score
      - for each positive target B of A that lies in the candidate list:
          * reciprocal rank = 1 / rank_of_B (1-indexed)
          * hit@k           = 1 if rank_of_B <= k else 0

    Returns a dict:
        results[window] = {
            'MRR':      float,
            'R@{k}':    float,   # for each k in k_values
            'n_queries': int,    # how many (query, target) pairs were evaluated
        }

    Queries where the positive target does not fall inside the window are
    counted in the denominator (reciprocal rank = 0, hit@k = 0).
    """
    if k_values is None:
        k_values = RECALL_K_VALUES
    if window_days_list is None:
        window_days_list = EVAL_WINDOWS

    df_idx = df.set_index("patch_id", drop=False)
    time_index = TimeIndex(df_idx)

    # Build adjacency for ALL positive pairs so we know what is truly positive
    all_pos_set = set()
    for a, b in positive_pairs:
        all_pos_set.add((a, b))
        all_pos_set.add((b, a))

    # Get the feature column order expected by the model
    feature_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None

    results = {}

    for window in window_days_list:
        rr_list   = []   # one entry per (query, target) pair
        hits       = {k: [] for k in k_values}
        n_queries  = 0

        # Iterate over every test (source, target) pair
        for query_id, target_id in tqdm(
            test_pos_pairs,
            desc=f"   Ranking eval W={window}d",
            leave=False,
        ):
            if query_id not in df_idx.index or target_id not in df_idx.index:
                continue

            query_row = df_idx.loc[query_id]
            cand_ids  = time_index.window_ids(query_row["created_time"], window, query_id)

            n_queries += 1

            if not cand_ids or target_id not in cand_ids:
                # Target outside the window — counts as rank=∞
                rr_list.append(0.0)
                for k in k_values:
                    hits[k].append(0)
                continue

            # Build feature matrix for all candidates
            feat_rows = []
            for cid in cand_ids:
                if cid not in df_idx.index:
                    feat_rows.append(None)
                    continue
                feat_rows.append(build_feature_row(query_row, df_idx.loc[cid], embeddings))

            # Filter out None entries while keeping cand_ids aligned
            valid_cids   = [cid for cid, f in zip(cand_ids, feat_rows) if f is not None]
            valid_feats  = [f   for f in feat_rows                    if f is not None]

            if not valid_feats:
                rr_list.append(0.0)
                for k in k_values:
                    hits[k].append(0)
                continue

            X_cands = pd.DataFrame(valid_feats)
            if feature_cols is not None:
                for col in feature_cols:
                    if col not in X_cands.columns:
                        X_cands[col] = 0.0
                X_cands = X_cands[feature_cols]

            scores = model.predict_proba(X_cands)[:, 1]

            # Rank candidates by descending score (rank 1 = best)
            order       = np.argsort(-scores)
            ranked_cids = [valid_cids[i] for i in order]

            if target_id in ranked_cids:
                rank = ranked_cids.index(target_id) + 1  # 1-indexed
                rr_list.append(1.0 / rank)
                for k in k_values:
                    hits[k].append(1 if rank <= k else 0)
            else:
                rr_list.append(0.0)
                for k in k_values:
                    hits[k].append(0)

        mrr = float(np.mean(rr_list)) if rr_list else 0.0
        recall_at_k = {k: float(np.mean(hits[k])) if hits[k] else 0.0 for k in k_values}

        results[window] = {
            "MRR":      mrr,
            **{f"R@{k}": recall_at_k[k] for k in k_values},
            "n_queries": n_queries,
        }

    return results


def print_ranking_table(results, k_values=None, window_days_list=None):
    """Pretty-print Recall@k / MRR results as a table."""
    if k_values is None:
        k_values = RECALL_K_VALUES
    if window_days_list is None:
        window_days_list = EVAL_WINDOWS

    header_cols = ["Window", "Queries", "MRR"] + [f"R@{k}" for k in k_values]
    col_w = max(len(c) for c in header_cols) + 2

    header = "".join(c.ljust(col_w) for c in header_cols)
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for window in window_days_list:
        if window not in results:
            continue
        r   = results[window]
        row = [
            f"{window}d",
            str(r["n_queries"]),
            f"{r['MRR']:.4f}",
        ] + [f"{r[f'R@{k}']:.4f}" for k in k_values]
        print("".join(c.ljust(col_w) for c in row))
    print(sep)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to all_candidates.csv produced by build_dataset.py")
    parser.add_argument("--gt-csv", required=True, help="Path to ground_truth.csv produced by build_dataset.py")
    parser.add_argument("--project", required=True, help="Project key, e.g. openstack")
    parser.add_argument("--out", required=True, help="Output path for the .pkl, e.g. data/openstack/model_openstack.pkl")
    parser.add_argument("--window-days", type=int, default=14, help="Must match engine.py's default window (14)")
    parser.add_argument("--neg-ratio", type=int, default=2, help="Negatives sampled per positive link, per patch")
    parser.add_argument("--min-neg-per-patch", type=int, default=2, help="Negatives sampled even for patches with zero links")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.3,
                         help="Fraction of LINKED patches (sorted by time) held out for test (default 0.3 = 30%%)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    print(f"Loading dataset: {args.csv}")
    df = pd.read_csv(args.csv, dtype={"patch_id": str})
    # NOTE: pd.read_csv's parse_dates auto-detection silently fails on the
    # "...000000000" (9-digit nanosecond) timestamp format build_dataset.py
    # writes — it leaves the column as plain strings with no error. Parse
    # explicitly instead; errors="coerce" turns any unparsable value into
    # NaT so the dropna below catches it rather than crashing later deep
    # inside feature building.
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"   ⚠️  Dropped {n_before - len(df)} rows with missing/unparsable patch_id or created_time")
    df["files_parsed"] = df["files"].apply(lambda x: x if isinstance(x, list) else _safe_parse_list(x))
    known_ids = set(df["patch_id"])
    print(f"   -> {len(df)} patches loaded")

    print(f"Loading positive links from {args.gt_csv} ...")
    positive_pairs = load_positive_pairs(args.gt_csv, known_ids)
    print(f"   -> usable positive pairs: {len(positive_pairs)}")

    if not positive_pairs:
        print("⚠️  No usable positive pairs found — can't train a meaningful classifier. Stopping.")
        sys.exit(1)

    print("Loading SBERT (all-MiniLM-L6-v2 / microsoft/codebert-base / all-mpnet-base-v2)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Target-patch ("B") temporal split ───────────────────────────────────
    print(f"Computing target-patch temporal split "
          f"(train={1-args.test_size:.0%} / test={args.test_size:.0%} by creation time)...")
    train_bs, test_bs = split_linked_patches_temporal(
        df, positive_pairs, test_size=args.test_size
    )
    print(f"   -> target patches in train: {len(train_bs)}   in test (newest): {len(test_bs)}")

    # We train on pairs where the target (B) is in train_bs
    train_pos_pairs = [
        p for p in positive_pairs if p[1] in train_bs
    ]
    # We test on pairs where the target (B) is in test_bs
    test_pos_pairs = [
        p for p in positive_pairs if p[1] in test_bs
    ]
    print(f"   -> train positive pairs (target in train_bs): {len(train_pos_pairs)}   "
          f"test positive pairs (target in test_bs): {len(test_pos_pairs)}")

    print("Assembling training pairs + features (this mirrors engine.py's predict() feature code)...")
    X_train, y_train = assemble_training_set(
        df, active_pairs=train_pos_pairs,
        window_days=args.window_days,
        neg_ratio=args.neg_ratio,
        min_neg_per_patch=args.min_neg_per_patch,
        sbert=sbert,
        seed=args.random_state,
        all_positive_pairs=positive_pairs,
        test_bs=test_bs, # exclude test targets from negative candidates
        embeddings_cache_path=f"data/{args.project}/embeddings_train_cache.pkl"
    )

    print(f"Training class balance: {int(y_train.sum())} positive / "
          f"{int((y_train == 0).sum())} negative ({y_train.mean():.1%} positive)")

    print(f"Training RandomForestClassifier (n_estimators={args.n_estimators})...")
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth":    [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
    }
    scale_pos_weight = (y_train == 0).sum() / y_train.sum()
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", n_jobs=-1),
        # XGBClassifier(scale_pos_weight=scale_pos_weight, n_jobs=-1, learning_rate=0.05),
        param_dist, n_iter=20, cv=3, scoring="roc_auc",
        random_state=args.random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    print("Assembling test pairs + features (test targets only, never seen during fit)...")
    X_test, y_test = assemble_training_set(
        df, active_pairs=test_pos_pairs,
        window_days=args.window_days,
        neg_ratio=args.neg_ratio,
        min_neg_per_patch=args.min_neg_per_patch,
        sbert=sbert,
        seed=args.random_state,
        all_positive_pairs=positive_pairs,
        test_bs=None, # no need to restrict candidates for the test set
        embeddings_cache_path=f"data/{args.project}/embeddings_test_cache.pkl"
    )

    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(X_test.columns)
        for c in missing:
            X_test[c] = 0
        X_test = X_test[model.feature_names_in_]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n--- Evaluation on held-out test split (patch-level temporal) ---")
    print(classification_report(y_test, preds, digits=3))
    try:
        print(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")
    except ValueError:
        pass

    importances = sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda t: -t[1])
    print("\nTop feature importances:")
    for name, imp in importances[:10]:
        print(f"   {name:20s} {imp:.3f}")

    # ── Ranking evaluation: Recall@k and MRR ────────────────────────────────
    print("\n--- Ranking metrics: Recall@k & MRR (test queries only) ---")
    print(f"    k values  : {RECALL_K_VALUES}")
    print(f"    windows   : {EVAL_WINDOWS} days")

    # Re-use the already-computed test embeddings cache
    embeddings_for_ranking, _, _ = compute_embeddings_cached(
        df.set_index("patch_id", drop=False),
        sbert,
        all_ids=list(df["patch_id"]),
        cache_path=f"data/{args.project}/embeddings_test_cache.pkl",
    )

    ranking_results = compute_ranking_metrics(
        model,
        df,
        test_pos_pairs,
        positive_pairs,
        embeddings_for_ranking,
        k_values=RECALL_K_VALUES,
        window_days_list=EVAL_WINDOWS,
    )
    print_ranking_table(ranking_results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\n✅ Saved model -> {args.out}")


def _safe_parse_list(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


if __name__ == "__main__":
    main()