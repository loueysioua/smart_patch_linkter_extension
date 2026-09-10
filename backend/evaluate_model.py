"""
evaluate_model.py

Evaluates an already-trained model_<project>.pkl three ways:

1) PAIR-LEVEL CLASSIFICATION METRICS (precision/recall/F1, confusion
   matrix, ROC-AUC, PR-AUC) on a held-out test set.

   To do this without leaking training data into the evaluation, this
   script reconstructs the exact same (X, y) that train_model.py built —
   same positive/negative pair assembly, same SBERT encoding, same
   random seed — and re-runs the identical train_test_split() call. That
   recovers the exact rows that were held out and never passed to
   model.fit(), as long as you pass the SAME --window-days, --neg-ratio,
   --min-neg-per-patch, --test-size, and --random-state you used when you
   ran train_model.py. If any of those differ, the "held-out" set won't
   line up with what was actually excluded from training, and the numbers
   will be optimistic. If you're not sure what you trained with, rerun
   train_model.py first and note the args, or just re-train (it prints
   these same metrics at the end already).

2) RANKING METRICS (Recall@k, MRR) — the metric that actually reflects
   the deployed behavior. /predict_topk doesn't classify a single pair;
   for a query patch it scores every candidate in its time window and
   returns the top-k. So for every patch with a *known* related patch,
   this script pulls its real time-window candidate pool (same window
   the engine would use), scores all of them with the model, and checks
   what rank the true related patch landed at. Recall@5 = 0.80 means:
   80% of the time, the actually-related patch was in the top 5 the
   engine would have shown.

   Caveat: this evaluates over ALL patches (not just the held-out split),
   because query/candidate roles overlap in complicated ways across the
   graph of links — it should be read as "how good is this model at the
   job it's deployed for", not as a strict generalization bound. Use (1)
   for the strict generalization number, (2) for the product-relevant one.

3) LLM RERANKING METRICS (MRR + Recall@k after gpt reranking) — takes
   the ML model's full ranked candidate list for each test query and sends
   the top-k entries to the LLM for semantic reranking.  The post-LLM rank
   is then used to compute MRR and Recall@k.  When the true link was NOT
   among the top-k candidates shown to the LLM, the ML model's original
   rank is kept as a fallback, so every test query contributes to the
   denominator and the numbers are directly comparable to part (2).

Usage:
    python evaluate_model.py \\
        --root /path/to/gerrit_dump \\
        --csv data/openstack/all_candidates.csv \\
        --project openstack \\
        --model data/openstack/model_openstack.pkl \\
        --window-days 14 --neg-ratio 3 --min-neg-per-patch 2 \\
        --test-size 0.2 --random-state 42 \\
        --ranking-sample 3000
"""

import argparse
import os
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
# train_test_split no longer used here — split is done at the patch level
# via tm.split_patches_temporal() so both scripts agree on what's "test".

try:
    import train_model as tm
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_model as tm

get_path_similarity_stats = tm.get_path_similarity_stats


# ----------------------------------------------------------------------
# Part 1: pair-level classification metrics on the held-out split
# ----------------------------------------------------------------------

def evaluate_classification(df, positive_pairs, model, sbert, args):
    print("Computing linked-patch temporal split "
          f"(train={1-args.test_size:.0%} / test={args.test_size:.0%} by creation time)...")
    train_bs, test_bs = tm.split_linked_patches_temporal(
        df, positive_pairs, test_size=args.test_size
    )
    print(f"   -> linked patches in train: {len(train_bs)}   in test (newest): {len(test_bs)}")

    test_pos_pairs = [p for p in positive_pairs if p[1] in test_bs]

    print("Assembling test-set feature rows (test Bs never seen during model.fit)...")
    X_test, y_test = tm.assemble_training_set(
        df, active_pairs=test_pos_pairs,
        window_days=args.window_days,
        neg_ratio=args.neg_ratio,
        min_neg_per_patch=args.min_neg_per_patch,
        sbert=sbert,
        seed=args.random_state,
        all_positive_pairs=positive_pairs,
        test_bs=None,  # no need to restrict candidates for the test set
    )

    if y_test.size == 0:
        print("\u26a0\ufe0f  No test rows assembled — try a smaller --test-size or check that "
              "positive pairs exist within the test patch window.")
        return {}

    # Align columns exactly like engine.py / train_model.py do at inference time
    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(X_test.columns)
        for c in missing:
            X_test[c] = 0
        X_test = X_test[model.feature_names_in_]

    probs = model.predict_proba(X_test)[:, 1]
    preds_default = (probs >= 0.5).astype(int)

    print(f"\nHeld-out test set (linked-patch temporal, test Bs): {len(y_test)} rows "
          f"({int(y_test.sum())} positive / {int((y_test == 0).sum())} negative, "
          f"{y_test.mean():.1%} positive)")

    print("\n--- Classification report @ threshold 0.5 ---")
    print(classification_report(y_test, preds_default, digits=3))

    cm = confusion_matrix(y_test, preds_default)
    print("Confusion matrix @ 0.5  (rows=actual, cols=predicted, [0,1]):")
    print(cm)

    try:
        auc = roc_auc_score(y_test, probs)
        print(f"\nROC-AUC: {auc:.3f}")
    except ValueError:
        auc = None
        print("\nROC-AUC: undefined (only one class present in y_test)")

    try:
        pr_auc = average_precision_score(y_test, probs)
        print(f"PR-AUC (average precision): {pr_auc:.3f}  "
              f"(baseline for a random model \u2248 {y_test.mean():.3f}, i.e. the positive rate)")
    except ValueError:
        pr_auc = None

    # Best-F1 threshold, since 0.5 is rarely optimal for an imbalanced task
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9),
        0,
    )
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) else None
    if best_idx is not None:
        best_thr = thresholds[best_idx]
        preds_best = (probs >= best_thr).astype(int)
        print(f"\n--- Classification report @ best-F1 threshold ({best_thr:.3f}) ---")
        print(classification_report(y_test, preds_best, digits=3))

    # Simple calibration check: within each predicted-probability bucket,
    # what fraction were actually positive? A well-calibrated model should
    # have these roughly line up (bucket 0.7-0.8 -> ~70-80% actually positive).
    print("--- Calibration (predicted probability bucket -> actual positive rate) ---")
    bins = np.linspace(0, 1, 11)
    bucket_idx = np.digitize(probs, bins) - 1
    for b in range(10):
        mask = bucket_idx == b
        n = mask.sum()
        if n == 0:
            continue
        actual_rate = y_test[mask].mean()
        print(f"   [{bins[b]:.1f}-{bins[b+1]:.1f}) n={n:6d}   "
              f"predicted~{(bins[b]+bins[b+1])/2:.2f}   actual={actual_rate:.3f}")

    return {"roc_auc": auc, "pr_auc": pr_auc, "confusion_matrix": cm.tolist()}


# ----------------------------------------------------------------------
# Part 2: ranking metrics (Recall@k, MRR) — mirrors /predict_topk exactly
# ----------------------------------------------------------------------

def evaluate_ranking(df, positive_pairs, model, sbert, window_days,
                      ks=(1, 2, 4, 6, 8, 10), sample_queries=None, seed=42,
                      vecs=None, id_to_idx=None, quiet=False):
    """
    vecs / id_to_idx: pass these in (from a prior compute_embeddings call)
    to skip re-encoding — useful when sweeping multiple window_days values
    against the same model, since embeddings don't depend on the window.
    """
    df = df.set_index("patch_id", drop=False)
    all_ids = list(df.index)

    pos_adjacency = {}
    for pair in positive_pairs:
        a, b = tuple(pair)
        if a in df.index:
            pos_adjacency.setdefault(a, set()).add(b)
        if b in df.index:
            pos_adjacency.setdefault(b, set()).add(a)

    if vecs is None or id_to_idx is None:
        _embeddings, vecs, ids_order = tm.compute_embeddings(df, sbert, all_ids)
        id_to_idx = {pid: i for i, pid in enumerate(ids_order)}

    time_index = tm.TimeIndex(df)

    query_ids = list(pos_adjacency.keys())
    rng = random.Random(seed)
    if sample_queries and len(query_ids) > sample_queries:
        if not quiet:
            print(f"Sampling {sample_queries} of {len(query_ids)} linked patches for ranking eval "
                  f"(use --ranking-sample 0 to evaluate all)")
        query_ids = rng.sample(query_ids, sample_queries)

    ranks = []
    not_in_window = 0
    candidate_counts = []

    for pid in tqdm(query_ids, desc=f"Ranking eval (window={window_days}d)", disable=quiet):
        linked = pos_adjacency[pid]
        query_row = df.loc[pid]
        cand_ids = [c for c in time_index.window_ids(query_row["created_time"], window_days, pid)
                    if c in df.index]
        if not cand_ids:
            continue
        candidate_counts.append(len(cand_ids))

        q_emb = vecs[id_to_idx[pid]]
        c_emb = vecs[[id_to_idx[c] for c in cand_ids]]
        sims = cosine_similarity(q_emb.reshape(1, -1), c_emb)[0]

        q_files = query_row["files_parsed"]
        feats_list = []
        for cid, sim in zip(cand_ids, sims):
            cand_row = df.loc[cid]
            c_files = cand_row["files_parsed"]
            file_stats = get_path_similarity_stats(c_files, q_files)
            delta_hours = abs((cand_row["created_time"] - query_row["created_time"]).total_seconds() / 3600)
            set_q, set_c = set(q_files), set(c_files)
            len_a, len_b = len(c_files), len(q_files)
            feats_list.append({
                **file_stats,
                "sim_cosine": float(sim),
                "delta_time_hours": delta_hours,
                "len_A": len_a,
                "len_B": len_b,
                "file_set_jaccard": len(set_q & set_c) / max(len(set_q | set_c), 1),
                "has_shared_directory": int(any(
                    os.path.dirname(a) == os.path.dirname(b)
                    for a in c_files for b in q_files
                )),
                "delta_time_days": delta_hours / 24,
                "log_delta_time_hours": np.log1p(delta_hours),
                "file_count_ratio": min(len_a, len_b) / max(len_a, len_b, 1),
            })

        X = pd.DataFrame(feats_list)
        if hasattr(model, "feature_names_in_"):
            missing = set(model.feature_names_in_) - set(X.columns)
            for c in missing:
                X[c] = 0
            X = X[model.feature_names_in_]
        scores = model.predict_proba(X)[:, 1]

        order = np.argsort(-scores)
        ranked_ids = [cand_ids[i] for i in order]
        rank_lookup = {cid: r + 1 for r, cid in enumerate(ranked_ids)}  # 1-indexed

        true_in_window = [t for t in linked if t in rank_lookup]
        if not true_in_window:
            not_in_window += 1
            continue
        ranks.append(min(rank_lookup[t] for t in true_in_window))

    ranks = np.array(ranks)
    n = len(ranks)

    if not quiet:
        print(f"\nQueries with >=1 known link, evaluated: {len(query_ids)}")
        print(f"   -> true link WAS in the time window (rankable): {n}")
        print(f"   -> true link was outside the {window_days}-day window "
              f"(engine can never surface it, regardless of model quality): {not_in_window}")
        if candidate_counts:
            print(f"   -> avg candidates per query window: {np.mean(candidate_counts):.0f} "
                  f"(median {np.median(candidate_counts):.0f})")

    if n == 0:
        if not quiet:
            print("No rankable queries — can't compute ranking metrics.")
        return {}

    results = {
        "mrr": float(np.mean(1.0 / ranks)), "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "queries_evaluated": int(len(query_ids)), "queries_rankable": int(n),
        "queries_outside_window": int(not_in_window),
    }
    if not quiet:
        print(f"\nMRR (mean reciprocal rank): {results['mrr']:.3f}")
        print(f"Mean rank of true link: {results['mean_rank']:.1f}   Median: {results['median_rank']:.0f}")
        print("\nRecall@k (true link within top-k of the window's candidates):")
    for k in ks:
        r = float(np.mean(ranks <= k))
        results[f"recall@{k}"] = r
        if not quiet:
            print(f"   Recall@{k:<3d} {r:.3f}")

    return results


# ----------------------------------------------------------------------
# Part 3: LLM reranking metrics (MRR + Recall@k on top of ML results)
# ----------------------------------------------------------------------

def _load_gpt_clients():
    """Load all available OpenAI API keys and return a list of (client, key_label) tuples."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  openai not installed — skipping LLM reranking eval.\n"
              "   Run: pip install openai")
        return []

    clients = []
    # Primary key
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        clients.append((OpenAI(api_key=key), "OPENAI_API_KEY"))
    # Rotated extra keys
    for i in range(2, 20):
        key = os.environ.get(f"OPENAI_API_KEY_{i}")
        if key:
            clients.append((OpenAI(api_key=key), f"OPENAI_API_KEY_{i}"))
    return clients


def _llm_rerank_candidates(client, df, query_pid, ranked_candidate_ids, top_k, llm_model):
    """
    Ask the LLM to rerank the first `top_k` candidates in `ranked_candidate_ids`.
    Returns a list of all candidate IDs in the new order:
      - first top_k slots: reranked by LLM
      - remaining slots: kept in original ML order
    On any LLM failure, returns the original order unchanged.
    """
    from llm_rerank import rerank_candidates  # local import to keep it optional

    df_idx = df.set_index("patch_id", drop=False) if df.index.name != "patch_id" else df

    def _row_to_dict(pid):
        if pid not in df_idx.index:
            return {"patch_id": pid, "title": pid, "description": "", "score": 0.0}
        r = df_idx.loc[pid]
        return {
            "patch_id": pid,
            "title": str(r.get("title", "") or ""),
            "description": str(r.get("description", "") or ""),
            "score": 0.0,  # scores not needed for reranking logic
        }

    query_dict = _row_to_dict(query_pid)
    candidates_for_llm = [_row_to_dict(pid) for pid in ranked_candidate_ids[:top_k]]
    tail = ranked_candidate_ids[top_k:]

    reranked = rerank_candidates(client, query_dict, candidates_for_llm, model_name=llm_model)
    reranked_ids = [c["patch_id"] for c in reranked]

    # Append any tail candidates the LLM didn't see
    return reranked_ids + list(tail)


def evaluate_llm_reranking(
    df, positive_pairs, model, sbert, window_days,
    ks=(1, 2, 4, 6, 8, 10),
    sample_queries=None, seed=42,
    vecs=None, id_to_idx=None,
    llm_top_k=10,
    llm_model="gpt-4o-mini",
    rate_limit_delay=4.0,
):
    """
    Part 3 — Evaluate MRR + Recall@k after LLM reranking of the ML model's output.

    For each test-set query:
      1. Run the ML model to get a full ranked list of window candidates.
      2. Send the top `llm_top_k` candidates to the gpt LLM for reranking.
      3. Compute the rank of the true link in the final (LLM-reranked) list.
         Fallback: if the true link was outside the LLM's view (rank > llm_top_k),
         use the original ML rank so every query contributes to the denominator.

    Parameters
    ----------
    rate_limit_delay : float
        Seconds to sleep between LLM calls to stay within free-tier rate limits.
        Rotates across all OPENAI_API_KEY_* environment variables.
    """
    print("Loading Openai clients (key rotation)...")
    clients = _load_gpt_clients()
    if not clients:
        print("❌  No OpenAI API keys found — skipping LLM reranking evaluation.")
        return {}

    print(f"   -> {len(clients)} API key(s) loaded for rotation.")

    df = df.set_index("patch_id", drop=False) if df.index.name != "patch_id" else df
    all_ids = list(df.index)

    pos_adjacency = {}
    for pair in positive_pairs:
        a, b = tuple(pair)
        if a in df.index:
            pos_adjacency.setdefault(a, set()).add(b)
        if b in df.index:
            pos_adjacency.setdefault(b, set()).add(a)

    if vecs is None or id_to_idx is None:
        _embeddings, vecs, ids_order = tm.compute_embeddings(df, sbert, all_ids)
        id_to_idx = {pid: i for i, pid in enumerate(ids_order)}

    time_index = tm.TimeIndex(df)

    query_ids = list(pos_adjacency.keys())
    rng = random.Random(seed)
    if sample_queries and len(query_ids) > sample_queries:
        print(f"Sampling {sample_queries} of {len(query_ids)} queries for LLM reranking eval")
        query_ids = rng.sample(query_ids, sample_queries)

    ranks = []          # final rank of true link (post-LLM where visible, ML otherwise)
    ml_ranks = []       # pure-ML rank of same queries (baseline for comparison)
    not_in_window = 0
    client_idx = 0

    for pid in tqdm(query_ids, desc=f"LLM reranking eval (window={window_days}d, top_k={llm_top_k})"):
        linked = pos_adjacency[pid]
        query_row = df.loc[pid]
        cand_ids = [
            c for c in time_index.window_ids(query_row["created_time"], window_days, pid)
            if c in df.index
        ]
        if not cand_ids:
            not_in_window += 1
            continue

        # ---- ML ranking (same logic as evaluate_ranking) ----
        q_emb = vecs[id_to_idx[pid]]
        c_emb = vecs[[id_to_idx[c] for c in cand_ids]]
        sims = cosine_similarity(q_emb.reshape(1, -1), c_emb)[0]

        q_files = query_row["files_parsed"]
        feats_list = []
        for cid, sim in zip(cand_ids, sims):
            cand_row = df.loc[cid]
            c_files = cand_row["files_parsed"]
            file_stats = get_path_similarity_stats(c_files, q_files)
            delta_hours = abs(
                (cand_row["created_time"] - query_row["created_time"]).total_seconds() / 3600
            )
            set_q, set_c = set(q_files), set(c_files)
            len_a, len_b = len(c_files), len(q_files)
            feats_list.append({
                **file_stats,
                "sim_cosine": float(sim),
                "delta_time_hours": delta_hours,
                "len_A": len_a,
                "len_B": len_b,
                "file_set_jaccard": len(set_q & set_c) / max(len(set_q | set_c), 1),
                "has_shared_directory": int(any(
                    os.path.dirname(a) == os.path.dirname(b)
                    for a in c_files for b in q_files
                )),
                "delta_time_days": delta_hours / 24,
                "log_delta_time_hours": np.log1p(delta_hours),
                "file_count_ratio": min(len_a, len_b) / max(len_a, len_b, 1),
            })

        X = pd.DataFrame(feats_list)
        if hasattr(model, "feature_names_in_"):
            missing = set(model.feature_names_in_) - set(X.columns)
            for c in missing:
                X[c] = 0
            X = X[model.feature_names_in_]
        scores = model.predict_proba(X)[:, 1]

        order = np.argsort(-scores)
        ml_ranked_ids = [cand_ids[i] for i in order]
        ml_rank_lookup = {cid: r + 1 for r, cid in enumerate(ml_ranked_ids)}

        true_in_window = [t for t in linked if t in ml_rank_lookup]
        if not true_in_window:
            not_in_window += 1
            continue

        ml_best_rank = min(ml_rank_lookup[t] for t in true_in_window)
        ml_ranks.append(ml_best_rank)

        # ---- LLM reranking of top-k ----
        current_client, key_label = clients[client_idx % len(clients)]
        try:
            llm_ranked_ids = _llm_rerank_candidates(
                current_client, df, pid, ml_ranked_ids, llm_top_k, llm_model
            )
            llm_rank_lookup = {cid: r + 1 for r, cid in enumerate(llm_ranked_ids)}
            final_best_rank = min(
                llm_rank_lookup.get(t, ml_rank_lookup.get(t, len(ml_ranked_ids) + 1))
                for t in true_in_window
            )
        except Exception as e:
            print(f"\n⚠️  LLM call failed for {pid} (key={key_label}): {e} — using ML rank")
            final_best_rank = ml_best_rank

        ranks.append(final_best_rank)

        # Rotate to next key and respect rate limit
        client_idx += 1
        time.sleep(rate_limit_delay)

    ranks = np.array(ranks)
    ml_ranks = np.array(ml_ranks)
    n = len(ranks)

    print(f"\nLLM Reranking — Queries evaluated: {len(query_ids)}")
    print(f"   -> true link in time window (rankable): {n}")
    print(f"   -> true link outside {window_days}-day window: {not_in_window}")

    if n == 0:
        print("No rankable queries — can't compute LLM reranking metrics.")
        return {}

    results = {
        "llm_mrr": float(np.mean(1.0 / ranks)),
        "ml_mrr_baseline": float(np.mean(1.0 / ml_ranks)),
        "queries_evaluated": int(len(query_ids)),
        "queries_rankable": int(n),
        "queries_outside_window": int(not_in_window),
    }

    print(f"\n{'Metric':<22} {'ML baseline':>14} {'After LLM rerank':>16}")
    print("-" * 54)
    print(f"{'MRR':<22} {results['ml_mrr_baseline']:>14.3f} {results['llm_mrr']:>16.3f}")
    print("\nRecall@k:")
    for k in ks:
        r_llm = float(np.mean(ranks <= k))
        r_ml  = float(np.mean(ml_ranks <= k))
        results[f"recall@{k}"] = r_llm
        results[f"ml_recall@{k}_baseline"] = r_ml
        print(f"   Recall@{k:<3d}  ML={r_ml:.3f}   LLM={r_llm:.3f}")

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to all_candidates.csv")
    parser.add_argument("--gt-csv", required=True, help="Path to ground_truth.csv")
    parser.add_argument("--project", required=True, help="Project key (directory name under --root)")
    parser.add_argument("--model", required=True, help="Path to the trained model_<project>.pkl")
    parser.add_argument("--window-days", type=int, default=14,
                         help="MUST match what you trained with and what engine.py serves with")
    parser.add_argument("--neg-ratio", type=int, default=2, help="Must match training, for part (1)")
    parser.add_argument("--min-neg-per-patch", type=int, default=2, help="Must match training, for part (1)")
    parser.add_argument("--test-size", type=float, default=0.3,
                         help="Fraction of LINKED patches (sorted by time) held out for test (default 0.3 = 30%%); must match training")
    parser.add_argument("--random-state", type=int, default=42, help="Must match training, for part (1)")
    parser.add_argument("--skip-classification", action="store_true",
                         help="Skip part (1)")
    parser.add_argument("--skip-ranking", action="store_true", help="Skip part (2)")
    parser.add_argument("--skip-llm-reranking", action="store_true",
                         help="Skip part (3) — LLM reranking evaluation")
    parser.add_argument("--ranking-sample", type=int, default=0,
                         help="Cap ranking eval to N linked patches for speed (0 = evaluate all)")
    parser.add_argument("--llm-rerank-topk", type=int, default=10,
                         help="Number of top ML candidates sent to the LLM for reranking (default 10)")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                         help="gpt model name for LLM reranking (default gpt-4o-mini)")
    parser.add_argument("--llm-rate-limit-delay", type=float, default=4.0,
                         help="Seconds to sleep between LLM API calls (default 4.0)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    loaded_obj = joblib.load(args.model)
    if isinstance(loaded_obj, dict) and "model" in loaded_obj:
        model = loaded_obj["model"]
        threshold = loaded_obj.get("threshold", 0.5)
        print(f"   -> Found saved threshold: {threshold:.3f}")
    else:
        model = loaded_obj
        threshold = 0.5

    print(f"Loading dataset: {args.csv}")
    df = pd.read_csv(args.csv, dtype={"patch_id": str})
    # Same fix as train_model.py: read_csv's parse_dates silently fails on
    # the 9-digit-nanosecond timestamp format, leaving strings behind with
    # no error. Parse explicitly.
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"   ⚠️  Dropped {n_before - len(df)} rows with missing/unparsable patch_id or created_time")
    df["files_parsed"] = df["files"].apply(lambda x: x if isinstance(x, list) else tm._safe_parse_list(x))
    known_ids = set(df["patch_id"])
    print(f"   -> {len(df)} patches loaded")

    print(f"Loading positive links from {args.gt_csv} ...")
    positive_pairs = tm.load_positive_pairs(args.gt_csv, known_ids)
    print(f"   -> usable positive pairs: {len(positive_pairs)}")
    if not positive_pairs:
        print("⚠️  No usable positive pairs found — can't evaluate.")
        sys.exit(1)

    print("Loading SBERT (microsoft/codebert-base)...")
    sbert = SentenceTransformer("microsoft/codebert-base")

    if not args.skip_classification:
        print("\n" + "=" * 70)
        print("PART 1 — Pair-level classification metrics (held-out split)")
        print("=" * 70)
        evaluate_classification(df, positive_pairs, model, sbert, args)

    if not args.skip_ranking:
        print("\n" + "=" * 70)
        print("PART 2 — Ranking metrics (test Bs as queries, full time window as candidates)")
        print("=" * 70)
        # Split: test_bs = newest 30 % of linked patches by created_time.
        # Queries: test_bs patches that have at least one known link.
        # Candidates: every patch in the full df that falls in the query's
        #   time window — engine-realistic, no restriction to test patches.
        # This means the model must rank an unseen B highly among ~hundreds
        # of real unrelated candidates, not just among other test patches.
        _train_bs, test_bs = tm.split_linked_patches_temporal(
            df, positive_pairs, test_size=args.test_size
        )
        # Positive pairs where the target (B) is a test B.
        test_positive_pairs = [
            p for p in positive_pairs if p[1] in test_bs
        ]
        if not test_positive_pairs:
            print("⚠️  No positive pairs found where the target is in the test-B set. "
                  "Try a larger --test-size or run with --skip-ranking.")
        else:
            sample = args.ranking_sample if args.ranking_sample > 0 else None
            # Pass the FULL df so the time window can pull any candidate,
            # but only test_positive_pairs to restrict which patches become queries.
            evaluate_ranking(df, test_positive_pairs, model, sbert, args.window_days,
                              sample_queries=sample, seed=args.random_state)

    if not args.skip_llm_reranking:
        print("\n" + "=" * 70)
        print("PART 3 — LLM reranking metrics (ML top-k → gpt rerank → MRR/Recall@k)")
        print("=" * 70)
        _train_bs, test_bs = tm.split_linked_patches_temporal(
            df, positive_pairs, test_size=args.test_size
        )
        test_positive_pairs_llm = [
            p for p in positive_pairs if p[1] in test_bs
        ]
        if not test_positive_pairs_llm:
            print("⚠️  No positive pairs found where the target is in the test-B set. "
                  "Try a larger --test-size or run with --skip-llm-reranking.")
        else:
            sample = args.ranking_sample if args.ranking_sample > 0 else None
            evaluate_llm_reranking(
                df, test_positive_pairs_llm, model, sbert, args.window_days,
                sample_queries=sample, seed=args.random_state,
                llm_top_k=args.llm_rerank_topk,
                llm_model=args.llm_model,
                rate_limit_delay=args.llm_rate_limit_delay,
            )


if __name__ == "__main__":
    main()