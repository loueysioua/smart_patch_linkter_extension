"""
Self-check for the train/serve contract that used to fail silently.

Two bugs motivated this file:

1. The embeddings cache was validated on ``model_name`` + ``n_rows`` only.
   A permutation of the dataframe preserves both, so a cache written before a
   ``sort_values("created_time")`` removal stayed "valid" while every embedding
   was attributed to the wrong patch. No exception, wrong ranking.

2. Training and serving each had their own copy of the feature builder and
   they drifted. The model expected 17 features, serving produced 14, and
   ``reindex(columns=feature_cols, fill_value=0)`` silently zero-filled the
   four that were missing.

Run:  python scripts/check_pipeline_contract.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.utils import (  # noqa: E402
    build_pair_features,
    cache_is_valid,
    embeddings_fingerprint,
    title_desc_texts,
)

BACKEND_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.path.join(BACKEND_ROOT, "data", "onap", "model_onap.pkl")


def _toy_df(n=6, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    return pd.DataFrame({
        "patch_id": [str(100 + i) for i in range(n)],
        "title": [f"patch title {i}" for i in range(n)],
        "description": [f"body {i}" for i in range(n)],
        "created_time": [base + pd.Timedelta(days=int(d)) for d in rng.integers(0, 60, n)],
        "files_parsed": [["a/b.py", "c/d.py"] if i % 2 else ["x/y.py"] for i in range(n)],
    })


def test_cache_rejects_reordering():
    """The reported bug: a permuted dataframe must invalidate the cache."""
    df = _toy_df()
    texts = title_desc_texts(df)
    cache = {
        "n_rows": len(df),
        "model_name": "all-MiniLM-L6-v2",
        "embeddings": np.zeros((len(df), 4)),
        "fingerprint": embeddings_fingerprint(df["patch_id"], texts),
    }

    # The old check would have accepted both of these.
    assert cache["n_rows"] == len(df)
    assert cache["model_name"] == "all-MiniLM-L6-v2"

    assert cache_is_valid(cache, df, "all-MiniLM-L6-v2", texts) is True

    shuffled = df.sort_values("created_time").reset_index(drop=True)
    assert list(shuffled["patch_id"]) != list(df["patch_id"]), "toy df did not reorder"
    assert len(shuffled) == len(df), "permutation must preserve row count"
    assert cache_is_valid(cache, shuffled, "all-MiniLM-L6-v2") is False

    # Same order, edited text -> also stale.
    edited = df.copy()
    edited.loc[0, "title"] = "patch title 0 (edited)"
    assert cache_is_valid(cache, edited, "all-MiniLM-L6-v2") is False

    # Different model, and a pre-fingerprint cache, are both rejected.
    assert cache_is_valid(cache, df, "some-other-model", texts) is False
    assert cache_is_valid({"n_rows": len(df), "model_name": "all-MiniLM-L6-v2",
                           "embeddings": np.zeros((len(df), 4))},
                          df, "all-MiniLM-L6-v2", texts) is False


def test_features_match_saved_model():
    """Training and serving must compute the model's exact feature set."""
    df = _toy_df()
    df["ticket_refs"] = [set() if i % 2 else {"BUG-1"} for i in range(len(df))]

    rng = np.random.default_rng(1)
    emb = rng.normal(size=(len(df), 8))

    feats = build_pair_features(df, emb, 0, 1)

    # symmetric: LambdaRank groups must not depend on argument order
    flipped = build_pair_features(df, emb, 1, 0)
    for key in feats:
        assert np.isclose(feats[key], flipped[key]), f"asymmetric feature: {key}"

    expected = {
        "jaccard", "nb_shared", "abs_delta_files", "min_len_files", "max_len_files",
        "same_component", "LCP_mean", "LCP_max", "LCSuff_mean", "LCSuff_max",
        "sim_cosine", "token_jaccard", "delta_time_hours", "shares_ticket_ref",
    }
    # Guard the leak fix: these three compared the anchor's post-submission
    # review history against the candidate's. They must never come back.
    for leaked in ("sim_cosine_discussion", "token_jaccard_discussion",
                   "has_discussion_both"):
        assert leaked not in feats, f"leaky feature reintroduced: {leaked}"
    assert set(feats) == expected, (
        f"feature set drifted: missing={expected - set(feats)} extra={set(feats) - expected}"
    )

    if not os.path.exists(MODEL_PATH):
        print(f"  (skipped saved-model check: {MODEL_PATH} not found)")
        return

    import joblib
    feature_cols = set(joblib.load(MODEL_PATH)["feature_cols"])
    extra = feature_cols - set(feats)
    assert not extra, (
        f"saved model at {MODEL_PATH} still expects {sorted(extra)} -- it was "
        f"trained before the discussion-leak fix. Retrain to get a 14-feature model."
    )
    print(f"  saved model needs {len(feature_cols)} features, all produced")


def main():
    test_cache_rejects_reordering()
    print("ok: cache rejects reordering / text edits / model changes")
    test_features_match_saved_model()
    print("ok: feature set is symmetric and matches the saved model")
    print("PASS")


if __name__ == "__main__":
    main()
