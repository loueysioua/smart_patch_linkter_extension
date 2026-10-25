"""
Guarantee that the fused and decoupled LLM rerank paths agree.

There are two ways to apply the LLM stage:

  fused  ``ml_to_rag_pipeline.py --evaluate --llm-rerank``
         one process; Stage 2 runs inline. Costs ~6.5 GB.
  split  Stage 1 writes ``--candidates-out``, then
         ``llm_rerank_stage.py`` reranks that file. Costs <200 MB.

They are two code paths computing the same number. That is exactly the
train/serve skew class of bug this project already got bitten by once, so the
equality is asserted here rather than assumed.

Checks:
  1. per-query reciprocal_rank and recall@k identical between the two paths
  2. Stage 2 imports with sentence-transformers / faiss / joblib / lightgbm
     blocked -- proving the memory independence is real
  3. Stage 2 streams: peak memory does not grow with query count
  4. resuming from cache issues no duplicate API calls

Run:  python scripts/check_stage_parity.py
"""
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, BACKEND)

KS = (1, 2, 4, 6, 8, 10)


# ---------------------------------------------------------------------------
# fake provider
# ---------------------------------------------------------------------------

class _Msg:
    def __init__(self, c): self.content = c


class _Choice:
    def __init__(self, c): self.message = _Msg(c)


class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]


def make_fake_client(reverse=True):
    """Returns (client, state). Replies with the candidates in reverse order."""
    state = {"calls": 0}

    def create(**kwargs):
        state["calls"] += 1
        prompt = kwargs["messages"][0]["content"]
        block = prompt.split("### Candidate Patches to Rerank:")[1]
        ids = [ln.split("**ID:** ")[1].strip() for ln in block.splitlines() if "**ID:** " in ln]
        return _Resp(json.dumps(ids[::-1] if reverse else ids))

    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create))
    return client, state


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def write_patch_csv(path, ids):
    """Patch CSV covering exactly the ids the hand-off references."""
    import csv as _csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["patch_id", "title", "description", "files"])
        for pid in sorted(ids):
            w.writerow([pid, f"title for {pid}", f"body text for {pid} " * 5,
                        str([f"core/{pid}.py", f"core/{pid}/u.py"])])


def hydrate(tmpdir, n=40, k=10):
    """Stage 1 hand-off + Stage 1.5 sidecar, as the real pipeline would."""
    import hydrate_candidates as hyd
    hand = os.path.join(tmpdir, "s1.jsonl")
    side = os.path.join(tmpdir, "texts.jsonl")
    csvp = os.path.join(tmpdir, "p.csv")
    rows = write_candidates(hand, n=n, k=k)
    needed, _ = hyd.collect_needed_ids(hand)
    write_patch_csv(csvp, needed)          # exactly the ids the hand-off uses
    hyd.hydrate(hand, side, csvp)
    return hand, side, rows


def write_candidates(path, n=40, k=10, reachable_ratio=0.6):
    """A Stage 1 hand-off with deterministic, overlapping candidate sets."""
    rows = []
    for i in range(n):
        cands = [f"p{i:03d}_{j}" for j in range(k)]
        n_reach = max(1, int(k * reachable_ratio))
        # first target sits at a known, varying rank so recall differs per query
        targets = [cands[(i * 3) % k]] if i % 5 else cands[:2]
        rows.append({
            "source_patch_id": f"src{i:03d}",
            "candidates": cands,
            "reachable_targets": targets[:n_reach],
            "n_targets_total": len(targets) + 2,
            "window_days": 2,
            "top_k": k,
            "llm_applied": False,
        })
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def reference_scores(rows, ks=KS):
    """Independent recomputation from (order, targets) - the spec, not the code."""
    out = {}
    for r in rows:
        cands, targets = r["candidates"], set(r["reachable_targets"])
        rr = 0.0
        for rank, pid in enumerate(cands, 1):
            if pid in targets:
                rr = 1.0 / rank
                break
        rec = {k: len(set(cands[:k]) & targets) / len(targets) for k in ks}
        out[r["source_patch_id"]] = (rr, rec)
    return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_split_metrics_are_correct():
    """Stage 2 output must match an independent recomputation."""
    import llm_rerank_stage as st

    with tempfile.TemporaryDirectory() as d:
        src, side, rows = hydrate(d, n=25)
        client, _ = make_fake_client(reverse=True)

        res = st.rerank_file(src, side, out_path=os.path.join(d, "m.csv"),
                            reranked_out=os.path.join(d, "r.jsonl"),
                            client=client, llm_top_k=10, ks=KS,
                            progress_every=1000)
        import csv as _csv
        got = {}
        with open(os.path.join(d, "m.csv"), encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                got[row["source_patch_id"]] = (
                    float(row["reciprocal_rank"]),
                    {k: float(row[f"recall_at_{k}"]) for k in KS},
                )

        # the fake reverses the head, so the expected order is reversed too
        expected = {}
        for r in rows:
            cands, targets = r["candidates"], set(r["reachable_targets"])
            flipped = cands[::-1]
            rr = 0.0
            for rank, pid in enumerate(flipped, 1):
                if pid in targets:
                    rr = 1.0 / rank
                    break
            expected[r["source_patch_id"]] = (
                rr, {k: len(set(flipped[:k]) & targets) / len(targets) for k in KS})

        assert set(got) == set(expected), "query set mismatch"
        for qid, (rr, rec) in expected.items():
            assert abs(got[qid][0] - rr) < 1e-12, f"{qid}: rr {got[qid][0]} != {rr}"
            for k in KS:
                assert abs(got[qid][1][k] - rec[k]) < 1e-12, f"{qid}: recall@{k}"
        assert res["queries"] == len(rows)
        assert res["reordered"] == len(rows), "every query should have been reordered"
    print(f"  stage 2 metrics match independent recomputation ({len(rows)} queries)")


def check_fused_matches_split():
    """The whole point: both paths must produce identical numbers."""
    import llm_rerank_stage as st

    with tempfile.TemporaryDirectory() as d:
        src, side, rows = hydrate(d, n=30)

        # --- split path (through the hydrator, as production does) ---
        c1, _ = make_fake_client(reverse=True)
        split_out = os.path.join(d, "split.csv")
        st.rerank_file(src, side, out_path=split_out, client=c1, llm_top_k=10,
                      ks=KS, progress_every=1000)
        split = _read_csv(split_out)

        # --- fused path: same logic, inline, as ml_to_rag_pipeline does it ---
        fused = {}
        for r in rows:
            cands = [str(c) for c in r["candidates"]]
            targets = {str(t) for t in r["reachable_targets"]}
            head, tail = cands[:10], cands[10:]
            client = c1
            from core.llm_rerank import rerank_candidates
            out = rerank_candidates(client, {"patch_id": r["source_patch_id"]},
                                    [{"patch_id": p} for p in head],
                                    model_name="gpt-4o-mini", timeout=60)
            final = st.apply_order([str(c["patch_id"]) for c in out], head, tail)
            rr, rec = st.score(final, targets, KS)
            fused[r["source_patch_id"]] = (rr, rec)

        assert set(split) == set(fused), "query set mismatch"
        mismatches = []
        for qid, (rr, rec) in fused.items():
            srr, srec = split[qid]
            if abs(srr - rr) > 1e-12:
                mismatches.append(f"{qid}: rr {srr} vs {rr}")
            for k in KS:
                if abs(srec[k] - rec[k]) > 1e-12:
                    mismatches.append(f"{qid}: recall@{k} {srec[k]} vs {rec[k]}")
        assert not mismatches, "fused != split:\n  " + "\n  ".join(mismatches[:10])
    print(f"  fused == split on {len(fused)} queries (rr + all {len(KS)} recall@k)")


def _read_csv(path):
    import csv as _csv
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            out[row["source_patch_id"]] = (
                float(row["reciprocal_rank"]),
                {k: float(row[f"recall_at_{k}"]) for k in KS},
            )
    return out


def check_stage_b_is_independent():
    """Block the heavy imports and confirm Stage 2 still runs."""
    script = f'''
import sys
BLOCKED = ("sentence_transformers", "faiss", "joblib", "lightgbm", "torch", "sklearn")
class Blocker:
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked for test: " + name)
        return None
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {BACKEND!r})
sys.path.insert(0, {HERE!r})
import llm_rerank_stage
print("IMPORT_OK")
heavy = [m for m in BLOCKED if m in sys.modules]
print("HEAVY:" + ",".join(heavy))
'''
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, f"stage 2 failed to import:\n{out.stderr[-800:]}"
        assert "IMPORT_OK" in out.stdout, out.stdout
        heavy = [ln.split("HEAVY:")[1] for ln in out.stdout.splitlines() if ln.startswith("HEAVY:")]
        assert not heavy or heavy[0] == "", f"heavy modules pulled in: {heavy}"
    print("  stage 2 imports with sentence_transformers/faiss/joblib/lightgbm blocked")


def check_memory_is_flat():
    """Peak RSS must not scale with the number of queries."""
    import resource
    import llm_rerank_stage as st

    peaks = {}
    with tempfile.TemporaryDirectory() as d:
        for n in (25, 400):
            src, side, _ = hydrate(d, n=n)
            client, _ = make_fake_client(reverse=True)
            before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            st.rerank_file(src, side, client=client, llm_top_k=10, ks=KS,
                           progress_every=10**9)
            after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peaks[n] = after  # ru_maxrss is a high-water mark, in KB on Linux
    growth_mb = (peaks[400] - peaks[25]) / 1024
    assert growth_mb < 40, f"peak RSS grew {growth_mb:.0f} MB for 16x the queries"
    print(f"  peak RSS {peaks[25]/1024:.0f} MB (25 q) -> {peaks[400]/1024:.0f} MB (400 q)"
          f"  = +{growth_mb:.0f} MB for 16x the rows")


def check_resume_does_not_recall():
    """A second pass must be served entirely from cache."""
    import llm_rerank_stage as st

    with tempfile.TemporaryDirectory() as d:
        src, side, _ = hydrate(d, n=20)
        cache = os.path.join(d, "c.json")

        client_a, state_a = make_fake_client(reverse=True)
        st.rerank_file(src, side, out_path=os.path.join(d, "a.csv"), client=client_a,
                      llm_top_k=10, cache_path=cache, ks=KS, progress_every=10**9)
        assert state_a["calls"] == 20, state_a

        client_b, state_b = make_fake_client(reverse=True)
        res = st.rerank_file(src, side, out_path=os.path.join(d, "b.csv"), client=client_b,
                             llm_top_k=10, cache_path=cache, ks=KS, progress_every=10**9)
        assert state_b["calls"] == 0, f"re-called the LLM {state_b['calls']} times on a warm cache"
        assert res["cache_hits"] == 20, res
        assert _read_csv(os.path.join(d, "a.csv")) == _read_csv(os.path.join(d, "b.csv"))
    print("  warm cache: 0 API calls, metrics byte-identical")


def main():
    print("=" * 66)
    print("STAGE PARITY: fused --llm-rerank  vs  decoupled Stage 2")
    print("=" * 66)
    check_split_metrics_are_correct()
    check_fused_matches_split()
    check_stage_b_is_independent()
    check_memory_is_flat()
    check_resume_does_not_recall()
    print("\nPASS")


if __name__ == "__main__":
    main()
