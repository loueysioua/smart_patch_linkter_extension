"""
Regression tests for the hydration layer (Stage 1.5) and the bug it fixes.

The bug: ``llm_rerank_stage.py`` originally built its LLM payload from bare
patch ids, so the prompt contained no title, description or files at all. An LLM
cannot rank a list of numbers, but the Stage 2 fail-safe accepts any valid
permutation -- so it produced plausible-looking metrics from an arbitrary
ordering, silently.

The previous parity test missed this because its fake client only parsed ids out
of the prompt and never inspected the content. Both problems are addressed here.

Run:  python scripts/check_hydration.py
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
sys.path.insert(0, HERE)

KS = (1, 2, 4, 6, 8, 10)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def write_csv(path, patches):
    import csv as _csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["patch_id", "title", "description", "files"])
        for pid, title, desc, files in patches:
            w.writerow([pid, title, desc, str(files)])


def write_hand_off(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def sample_patches(n=12):
    return [
        (f"p{i:02d}",
         f"change {i} in subsystem {i % 4} fixing BUG-{1000+i}",
         f"description body for patch {i}. " * (3 + i),
         [f"core/mod{i}.py", f"core/sub{i}/util.py", f"docs/p{i}.md"])
        for i in range(n)
    ]


def sample_rows(n=8, k=6):
    rows = []
    for i in range(n):
        cands = [f"p{j:02d}" for j in range(k)]
        rows.append({
            "source_patch_id": f"p{(i * 2) % 12:02d}",
            "candidates": cands,
            "reachable_targets": [cands[(i + 1) % k]],
            "n_targets_total": 2,
            "window_days": 2,
            "top_k": k,
            "llm_applied": False,
        })
    return rows


def fake_client(capture=None):
    """Reverses the ids it is given, and records the prompt it was sent."""
    state = {"calls": 0}

    def create(**kwargs):
        state["calls"] += 1
        prompt = kwargs["messages"][0]["content"]
        if capture is not None:
            capture.append(prompt)
        block = prompt.split("### Candidate Patches to Rerank:")[1]
        ids = [ln.split("**ID:** ")[1].strip()
               for ln in block.splitlines() if "**ID:** " in ln]
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=json.dumps(ids[::-1])))])

    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create))
    return client, state


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_prompt_has_text():
    """THE regression test: the prompt must carry real content, not just ids."""
    import hydrate_candidates as hyd
    import llm_rerank_stage as st

    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "p.csv")
        hand = os.path.join(d, "s1.jsonl")
        side = os.path.join(d, "texts.jsonl")
        write_csv(csv_path, sample_patches())
        write_hand_off(hand, sample_rows())

        hyd.hydrate(hand, side, csv_path)
        texts = st.load_texts(side)
        assert texts, "sidecar is empty"

        prompts = []
        client, _ = fake_client(prompts)
        st.rerank_file(hand, side, client=client, llm_top_k=6, ks=KS,
                       progress_every=10**9)
        assert prompts, "no prompt was captured"

        first = prompts[0]
        rows = sample_rows()
        row = rows[0]
        src_title = dict((p[0], p[1]) for p in sample_patches())[row["source_patch_id"]]

        assert src_title in first, f"target title absent from prompt:\n{first[:400]}"
        assert "n/a" not in first, "prompt has empty fields rendered as n/a"
        for pid in row["candidates"]:
            title = dict((p[0], p[1]) for p in sample_patches())[pid]
            assert title in first, f"candidate {pid} title missing from prompt"
        print("  prompt contains the target title and every candidate title, no 'n/a'")


def check_fake_client_can_detect_empty_prompt():
    """The test gap that let the bug through: assert non-degenerate prompts."""
    import llm_rerank_stage as st

    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "p.csv")
        hand = os.path.join(d, "s1.jsonl")
        side = os.path.join(d, "texts.jsonl")
        write_csv(csv_path, sample_patches())
        write_hand_off(hand, sample_rows())

        import hydrate_candidates as hyd
        hyd.hydrate(hand, side, csv_path)

        # guard: a fake that REFUSES empty prompts must accept the hydrated one
        seen = []

        def strict_create(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            seen.append(prompt)
            if prompt.count("n/a") > 2:
                raise AssertionError(
                    "fake client received an id-only prompt (the original bug)")
            block = prompt.split("### Candidate Patches to Rerank:")[1]
            ids = [ln.split("**ID:** ")[1].strip()
                   for ln in block.splitlines() if "**ID:** " in ln]
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=json.dumps(ids[::-1])))])

        client = types.SimpleNamespace()
        client.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=strict_create))
        st.rerank_file(hand, side, client=client, llm_top_k=6, ks=KS,
                       progress_every=10**9)
        assert seen
        print(f"  strict fake client accepted all {len(seen)} prompts "
              f"(each had content, not ids)")


def check_unresolved_sidecar_is_rejected_loudly():
    """A wrong/empty sidecar must fail fast, not silently rerank numbers.

    Stage 1's ids-only hand-off is the normal input, so it cannot itself be the
    error. The failure mode is a sidecar that resolves nothing -- then every
    prompt is id-only, which is the original bug.
    """
    import llm_rerank_stage as st

    with tempfile.TemporaryDirectory() as d:
        hand = os.path.join(d, "s1.jsonl")
        wrong = os.path.join(d, "wrong.jsonl")
        write_hand_off(hand, sample_rows())
        with open(wrong, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"patch_id": "zzz", "title": "unrelated",
                                 "description": "x", "files": []}) + "\n")
        client, state = fake_client()
        try:
            st.rerank_file(hand, wrong, client=client, llm_top_k=6, ks=KS,
                           progress_every=10**9)
        except SystemExit as exc:
            assert "hydrate_candidates" in str(exc), str(exc)
            assert state["calls"] == 0, "must fail before spending an API call"
            print("  unresolvable sidecar rejected before any API call, names the fix")
            return
        raise AssertionError("an id-only prompt was allowed through")


def check_values_are_untrimmed():
    """Storage keeps raw text; clipping happens only at send time."""
    import hydrate_candidates as hyd
    from core.llm_rerank import MAX_DESCRIPTION_CHARS, build_rerank_prompt

    big_desc = "D" * (MAX_DESCRIPTION_CHARS * 3)
    patches = [("p00", "short title", big_desc, ["a.py"]),
               ("p01", "T" * 900, "small body", ["b.py"])]
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "p.csv")
        hand = os.path.join(d, "s1.jsonl")
        side = os.path.join(d, "texts.jsonl")
        write_csv(csv_path, patches)
        write_hand_off(hand, [{
            "source_patch_id": "p00", "candidates": ["p01"],
            "reachable_targets": ["p01"], "window_days": 2, "top_k": 2,
        }])
        res = hyd.hydrate(hand, side, csv_path)
        assert res["oversized"]["description"] == 1, res["oversized"]
        assert res["oversized"]["title"] == 1, res["oversized"]

        with open(side, encoding="utf-8") as fh:
            texts = {json.loads(l)["patch_id"]: json.loads(l) for l in fh}
        assert texts["p00"]["description"] == big_desc, "description was trimmed in storage"
        assert texts["p01"]["title"] == "T" * 900, "title was trimmed in storage"

        # ... and the prompt still clips it
        prompt = build_rerank_prompt(
            {"patch_id": "p00", **texts["p00"]}, [dict(texts["p01"])])
        assert big_desc not in prompt, "prompt failed to clip an oversized description"
        assert len(prompt) < len(big_desc), "prompt should be smaller than raw text"
    print("  oversized values stored raw, clipped only in the prompt, and reported")


def check_missing_ids_reported():
    """A failed join must be visible, not silently empty."""
    import hydrate_candidates as hyd

    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "p.csv")
        hand = os.path.join(d, "s1.jsonl")
        side = os.path.join(d, "texts.jsonl")
        write_csv(csv_path, sample_patches(6))          # rows p00..p05
        write_hand_off(hand, [{
            "source_patch_id": "p00",
            "candidates": ["p01", "p99"],                # p99 absent from the CSV
            "reachable_targets": ["p01"], "window_days": 2, "top_k": 2,
        }])
        res = hyd.hydrate(hand, side, csv_path)
        assert res["ids_missing"] == 1, res
        assert res["ids_resolved"] == 2, res
    print("  1 missing patch_id reported, not crashed")


def check_chunked_equals_single_read():
    """The memory optimisation must be lossless."""
    import hydrate_candidates as hyd

    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "p.csv")
        hand = os.path.join(d, "s1.jsonl")
        write_csv(csv_path, sample_patches(12))
        write_hand_off(hand, sample_rows())
        needed, _ = hyd.collect_needed_ids(hand)
        a = hyd.load_texts(csv_path, needed, chunk_size=3)
        b = hyd.load_texts(csv_path, needed, chunk_size=10_000)
        assert a == b, "chunked read differs from single read"
        assert len(a) == len(needed)
    print(f"  chunked (3 rows) == single read, {len(a)} patches identical")


def check_stage_b_light_and_stage_15_too():
    """Both new stages must stay free of the heavy imports."""
    for mod in ("hydrate_candidates", "llm_rerank_stage"):
        script = f'''
import sys
BLOCKED = ("sentence_transformers", "faiss", "joblib", "lightgbm", "torch")
class B:
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, B())
sys.path.insert(0, {BACKEND!r}); sys.path.insert(0, {HERE!r})
import {mod}
print("OK")
'''
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, f"{mod}: {out.stderr[-500:]}"
        assert "OK" in out.stdout
    print("  both stages import with transformers/faiss/joblib/lightgbm blocked")


def main():
    print("=" * 68)
    print("HYDRATION: does the LLM actually receive the patch text?")
    print("=" * 68)
    check_prompt_has_text()
    check_fake_client_can_detect_empty_prompt()
    check_unresolved_sidecar_is_rejected_loudly()
    check_values_are_untrimmed()
    check_missing_ids_reported()
    check_chunked_equals_single_read()
    check_stage_b_light_and_stage_15_too()
    print("\nPASS")


if __name__ == "__main__":
    main()
