"""
Self-check for the optional LLM rerank stage (Stage 3).

The stage is a *permutation* of the ML+RAG top-k, so the properties that
matter are: the output always contains exactly the input candidates, a broken
or unreachable LLM never loses the ranking, and the prompt stays inside a sane
size budget. No network access required.

Run:  python scripts/check_llm_rerank.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.llm_rerank import (  # noqa: E402
    apply_rerank_order,
    build_rerank_prompt,
    parse_rerank_response,
    rerank_candidates,
)

CANDIDATES = [
    {"patch_id": "101", "title": "fix null deref in parser", "description": "x" * 40, "files": ["a.py"]},
    {"patch_id": "102", "title": "refactor parser", "description": "y" * 40, "files": ["b.py"]},
    {"patch_id": "103", "title": "add parser tests", "description": "z" * 40, "files": ["c.py"]},
]


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeCompletions:
    def __init__(self, reply=None, raises=None):
        self.reply, self.raises, self.calls = reply, raises, 0

    def create(self, **kwargs):
        self.calls += 1
        if self.raises:
            raise self.raises
        return _Resp(self.reply)


class FakeClient:
    def __init__(self, reply=None, raises=None):
        self.chat = type("C", (), {"completions": FakeCompletions(reply, raises)})()


def ids(cands):
    return [str(c["patch_id"]) for c in cands]


def test_output_is_always_a_permutation():
    """Whatever the LLM says, the candidate SET must be unchanged."""
    cases = {
        "clean order":        '["102", "101", "103"]',
        "reversed":           '["103", "102", "101"]',
        "markdown fenced":    '```json\n["102", "101", "103"]\n```',
        "bare fence":         '```\n["103", "101", "102"]\n```',
        "partial list":       '["103"]',
        "unknown id mixed":   '["999", "102", "101"]',
        "duplicate ids":      '["102", "102", "101"]',
        "whitespace":         '  ["102","101","103"]  ',
    }
    for label, reply in cases.items():
        out, err = parse_rerank_response(reply, CANDIDATES)
        assert err is None, f"{label}: unexpected parse error {err}"
        assert sorted(ids(out)) == sorted(ids(CANDIDATES)), f"{label}: set changed -> {ids(out)}"
        assert len(out) == len(CANDIDATES), f"{label}: wrong length {len(out)}"
    print(f"  permutation holds for {len(cases)} reply shapes")

    # order actually respected when the model is well-behaved
    out, _ = parse_rerank_response('["103", "101", "102"]', CANDIDATES)
    assert ids(out) == ["103", "101", "102"], ids(out)

    # forgotten candidates appended in original relative order
    out, _ = parse_rerank_response('["103"]', CANDIDATES)
    assert ids(out) == ["103", "101", "102"], ids(out)


def test_recall_at_k_is_invariant():
    """Recall@top_k depends only on the SET, so it cannot change."""
    out, _ = parse_rerank_response('["103", "102", "101"]', CANDIDATES)
    top2_before, top2_after = set(ids(CANDIDATES)[:2]), set(ids(out)[:2])
    # ordering may differ; membership of the full set may not
    assert set(ids(out)) == set(ids(CANDIDATES))
    assert top2_before <= set(ids(out)) and top2_after <= set(ids(CANDIDATES))
    print("  full-set membership preserved (Recall@top_k invariant by construction)")


def test_bad_replies_are_rejected_not_guessed():
    for label, reply in {
        "not json":       "I think 103 is best",
        "json object":    '{"ranking": ["103"]}',
        "numbers":        "[103, 101, 102]",
        "empty string":   "",
    }.items():
        out, err = parse_rerank_response(reply, CANDIDATES)
        assert out is None, f"{label}: should have been rejected"
        assert err, f"{label}: no reason given"
    print("  malformed replies rejected with a reason")


def test_api_failure_keeps_original_ranking():
    """Transport error, unparseable reply, and empty input all degrade safely."""
    boom = FakeClient(raises=ConnectionError("no route to host"))
    assert ids(rerank_candidates(boom, {"patch_id": "1"}, CANDIDATES)) == ids(CANDIDATES)
    assert boom.chat.completions.calls == 1, "expected exactly one attempt"

    garbage = FakeClient(reply="sorry, I cannot help")
    assert ids(rerank_candidates(garbage, {"patch_id": "1"}, CANDIDATES)) == ids(CANDIDATES)

    assert rerank_candidates(FakeClient(), {"patch_id": "1"}, []) == []

    # and the happy path still reorders
    good = FakeClient(reply='["103", "101", "102"]')
    assert ids(rerank_candidates(good, {"patch_id": "1"}, CANDIDATES)) == ["103", "101", "102"]
    print("  transport error / bad reply / empty input all fall back to input order")


def test_prompt_is_size_bounded():
    """Huge change_log blobs must not reach the provider unbounded."""
    fat = [{
        "patch_id": str(i),
        "title": "T" * 5000,
        "description": "D" * 200_000,
        "files": [f"f{n}.py" for n in range(500)],
    } for i in range(10)]

    prompt = build_rerank_prompt(
        {"patch_id": "q", "title": "T" * 5000, "description": "D" * 200_000, "files": fat[0]["files"]},
        fat,
        max_description_chars=1500,
    )
    assert "200,000" not in prompt
    assert len(prompt) < 40_000, f"prompt too large: {len(prompt)} chars"
    # 10 candidates * (1500 desc + 300 title) + overhead, plus the target
    assert len(prompt) > 10_000, "suspiciously small - truncation may be too aggressive"
    assert "..." in prompt or "[...]" in prompt
    print(f"  10 fat candidates -> prompt {len(prompt):,} chars (was ~2,000,000 untruncated)")


def test_timeout_is_forwarded():
    seen = {}

    class Recorder(FakeCompletions):
        def create(self, **kwargs):
            seen.update(kwargs)
            return super().create(**kwargs)

    client = FakeClient(reply='["101", "102", "103"]')
    client.chat.completions = Recorder('["101", "102", "103"]')
    rerank_candidates(client, {"patch_id": "1"}, CANDIDATES, timeout=12.5)
    assert seen.get("timeout") == 12.5, seen
    assert seen.get("temperature") == 0.0, "reranking must be deterministic"
    print("  timeout=12.5 and temperature=0.0 reach the provider")


def main():
    test_output_is_always_a_permutation()
    test_recall_at_k_is_invariant()
    test_bad_replies_are_rejected_not_guessed()
    test_api_failure_keeps_original_ranking()
    test_prompt_is_size_bounded()
    test_timeout_is_forwarded()
    print("PASS")


if __name__ == "__main__":
    main()
