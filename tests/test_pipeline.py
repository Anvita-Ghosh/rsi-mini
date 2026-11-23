"""Pipeline tests with a scripted (mock) LLM: no model, no network needed beyond the cached dataset."""
import json
import re

import pytest

from src.agent import Agent, TaskView, default_genome, extract_code, extract_asserts, retrieve_tips, genome_hash
from src.data import load_mbpp, make_splits
from src.evaluate import wilson, run_genome, solve_and_grade
from src.llm import LLM, MockBackend, CostTracker, BudgetExceeded
from src.meta import apply_edits, run_meta, parse_json

CFG = {"seed": 0, "workers": 2, "max_new_tokens": 300, "sandbox_timeout": 5, "grade_timeout": 10,
       "meta_iterations": 3, "accept_margin_problems": 3, "meta_temperature": 0.7, "meta_max_failures": 4}


@pytest.fixture(scope="module")
def problems():
    return load_mbpp()


def fenced(code):
    return f"Reasoning: ok.\n```python\n{code}\n```"


# ----------------------------------------------------------------- data
def test_splits_are_disjoint_deterministic_and_sized(problems):
    a, b = make_splits(problems, 0), make_splits(problems, 0)
    assert [p.task_id for p in a["dev"]] == [p.task_id for p in b["dev"]]
    assert (len(a["train"]), len(a["dev"]), len(a["test"])) == (100, 100, 227)
    ids = [set(p.task_id for p in a[k]) for k in ("train", "dev", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    c = make_splits(problems, 1)
    assert [p.task_id for p in a["dev"]] != [p.task_id for p in c["dev"]]


def test_visible_test_is_first_assert(problems):
    p = problems[0]
    assert p.visible_test == p.tests[0]


# ----------------------------------------------------------- helpers
def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(100, 100)
    assert abs(hi - 1.0) < 1e-9 and lo > 0.95


def test_extract_code_and_asserts():
    assert extract_code("hi\n```python\ndef f():\n    pass\n```\n") == "def f():\n    pass"
    assert extract_code("no code here") is None
    assert extract_asserts("assert f(1) == 2\nassert f(2) ==\nassert f(3) == 4", "assert f(3) == 4") == ["assert f(1) == 2"]


def test_tip_retrieval_by_keyword_and_category():
    skills = [{"id": "s1", "category": "strings", "keywords": ["substring"], "tip": "a"},
              {"id": "s2", "category": "math", "keywords": ["prime"], "tip": "b"},
              {"id": "s3", "category": "general", "keywords": [], "tip": "c"}]
    got = [s["id"] for s in retrieve_tips(skills, "find the substring in strings", 2)]
    assert got == ["s1", "s3"]
    assert [s["id"] for s in retrieve_tips(skills, "check prime numbers", 1)] == ["s2"]
    assert retrieve_tips(skills, "anything", 0) == []


# ------------------------------------------------------------ budget
def test_cost_cap_stops_calls():
    t = CostTracker(price_in_per_mtok=1000, price_out_per_mtok=1000, cap_usd=0.001)
    llm = LLM(MockBackend(lambda m, j: "x " * 50), t)
    with pytest.raises(BudgetExceeded):
        for _ in range(50):
            llm.chat([{"role": "user", "content": f"hello {_}"}])


def test_cache_replays_without_cost(tmp_path):
    calls = []
    be = MockBackend(lambda m, j: calls.append(1) or "answer")
    llm = LLM(be, CostTracker(), tmp_path / "c.jsonl")
    llm.chat([{"role": "user", "content": "q"}]); llm.chat([{"role": "user", "content": "q"}])
    assert len(calls) == 1 and llm.tracker.cached_calls == 1 and llm.tracker.fresh_calls == 1
    llm2 = LLM(be, CostTracker(), tmp_path / "c.jsonl")       # new process reads the file
    assert llm2.chat([{"role": "user", "content": "q"}]).cached and len(calls) == 1


# ------------------------------------------------------------- agent
ADD_TASK = TaskView(1, "Write a function add(a, b) returning a+b.", "assert add(1, 2) == 3")


def test_agent_revises_after_visible_failure():
    replies = iter([fenced("def add(a, b):\n    return a - b"), fenced("def add(a, b):\n    return a + b")])
    llm = LLM(MockBackend(lambda m, j: next(replies)))
    tr = Agent(llm, default_genome()).solve(ADD_TASK)
    acts = [s["action"] for s in tr["steps"]]
    assert acts == ["WriteCode", "RunVisibleTest", "Revise", "RunVisibleTest", "Finish"]
    assert tr["visible_pass"] and tr["rounds"] == 2
    assert "got -1, expected 3" in tr["steps"][1]["observation"]   # failure feedback reaches the agent


def test_agent_respects_max_rounds_and_keeps_best_visible_code():
    llm = LLM(MockBackend(lambda m, j: fenced("def add(a, b):\n    return 0")))
    g = default_genome(); g["max_rounds"] = 2
    tr = Agent(llm, g).solve(ADD_TASK)
    assert tr["rounds"] == 2 and not tr["visible_pass"]


def test_agent_self_tests_flag_adds_write_tests_action():
    def fn(messages, js):
        if "Write 3 NEW" in messages[-1]["content"]:
            return "```python\nassert add(2, 2) == 4\nassert add(0, 0) == 0\n```"
        return fenced("def add(a, b):\n    return a + b")
    g = default_genome(); g["self_tests"] = True
    tr = Agent(LLM(MockBackend(fn)), g).solve(ADD_TASK)
    assert "WriteTests" in [s["action"] for s in tr["steps"]] and tr["n_extra_tests"] == 2 and tr["visible_pass"]


def test_agent_never_sees_hidden_tests_and_overfitting_is_measured(problems):
    p = next(p for p in problems if len(p.tests) >= 3)
    seen = []

    def fn(messages, js):
        seen.append("\n".join(m["content"] for m in messages))
        # overfit: hard-code the answer to the visible assert only
        m = re.match(r"assert .*?\((.*)\) == (.*)$", p.visible_test, re.S)
        fname = re.search(r"assert (?:\w+\()?(\w+)\(", p.visible_test).group(1)
        return fenced(f"def {fname}(*a, **k):\n    return {m.group(2)}")

    llm = LLM(MockBackend(fn))
    tr = solve_and_grade(llm, default_genome(), p, CFG)
    for t in p.tests[1:]:
        assert t not in "".join(seen)                      # hidden asserts never reach the prompt
    assert p.reference not in "".join(seen)
    assert tr["visible_pass"] is True and tr["hidden_pass"] is False


# --------------------------------------------------------------- edits
def test_apply_edits_validates():
    g = default_genome()
    new, ok, bad = apply_edits(g, [
        {"op": "add_skill", "category": "Strings", "keywords": ["String"], "tip": "Use str methods."},
        {"op": "add_skill", "category": "x", "keywords": [], "tip": "t" * 500},         # too long
        {"op": "set_param", "name": "max_rounds", "value": 99},                          # out of range
        {"op": "set_param", "name": "self_tests", "value": True},                        # 4th edit: over the cap of 3
    ])
    assert len(ok) == 1 and len(bad) == 2
    assert new["skills"][0]["category"] == "strings" and new["skills"][0]["keywords"] == ["string"]
    assert new["self_tests"] is False and g["skills"] == []                              # cap enforced; original untouched
    new2, ok2, bad2 = apply_edits(g, [{"op": "set_param", "name": "self_tests", "value": True},
                                      {"op": "nuke_everything"}])
    assert new2["self_tests"] is True and len(ok2) == 1 and len(bad2) == 1
    assert genome_hash(new) != genome_hash(g)
    assert apply_edits(g, "garbage")[1] == []


def test_parse_json_tolerates_chatter():
    assert parse_json('Sure! {"edits": []} done') == {"edits": []}
    assert parse_json("no json") is None


# --------------------------------------- meta loop: accept / reject logic
def test_meta_accepts_real_gain_and_rejects_noise(problems, tmp_path):
    """Mock agent solves only odd task_ids, unless the tip 'MAGIC' is in the prompt (then everything).
    Meta call 1 adds the MAGIC tip (real gain), call 2 adds a useless tip, call 3 emits garbage."""
    by_text = {p.text: p for p in problems}
    meta_calls = []

    def fn(messages, json_mode):
        if json_mode:
            meta_calls.append(1)
            n = len(meta_calls)
            if n == 1:
                return json.dumps({"analysis": "needs magic", "edits": [
                    {"op": "add_skill", "category": "general", "keywords": [], "tip": "MAGIC"}]})
            if n == 2:
                return json.dumps({"analysis": "meh", "edits": [
                    {"op": "add_skill", "category": "general", "keywords": [], "tip": "be nice"}]})
            return "this is not json"
        user = messages[1]["content"]
        text = re.search(r"Task: (.*?)\n\nThe function must", user, re.S).group(1)
        p = by_text[text]
        if "MAGIC" in user or p.task_id % 2 == 1:
            return fenced(p.reference)
        return fenced("def nothing():\n    return None")

    splits = make_splits(problems, 0)
    # keep problems where the reference passes our grader; use 20 train / 20 dev
    sub = {"train": splits["train"][:20], "dev": splits["dev"][:20], "test": []}
    llm = LLM(MockBackend(fn))
    cfg = dict(CFG, meta_iterations=3)
    best = run_meta(llm, sub, cfg, tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "meta_log.jsonl").read_text().splitlines()]
    decisions = [r["decision"] for r in rows]
    assert decisions[0] == "baseline"
    assert decisions[1] == "accept" and rows[1]["delta_dev_pass"] >= 3
    assert decisions[2] == "reject"                      # 'be nice' adds nothing on top of MAGIC (+0 < margin)
    assert decisions[3].startswith("reject") and rows[3]["parse_ok"] is False
    assert any(s["tip"] == "MAGIC" for s in best["skills"]) and best["name"] == "iter1"
    assert (tmp_path / "best_genome.json").exists() and (tmp_path / "genomes" / "iter1.json").exists()


def test_meta_control_shuffles_failure_evidence(problems):
    from src.meta import failure_cases
    import random
    traces = [{"hidden_pass": False, "problem": f"P{i}", "visible_test": "assert f()", "visible_pass": bool(i % 2),
               "final_code": f"code{i}", "hidden_error": f"err{i}"} for i in range(5)]
    real = failure_cases(traces, 5, random.Random(0), None)
    ctrl = failure_cases(traces, 5, random.Random(0), "shuffled")
    assert all(c["final_code"][-1] == c["problem"][-1] for c in real)
    assert not any(c["final_code"][-1] == c["problem"][-1] for c in ctrl)
