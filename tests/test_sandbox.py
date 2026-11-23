import os
import tempfile
import time
from pathlib import Path

import pytest

from src import sandbox
from src.sandbox import run_python, run_tests

STRONG = sandbox.CONTAINMENT_LEVEL == "seatbelt+subprocess"
needs_seatbelt = pytest.mark.skipif(not STRONG, reason="Seatbelt (sandbox-exec) unavailable here")


def test_infinite_loop_is_killed():
    t0 = time.time()
    r = run_python("while True:\n    pass\n", timeout=2)
    assert r.timed_out
    assert time.time() - t0 < 6


def test_timeout_reported_by_grader():
    out = run_tests("def f():\n    while True: pass\n", ["assert f() == 1"], timeout=2)
    assert out.timed_out and not out.all_passed


@needs_seatbelt
def test_file_delete_outside_sandbox_is_contained():
    d = Path(tempfile.mkdtemp(prefix="rsi_canary_"))          # outside the sandbox's own temp dir
    canary = d / "precious.txt"
    canary.write_text("do not delete")
    r = run_python(f"import os\ntry:\n    os.remove({str(canary)!r})\n    print('DELETED')\n"
                   f"except Exception as e:\n    print('BLOCKED', type(e).__name__)\n")
    assert canary.exists(), "sandboxed code deleted a file outside its temp dir!"
    assert "BLOCKED" in r.stdout


@needs_seatbelt
def test_home_directory_is_unreadable_and_unwritable():
    secret = Path(__file__).parent / "_secret_canary.txt"       # lives under /Users/...
    secret.write_text("TOP-SECRET")
    try:
        r = run_python(f"try:\n    print(open({str(secret)!r}).read())\nexcept Exception as e:\n"
                       f"    print('BLOCKED', type(e).__name__)\n")
        assert "TOP-SECRET" not in r.stdout and "BLOCKED" in r.stdout
        w = Path(__file__).parent / "_write_canary.txt"
        run_python(f"try:\n    open({str(w)!r}, 'w').write('x')\nexcept Exception:\n    pass\n")
        assert not w.exists()
    finally:
        secret.unlink(missing_ok=True)


@needs_seatbelt
def test_network_is_blocked():
    r = run_python("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
                   "    print('CONNECTED')\nexcept Exception as e:\n    print('BLOCKED')\n", timeout=8)
    assert "BLOCKED" in r.stdout and "CONNECTED" not in r.stdout


def test_runs_in_temp_dir_and_can_write_there():
    r = run_python("import os\nopen('x.txt','w').write('hi')\nprint(os.getcwd())\n")
    assert "rsi_sbx_" in r.stdout
    assert not (Path.cwd() / "x.txt").exists()


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak")
    r = run_python("import os\nprint(os.environ.get('LLM_API_KEY'))\nprint(os.environ.get('HOME'))\n")
    assert "sk-should-not-leak" not in r.stdout
    assert "rsi_sbx_" in r.stdout


def test_grader_correct_wrong_and_overfit_code():
    tests = ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 0) == 0"]
    good = run_tests("def add(a, b):\n    return a + b\n", tests)
    assert good.all_passed and good.results == [True, True, True]
    bad = run_tests("def add(a, b):\n    return a - b\n", tests)
    assert not bad.all_passed and "got -1, expected 3" in bad.errors[0]
    # Hard-codes the visible test: passes test #1 only -> visible-pass but hidden-fail
    hardcoded = run_tests("def add(a, b):\n    return 3\n", tests)
    assert hardcoded.results == [True, False, False]


def test_candidate_exit_and_prints_do_not_break_grading():
    out = run_tests("print('noise')\nimport sys\ndef f():\n    return 1\n", ["assert f() == 1"])
    assert out.all_passed and "noise" in out.stdout
    out = run_tests("import sys\nsys.exit(3)\n", ["assert 1 == 1"])
    assert out.load_error is not None and not out.all_passed


def test_all_reference_solutions_pass_the_grader():
    """End-to-end validation of the grader: MBPP's own reference code must pass its own tests."""
    from src.data import load_mbpp
    from concurrent.futures import ThreadPoolExecutor
    probs = load_mbpp()
    assert len(probs) == 427

    def ok(p):
        return run_tests(p.reference, p.tests, p.imports, timeout=10).all_passed

    with ThreadPoolExecutor(8) as ex:
        passed = list(ex.map(ok, probs))
    assert sum(passed) / len(probs) >= 0.99, [p.task_id for p, g in zip(probs, passed) if not g]
