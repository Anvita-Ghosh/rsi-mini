"""Run untrusted, model-generated Python in a contained subprocess.

Containment layers (strongest first):
  1. macOS Seatbelt (`sandbox-exec`): no network, no writes outside the run's
     temp dir, no reads under /Users, /Volumes or other temp dirs. Used when
     available and passing a start-up self-check (`CONTAINMENT_LEVEL`).
  2. Always: fresh temp working dir (deleted afterwards), minimal environment
     (no API keys, HOME=temp dir), stdin closed, `python -I`, hard wall-clock
     timeout with the whole process group killed, CPU/file-size rlimits.

Docker was not available on the development machine, so it is not used.
Known gaps: no memory cap (macOS does not enforce RLIMIT_AS) and no fork cap;
the wall-clock timeout and process-group kill are the backstop.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

def _base_python() -> str:
    """The *base* interpreter: a venv python would need to read files under $HOME."""
    for name in ("python3", "python%d.%d" % sys.version_info[:2]):
        cand = os.path.join(sys.base_prefix, "bin", name)
        if os.path.exists(cand) and not os.path.realpath(cand).startswith("/Users"):
            return cand
    return sys.executable


_PYTHON = _base_python()
_SANDBOX_EXEC = shutil.which("sandbox-exec") if sys.platform == "darwin" else None

_HARNESS = r'''
import sys, json, ast
try:
    import resource
    _cpu = int(__import__("os").environ.get("RSI_CPU", "10"))
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu, _cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10**7, 10**7))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except Exception:
    pass
_p = json.load(open("payload.json"))
_ns = {"__name__": "__candidate__"}
_out = {"load_error": None, "results": [], "errors": []}

def _fmt(e):
    return (type(e).__name__ + ": " + str(e))[:300]

def _diff(test_src):
    """For `assert LHS == RHS` report what the code actually returned."""
    try:
        node = ast.parse(test_src).body[0]
        cmp_ = node.test
        if isinstance(cmp_, ast.Compare) and len(cmp_.ops) == 1:
            left = eval(compile(ast.Expression(cmp_.left), "<l>", "eval"), _ns)
            right = eval(compile(ast.Expression(cmp_.comparators[0]), "<r>", "eval"), _ns)
            return "got %s, expected %s" % (repr(left)[:120], repr(right)[:120])
    except BaseException:
        pass
    return "assertion failed"

for _imp in _p["imports"]:
    try:
        exec(_imp, _ns)
    except BaseException:
        pass
try:
    exec(compile(_p["code"], "<candidate>", "exec"), _ns)
except BaseException as e:
    _out["load_error"] = _fmt(e)
for _t in _p["tests"]:
    try:
        exec(compile(_t, "<test>", "exec"), _ns)
        _out["results"].append(True)
        _out["errors"].append(None)
    except AssertionError:
        _out["results"].append(False)
        _out["errors"].append(_diff(_t))
    except BaseException as e:
        _out["results"].append(False)
        _out["errors"].append(_fmt(e))
sys.stdout.write("\n" + _p["nonce"] + json.dumps(_out) + "\n")
sys.stdout.flush()
'''


def _profile(tmp: str) -> str:
    t = os.path.realpath(tmp)
    return (
        "(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n"
        f'(allow file-write* (subpath "{t}") (literal "/dev/null"))\n'
        '(deny file-read* (subpath "/Users") (subpath "/Volumes") (subpath "/private/var/folders"))\n'
        f'(allow file-read* (subpath "{t}"))\n'
    )


@dataclass
class RawResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float
    contained: bool


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run_script(script: str, files: dict, timeout: float, use_seatbelt: bool = True) -> RawResult:
    with tempfile.TemporaryDirectory(prefix="rsi_sbx_") as tmp:
        with open(os.path.join(tmp, "main.py"), "w") as f:
            f.write(script)
        for name, content in files.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(content)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "LANG": "en_US.UTF-8",
               "PYTHONHASHSEED": "0", "RSI_CPU": str(int(timeout) + 2)}
        cmd = [_PYTHON, "-I", "-B", "main.py"]
        contained = bool(use_seatbelt and _SANDBOX_EXEC)
        if contained:
            cmd = [_SANDBOX_EXEC, "-p", _profile(tmp)] + cmd
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True, text=True, errors="replace")
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            out, err = proc.communicate()
        _kill_group(proc)  # reap any stragglers spawned by the code
        return RawResult(proc.returncode, out[-20000:], err[-4000:], timed_out,
                         time.time() - t0, contained)


def run_python(source: str, timeout: float = 5.0) -> RawResult:
    """Run an arbitrary script in the sandbox (used by the sandbox unit tests)."""
    return _run_script(source, {}, timeout)


def _selfcheck() -> bool:
    """Is Seatbelt really working here? (It can fail when nested in another sandbox.)"""
    if not _SANDBOX_EXEC:
        return False
    r = _run_script("print('ok')", {}, 10.0)
    return r.contained and r.returncode == 0 and "ok" in r.stdout


CONTAINMENT_LEVEL = "seatbelt+subprocess" if _selfcheck() else "subprocess-only"
if CONTAINMENT_LEVEL == "subprocess-only":
    _SANDBOX_EXEC = None


@dataclass
class TestOutcome:
    results: List[bool]
    errors: List[Optional[str]]
    load_error: Optional[str]
    timed_out: bool
    stdout: str
    stderr: str

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(self.results) and self.load_error is None


def run_tests(code: str, tests: List[str], imports: Optional[List[str]] = None,
              timeout: float = 5.0) -> TestOutcome:
    """Run `code` then each assert in `tests`; report one bool per assert."""
    nonce = "@@RSI" + secrets.token_hex(8) + "@@"
    payload = json.dumps({"code": code, "tests": tests, "imports": imports or [], "nonce": nonce})
    raw = _run_script(_HARNESS, {"payload.json": payload}, timeout)
    n = len(tests)
    if raw.timed_out:
        return TestOutcome([False] * n, [f"Timeout after {timeout:.0f}s (infinite loop?)"] * n,
                           None, True, raw.stdout[:500], raw.stderr[-500:])
    idx = raw.stdout.rfind(nonce)
    if idx < 0:
        msg = (raw.stderr.strip().splitlines() or ["no result produced"])[-1][:300]
        return TestOutcome([False] * n, [f"Crash: {msg}"] * n, None, False,
                           raw.stdout[:500], raw.stderr[-500:])
    try:
        out = json.loads(raw.stdout[idx + len(nonce):].strip())
    except json.JSONDecodeError:
        return TestOutcome([False] * n, ["Crash: unreadable result"] * n, None, False,
                           raw.stdout[:500], raw.stderr[-500:])
    return TestOutcome(out["results"], out["errors"], out["load_error"], False,
                       raw.stdout[:idx][:500], raw.stderr[-500:])
