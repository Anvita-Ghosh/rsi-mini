"""A small USEagent-flavoured ReAct agent over coarse, intent-based actions.

Actions (cf. USEagent's CodeRetrieval / Reproduction / ExecuteTests / EditCode / Finish):
    WriteCode        - draft a solution from the problem + visible test + retrieved tips
    WriteTests       - (optional, genome flag) write extra asserts from the task text only
    RunVisibleTest   - execute the code against the visible test (+ own extra tests) in the sandbox
    Revise           - reason about the observed failure and rewrite the code
    Finish           - stop and return the best code

Control flow is a simple ReAct loop (thought -> action -> observation, shared
trace as memory). In this mini version the LLM chooses *content*; the choice of
*next action* follows a fixed rule (pass -> Finish, fail -> Revise). The agent's
behaviour is entirely determined by a JSON "genome" that the meta-agent edits.

The agent only ever receives a `TaskView`: problem text + the FIRST assert.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm import LLM
from .sandbox import run_tests

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the task with one self-contained Python "
    "function (standard library only; no input(), no printing). "
    "Reply with a single ```python code block."
)


def default_genome() -> dict:
    return {
        "name": "baseline",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "skills": [],              # [{id, category, keywords[], tip}]
        "max_rounds": 3,           # code attempts: 1 WriteCode + (max_rounds-1) Revise
        "self_tests": False,       # write own extra asserts before running
        "top_k_tips": 3,           # how many retrieved tips to inject
    }


def genome_hash(g: dict) -> str:
    g2 = {k: v for k, v in g.items() if k != "name"}
    return hashlib.sha1(json.dumps(g2, sort_keys=True).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class TaskView:
    task_id: int
    text: str
    visible_test: str
    imports: tuple = ()


# ----------------------------------------------------------------- helpers
_WORD = re.compile(r"[a-z]+")


def _norm(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def retrieve_tips(skills: List[dict], text: str, k: int) -> List[dict]:
    """Keyword/category retrieval. `general` tips always qualify (low priority)."""
    words = {_norm(w) for w in _WORD.findall(text.lower())}
    scored = []
    for s in skills:
        keys = {_norm(w.lower()) for w in s.get("keywords", [])} | {_norm(s["category"].lower())}
        hits = len(words & keys)
        if s["category"].lower() == "general":
            hits = max(hits, 0.5)
        if hits > 0:
            scored.append((-hits, s["id"], s))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [s for _, _, s in scored[:max(0, int(k))]]


_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> Optional[str]:
    blocks = _CODE_RE.findall(text)
    if blocks:
        return blocks[-1].strip("\n")
    if "def " in text:  # unterminated fence (truncated output)
        return re.sub(r"^```(?:python|py)?\s*\n", "", text.strip()).strip("\n")
    return None


def thought_of(text: str) -> str:
    return _CODE_RE.split(text)[0].strip()[:400]


def extract_asserts(text: str, visible: str, limit: int = 4) -> List[str]:
    out = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if line.startswith("assert ") and line != visible.strip() and line not in out:
            try:
                ast.parse(line)
            except SyntaxError:
                continue
            out.append(line)
    return out[:limit]


# ------------------------------------------------------------------- agent
class Agent:
    def __init__(self, llm: LLM, genome: dict, max_tokens: int = 700, sandbox_timeout: float = 5.0,
                 seed: int = 0):
        self.llm, self.g = llm, genome
        self.max_tokens, self.timeout, self.seed = max_tokens, sandbox_timeout, seed

    # one LLM call, with token bookkeeping into the trace
    def _chat(self, messages, trace, json_mode=False):
        r = self.llm.chat(messages, temperature=0.0, max_tokens=self.max_tokens, seed=self.seed,
                          json_mode=json_mode)
        trace["prompt_tokens"] += r.prompt_tokens
        trace["completion_tokens"] += r.completion_tokens
        trace["llm_calls"] += 1
        trace["llm_seconds"] += r.latency
        return r.text

    def _task_message(self, task: TaskView, tips: List[dict]) -> str:
        msg = f"Task: {task.text}\n\nThe function must pass this example test (it also fixes the " \
              f"function name and signature):\n{task.visible_test}\n"
        if tips:
            msg += "\nUseful tips from past experience:\n" + "\n".join(
                f"- [{t['category']}] {t['tip']}" for t in tips) + "\n"
        return msg + "\nWrite the function."

    def solve(self, task: TaskView) -> dict:
        g = self.g
        t0 = time.time()
        trace = {"task_id": task.task_id, "genome": g.get("name"), "steps": [], "prompt_tokens": 0,
                 "completion_tokens": 0, "llm_calls": 0, "llm_seconds": 0.0}
        tips = retrieve_tips(g["skills"], task.text, g["top_k_tips"])
        trace["tips_used"] = [t["id"] for t in tips]
        system = {"role": "system", "content": g["system_prompt"]}
        first = {"role": "user", "content": self._task_message(task, tips)}
        extra: List[str] = []
        code: Optional[str] = None
        convo: List[dict] = []          # shared memory: alternating assistant replies / observations
        best_visible_code: Optional[str] = None

        for rnd in range(1, int(g["max_rounds"]) + 1):
            # ---- WriteCode (no valid code yet) or Revise
            action = "WriteCode" if code is None else "Revise"
            raw = self._chat([system, first] + convo, trace)
            convo.append({"role": "assistant", "content": raw})
            new_code = extract_code(raw)
            step = {"round": rnd, "action": action, "thought": thought_of(raw)}
            if new_code is None:
                step["observation"] = "no code block produced"
                trace["steps"].append(step)
                convo.append({"role": "user", "content": "Your reply contained no ```python code block. "
                              "Reply with the complete function in one."})
                continue
            code = new_code
            trace["steps"].append(step)

            # ---- WriteTests (optional, once, from the task text only)
            if g.get("self_tests") and not extra:
                tmsg = [system, {"role": "user", "content":
                        f"Task: {task.text}\nExample test:\n{task.visible_test}\n\nWrite 3 NEW, different "
                        f"`assert` statements (one per line, in one ```python block) that a correct "
                        f"implementation must satisfy. Cover edge cases. Do not write the function."}]
                traw = self._chat(tmsg, trace)
                extra = extract_asserts(traw, task.visible_test)
                trace["steps"].append({"round": rnd, "action": "WriteTests", "thought": "",
                                       "observation": f"{len(extra)} extra tests"})

            # ---- RunVisibleTest
            out = run_tests(code, [task.visible_test] + extra, list(task.imports), self.timeout)
            vis_ok = bool(out.results and out.results[0]) and out.load_error is None
            extra_fail = [(t, e) for t, ok, e in zip(extra, out.results[1:], out.errors[1:]) if not ok]
            if vis_ok:
                best_visible_code = code
            if out.load_error:
                obs = f"The code failed to load: {out.load_error}"
            elif not vis_ok:
                obs = f"The example test FAILED: {out.errors[0]}"
            else:
                obs = "The example test passed."
            if vis_ok and extra_fail:
                obs += " But your own extra tests failed (they were written by you and may themselves be " \
                       "wrong):\n" + "\n".join(f"  {t}  -> {e}" for t, e in extra_fail)
            trace["steps"].append({"round": rnd, "action": "RunVisibleTest", "thought": "",
                                   "observation": obs[:500]})

            # ---- decide: Finish or Revise
            if vis_ok and not extra_fail:
                trace["steps"].append({"round": rnd, "action": "Finish", "thought": "", "observation": "all visible/extra tests pass"})
                break
            convo.append({"role": "user", "content": obs + "\n\nBriefly reason about the bug (at most 3 "
                          "sentences), then output the corrected, complete function in one ```python block."})
        else:
            trace["steps"].append({"round": g["max_rounds"], "action": "Finish", "thought": "",
                                   "observation": "round budget exhausted"})

        final = best_visible_code if best_visible_code is not None else code
        trace["final_code"] = final or ""
        trace["visible_pass"] = best_visible_code is not None
        trace["rounds"] = sum(1 for s in trace["steps"] if s["action"] in ("WriteCode", "Revise"))
        trace["n_extra_tests"] = len(extra)
        trace["seconds"] = round(time.time() - t0, 2)
        return trace
