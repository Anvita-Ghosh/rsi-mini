"""Meta-agent: proposes genome edits from TRAIN failure traces; DEV gates acceptance.

Per iteration:
  1. take the current best genome's TRAIN traces (re-run only if the best changed)
  2. show the meta-agent a handful of failures (+ train stats) -> it returns JSON edits
  3. validate/apply the edits -> candidate genome
  4. score the candidate on DEV (hidden-test pass rate)
  5. ACCEPT iff candidate_dev_passes >= best_dev_passes + margin, else REJECT
Every candidate, its dev score and the decision are appended to meta_log.jsonl.

The meta-agent never sees DEV or TEST problems. It does see TRAIN hidden-test
failure messages (that is its "experience"). TEST is untouched until `final`.

Control (--control shuffled): the failure cases handed to the meta-agent have
their (code, error) fields shuffled across problems, so they carry no real
signal. If gains persist, they come from noise/selection, not learning.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .agent import default_genome, genome_hash
from .evaluate import run_genome, write_jsonl, summarize
from .llm import LLM

MAX_SKILLS, MAX_TIP, MAX_KEYWORDS, MAX_PROMPT = 12, 240, 8, 1200

META_SYSTEM = (
    "You are a meta-agent that improves a Python coding agent by editing its JSON 'genome'. "
    "You are shown the current genome and examples of problems the agent failed. "
    "Propose a SMALL number of general, reusable improvements. Reply with JSON only."
)

META_INSTRUCTIONS = """\
HOW THE AGENT WORKS: for each problem it sees only the task text and ONE example assert. Loop: WriteCode ->
(optional WriteTests) -> RunVisibleTest -> Revise (using the failure message) -> ... -> Finish, up to
`max_rounds` code attempts. It is graded on HIDDEN asserts it never sees, so code that only satisfies the
example assert (e.g. hard-coding it) counts as a failure.

GENOME FIELDS YOU MAY EDIT
- system_prompt: string, <= %(mp)d chars.
- skills: list of {id, category, keywords[], tip}. A tip is a short GENERAL lesson (<= %(mt)d chars), injected
  into the prompt when the task text matches its keywords/category. Category "general" is always eligible.
  Max %(ms)d skills, <= %(mk)d keywords each. Tips must be general, never about one specific problem.
- max_rounds: integer 1..5.   - top_k_tips: integer 0..6.   - self_tests: true/false (agent writes its own extra asserts).

OUTPUT FORMAT (JSON only):
{"analysis": "<2-3 sentences: what patterns cause the failures>",
 "edits": [
   {"op": "add_skill", "category": "strings", "keywords": ["string","substring"], "tip": "..."},
   {"op": "edit_skill", "id": "s1", "tip": "..."},
   {"op": "remove_skill", "id": "s2"},
   {"op": "set_system_prompt", "value": "..."},
   {"op": "set_param", "name": "max_rounds", "value": 4}
 ]}
Return at most 3 edits.
""" % {"mp": MAX_PROMPT, "mt": MAX_TIP, "ms": MAX_SKILLS, "mk": MAX_KEYWORDS}


# ------------------------------------------------------------ edit application
def apply_edits(genome: dict, edits) -> Tuple[dict, List[dict], List[dict]]:
    """Validate + apply edits to a copy of the genome. Returns (new, applied, rejected)."""
    g = copy.deepcopy(genome)
    applied, rejected = [], []
    if not isinstance(edits, list):
        return g, [], [{"edit": edits, "why": "edits is not a list"}]

    def bad(e, why):
        rejected.append({"edit": e, "why": why})

    next_id = 1 + max([int(s["id"][1:]) for s in g["skills"] if s["id"][1:].isdigit()] + [0])
    for e in edits[:3]:
        if not isinstance(e, dict):
            bad(e, "not an object"); continue
        op = e.get("op")
        try:
            if op == "add_skill":
                tip, cat = str(e["tip"]).strip(), str(e["category"]).strip().lower()
                kws = [str(k).strip().lower() for k in e.get("keywords", []) if str(k).strip()][:MAX_KEYWORDS]
                if not tip or not cat:
                    bad(e, "empty tip/category")
                elif len(tip) > MAX_TIP:
                    bad(e, "tip too long")
                elif len(g["skills"]) >= MAX_SKILLS:
                    bad(e, "too many skills")
                elif any(s["tip"] == tip for s in g["skills"]):
                    bad(e, "duplicate tip")
                else:
                    g["skills"].append({"id": f"s{next_id}", "category": cat, "keywords": kws, "tip": tip})
                    next_id += 1; applied.append(e)
            elif op == "edit_skill":
                s = next((s for s in g["skills"] if s["id"] == e["id"]), None)
                tip = str(e.get("tip", s["tip"] if s else "")).strip() if s else ""
                if s is None: bad(e, "unknown skill id")
                elif not tip or len(tip) > MAX_TIP: bad(e, "bad tip length")
                else:
                    s["tip"] = tip
                    if "keywords" in e:
                        s["keywords"] = [str(k).strip().lower() for k in e["keywords"] if str(k).strip()][:MAX_KEYWORDS]
                    applied.append(e)
            elif op == "remove_skill":
                n0 = len(g["skills"]); g["skills"] = [s for s in g["skills"] if s["id"] != e["id"]]
                (applied if len(g["skills"]) < n0 else rejected).append(e if len(g["skills"]) < n0 else {"edit": e, "why": "unknown skill id"})
            elif op == "set_system_prompt":
                v = str(e["value"]).strip()
                if not v or len(v) > MAX_PROMPT: bad(e, "system prompt empty or too long")
                else: g["system_prompt"] = v; applied.append(e)
            elif op == "set_param":
                name, v = e["name"], e["value"]
                if name == "max_rounds" and isinstance(v, int) and 1 <= v <= 5: g[name] = v; applied.append(e)
                elif name == "top_k_tips" and isinstance(v, int) and 0 <= v <= 6: g[name] = v; applied.append(e)
                elif name == "self_tests" and isinstance(v, bool): g[name] = v; applied.append(e)
                else: bad(e, "unknown/out-of-range param")
            else:
                bad(e, f"unknown op {op!r}")
        except (KeyError, TypeError, ValueError) as ex:
            bad(e, f"malformed: {type(ex).__name__}")
    return g, applied, rejected


def parse_json(text: str) -> Optional[dict]:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                d = json.loads(text[i:j + 1]); return d if isinstance(d, dict) else None
            except json.JSONDecodeError:
                return None
    return None


# --------------------------------------------------------------- proposing
def failure_cases(traces: List[dict], k: int, rng: random.Random, control: Optional[str]) -> List[dict]:
    fails = [t for t in traces if not t["hidden_pass"]]
    rng.shuffle(fails)
    cases = [{"problem": t["problem"], "visible_test": t["visible_test"],
              "passed_visible_test": t["visible_pass"], "final_code": t["final_code"][:450],
              "hidden_failure": (t["hidden_error"] or "")[:200]} for t in fails[:k]]
    if control == "shuffled" and len(cases) > 1:   # break the link between problem and its failure
        codes = [(c["final_code"], c["hidden_failure"], c["passed_visible_test"]) for c in cases]
        shift = rng.randrange(1, len(cases))
        codes = codes[shift:] + codes[:shift]
        for c, (code, err, pv) in zip(cases, codes):
            c["final_code"], c["hidden_failure"], c["passed_visible_test"] = code, err, pv
    return cases


def propose(llm: LLM, genome: dict, train_summary: dict, cases: List[dict], cfg: dict, seed: int):
    g_view = {k: v for k, v in genome.items() if k != "name"}
    body = (META_INSTRUCTIONS + "\nCURRENT GENOME:\n" + json.dumps(g_view, indent=1) +
            f"\n\nTRAIN RESULTS: {train_summary['hidden_pass']}/{train_summary['n']} pass hidden asserts; "
            f"{train_summary['visible_pass']}/{train_summary['n']} pass the visible example; "
            f"{train_summary['visible_but_hidden_fail']} pass the example but fail hidden asserts.\n\n"
            "FAILED EXAMPLES:\n" + "\n".join(
                f"--- #{i + 1}\nTask: {c['problem']}\nExample: {c['visible_test']}\n"
                f"Passed example: {c['passed_visible_test']}\nFinal code:\n{c['final_code']}\n"
                f"Hidden-test failure: {c['hidden_failure']}" for i, c in enumerate(cases)) +
            "\n\nPropose edits now (JSON only).")
    r = llm.chat([{"role": "system", "content": META_SYSTEM}, {"role": "user", "content": body}],
                 temperature=cfg.get("meta_temperature", 0.7), max_tokens=900, seed=seed, json_mode=True)
    return parse_json(r.text), r


# ---------------------------------------------------------------- main loop
def run_meta(llm: LLM, splits: Dict[str, list], cfg: dict, out: Path, control: Optional[str] = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "genomes").mkdir(exist_ok=True)
    train = splits["train"][:cfg.get("n_train_eval") or None]
    dev = splits["dev"][:cfg.get("n_dev_eval") or None]
    margin = int(cfg.get("accept_margin_problems", 3))
    rng = random.Random(cfg["seed"] + (1000 if control else 0))
    log_path = out / "meta_log.jsonl"
    log_path.write_text("")

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("  LOG", json.dumps({k: row[k] for k in row if k in
              ("iter", "candidate", "dev_pass", "best_dev_pass", "decision")}), flush=True)

    best = default_genome()
    print(f"== iteration 0: baseline on train({len(train)}) and dev({len(dev)})", flush=True)
    best_train_traces, best_train = run_genome(llm, best, train, cfg, "base/train")
    dev_traces, best_dev = run_genome(llm, best, dev, cfg, "base/dev")
    write_jsonl(out / "traces" / "baseline_dev.jsonl", dev_traces)
    write_jsonl(out / "traces" / "baseline_train.jsonl", best_train_traces)
    (out / "genomes" / "baseline.json").write_text(json.dumps(best, indent=1))
    seen = {genome_hash(best): best_dev}
    log({"iter": 0, "candidate": "baseline", "genome_hash": genome_hash(best), "parent": None,
         "train": best_train, "dev": best_dev, "dev_pass": best_dev["hidden_pass"],
         "dev_n": best_dev["n"], "best_dev_pass": best_dev["hidden_pass"], "decision": "baseline",
         "control": control})

    for it in range(1, int(cfg.get("meta_iterations", 6)) + 1):
        t0 = time.time()
        print(f"== iteration {it}: proposing from {best['name']}", flush=True)
        cases = failure_cases(best_train_traces, int(cfg.get("meta_max_failures", 6)), rng, control)
        used_before = llm.tracker.snapshot()
        proposal, mresp = propose(llm, best, best_train, cases, cfg, cfg["seed"] + it)
        edits = (proposal or {}).get("edits", [])
        cand, applied, rejected = apply_edits(best, edits)
        cand["name"] = f"iter{it}"
        h = genome_hash(cand)
        row = {"iter": it, "candidate": cand["name"], "genome_hash": h, "parent": best["name"],
               "analysis": str((proposal or {}).get("analysis", ""))[:500],
               "edits_applied": applied, "edits_rejected": rejected,
               "meta_tokens": mresp.prompt_tokens + mresp.completion_tokens,
               "parse_ok": proposal is not None, "control": control,
               "n_failure_cases_shown": len(cases)}
        if h == genome_hash(best) or not applied:
            row.update(decision="reject (no effective change)", dev_pass=best_dev["hidden_pass"],
                       dev_n=best_dev["n"], best_dev_pass=best_dev["hidden_pass"], dev=None, seconds=round(time.time() - t0, 1))
            log(row); continue
        if h in seen:
            cdev = seen[h]
        else:
            dtr, cdev = run_genome(llm, cand, dev, cfg, f"{cand['name']}/dev")
            write_jsonl(out / "traces" / f"{cand['name']}_dev.jsonl", dtr)
            seen[h] = cdev
        (out / "genomes" / f"{cand['name']}.json").write_text(json.dumps(cand, indent=1))
        accept = cdev["hidden_pass"] >= best_dev["hidden_pass"] + margin
        row.update(dev=cdev, dev_pass=cdev["hidden_pass"], dev_n=cdev["n"],
                   dev_visible_pass=cdev["visible_pass"], delta_dev_pass=cdev["hidden_pass"] - best_dev["hidden_pass"],
                   margin_required=margin, decision="accept" if accept else "reject")
        if accept:
            best, best_dev = cand, cdev
            print(f"   accepted {cand['name']}; re-running train for next proposal", flush=True)
            best_train_traces, best_train = run_genome(llm, best, train, cfg, f"{cand['name']}/train")
            write_jsonl(out / "traces" / f"{cand['name']}_train.jsonl", best_train_traces)
            row["train"] = best_train
        row.update(best_dev_pass=best_dev["hidden_pass"], best_after=best["name"],
                   seconds=round(time.time() - t0, 1))
        log(row)

    (out / "best_genome.json").write_text(json.dumps(best, indent=1))
    return best
