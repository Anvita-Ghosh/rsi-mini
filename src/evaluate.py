"""Run a genome over a set of problems, grade with HIDDEN tests, and the CLI.

    python -m src.evaluate smoke  --n 10          # baseline on 10 train problems
    python -m src.evaluate meta   [--control shuffled] [--iters 6 --out results]
    python -m src.evaluate final  [--out results] # test split, baseline + best, ONCE
    python -m src.report                          # summary.json, tables, plots
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from .agent import Agent, TaskView, default_genome, genome_hash
from .data import Problem, load_splits, save_split_manifest
from .llm import LLM, llm_from_config
from .sandbox import CONTAINMENT_LEVEL, run_tests

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "configs/default.json") -> dict:
    return json.loads((ROOT / path).read_text())


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _view(p: Problem) -> TaskView:
    # The ONLY information an agent gets: text + first assert (+ imports for the test).
    return TaskView(p.task_id, p.text, p.visible_test, tuple(p.imports))


def solve_and_grade(llm: LLM, genome: dict, p: Problem, cfg: dict) -> dict:
    agent = Agent(llm, genome, cfg.get("max_new_tokens", 700), cfg.get("sandbox_timeout", 5.0), cfg["seed"])
    tr = agent.solve(_view(p))
    out = run_tests(tr["final_code"], p.tests, p.imports, cfg.get("grade_timeout", 10.0))
    tr["visible_pass"] = bool(out.results and out.results[0])        # first assert == the visible test
    tr["hidden_pass"] = out.all_passed                                  # ALL asserts
    tr["hidden_results"] = out.results
    tr["hidden_error"] = next((e for e in out.errors if e), out.load_error)
    tr["problem"] = p.text
    tr["visible_test"] = p.visible_test
    return tr


def summarize(traces: List[dict], wall: float = 0.0) -> dict:
    n = len(traces)
    vis = sum(t["visible_pass"] for t in traces)
    hid = sum(t["hidden_pass"] for t in traces)
    over = sum(t["visible_pass"] and not t["hidden_pass"] for t in traces)
    lo, hi = wilson(hid, n)
    return {"n": n, "visible_pass": vis, "hidden_pass": hid, "visible_but_hidden_fail": over,
            "visible_rate": vis / n if n else 0.0, "hidden_rate": hid / n if n else 0.0,
            "hidden_ci95": [lo, hi],
            "mean_rounds": sum(t["rounds"] for t in traces) / n if n else 0.0,
            "prompt_tokens": sum(t["prompt_tokens"] for t in traces),
            "completion_tokens": sum(t["completion_tokens"] for t in traces),
            "llm_calls": sum(t["llm_calls"] for t in traces),
            "llm_seconds": round(sum(t["llm_seconds"] for t in traces), 1),
            "wall_seconds": round(wall, 1)}


def run_genome(llm: LLM, genome: dict, problems: List[Problem], cfg: dict, label: str = "") -> Tuple[List[dict], dict]:
    t0 = time.time()
    done = 0
    workers = int(cfg.get("workers", 4))

    def one(p):
        nonlocal done
        tr = solve_and_grade(llm, genome, p, cfg)
        done += 1
        if done % 10 == 0 or done == len(problems):
            print(f"    [{label}] {done}/{len(problems)}  ({time.time() - t0:.0f}s)", flush=True)
        return tr

    with ThreadPoolExecutor(workers) as ex:
        traces = list(ex.map(one, problems))
    return traces, summarize(traces, time.time() - t0)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def get_llm(cfg: dict) -> LLM:
    return llm_from_config(cfg, ROOT / cfg.get("cache_path", "cache/llm_cache.jsonl"))


# --------------------------------------------------------------------- CLI
def cmd_smoke(args, cfg):
    splits = load_splits(cfg["seed"], cfg["n_train"], cfg["n_dev"])
    probs = splits["train"][:args.n]
    llm = get_llm(cfg)
    print(f"model={llm.model} containment={CONTAINMENT_LEVEL} problems={len(probs)}")
    traces, s = run_genome(llm, default_genome(), probs, cfg, "smoke")
    out = ROOT / "results" / "smoke.jsonl"
    write_jsonl(out, traces)
    print(json.dumps(s, indent=1))
    for t in traces:
        print(f"  task {t['task_id']:>4}  visible={'Y' if t['visible_pass'] else 'n'}  "
              f"hidden={'Y' if t['hidden_pass'] else 'n'}  rounds={t['rounds']}  "
              f"tok={t['prompt_tokens'] + t['completion_tokens']}  {t['seconds']}s")
    print("usage:", json.dumps(llm.tracker.snapshot()))


def cmd_meta(args, cfg):
    from .meta import run_meta
    if args.iters:
        cfg["meta_iterations"] = args.iters
    if args.n_dev:
        cfg["n_dev_eval"] = args.n_dev
    if args.n_train:
        cfg["n_train_eval"] = args.n_train
    out = ROOT / args.out
    splits = load_splits(cfg["seed"], cfg["n_train"], cfg["n_dev"])
    save_split_manifest(splits, ROOT / "results" / "splits.json", cfg["seed"])
    llm = get_llm(cfg)
    run_meta(llm, splits, cfg, out, control=args.control)
    print("usage:", json.dumps(llm.tracker.snapshot()))


def cmd_final(args, cfg):
    """Evaluate the baseline genome and the best genome on the untouched TEST split, once each."""
    out = ROOT / args.out
    marker = out / "test_eval.json"
    if marker.exists() and not args.force:
        sys.exit(f"{marker} exists: the test split has already been used. Refusing to re-evaluate "
                 f"(delete it deliberately, or pass --force, if you accept that this breaks 'test used once').")
    splits = load_splits(cfg["seed"], cfg["n_train"], cfg["n_dev"])
    test = splits["test"]
    llm = get_llm(cfg)
    genomes = {"baseline": default_genome(), "best": json.loads((out / "best_genome.json").read_text())}
    res = {"model": llm.model, "n_test": len(test), "containment": CONTAINMENT_LEVEL, "genomes": {}}
    if args.baseline_from:   # reuse the already-measured baseline (it is the same genome, same test set)
        src = ROOT / args.baseline_from
        res["genomes"]["baseline"] = json.loads(src.read_text())["genomes"]["baseline"]
        res["baseline_reused_from"] = args.baseline_from
        genomes.pop("baseline")
    for name, g in genomes.items():
        print(f"== TEST: {name} ({genome_hash(g)})", flush=True)
        traces, s = run_genome(llm, g, test, cfg, f"test/{name}")
        write_jsonl(out / "traces" / f"test_{name}.jsonl", traces)
        s["genome_hash"] = genome_hash(g)
        res["genomes"][name] = s
        print(json.dumps(s, indent=1))
    res["usage_this_process"] = llm.tracker.snapshot()
    marker.write_text(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.evaluate")
    ap.add_argument("--config", default="configs/default.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--n", type=int, default=10)
    m = sub.add_parser("meta")
    m.add_argument("--iters", type=int); m.add_argument("--out", default="results")
    m.add_argument("--control", choices=["shuffled"], default=None)
    m.add_argument("--n-dev", dest="n_dev", type=int); m.add_argument("--n-train", dest="n_train", type=int)
    f = sub.add_parser("final"); f.add_argument("--out", default="results"); f.add_argument("--force", action="store_true")
    f.add_argument("--baseline-from", dest="baseline_from", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"smoke": cmd_smoke, "meta": cmd_meta, "final": cmd_final}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
