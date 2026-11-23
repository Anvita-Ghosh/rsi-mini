"""Build summary.json, a markdown results table and plots strictly from results files.

    python -m src.report [--out results] [--readme README.md]

The table between <!-- RESULTS:START --> and <!-- RESULTS:END --> in the README is
overwritten with generated output. No number is ever typed by hand.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import ROOT, wilson

BLUE, ORANGE, GREY = "#2a6fdb", "#e07b00", "#6b7280"


def read_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def paired_bootstrap(a: List[bool], b: List[bool], n_boot: int = 10000, seed: int = 0):
    """95% bootstrap CI for mean(b) - mean(a) over the same problems (resampling problems)."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def mcnemar_counts(a: List[bool], b: List[bool]):
    return sum((not x) and y for x, y in zip(a, b)), sum(x and (not y) for x, y in zip(a, b))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci_str(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{pct(k / n)} [{pct(lo)}, {pct(hi)}]"


def summarize_run(out: Path) -> Optional[dict]:
    log = read_jsonl(out / "meta_log.jsonl")
    if not log:
        return None
    base = log[0]
    iters = [r for r in log[1:]]
    s = {"dev_n": base["dev_n"], "baseline_dev_pass": base["dev_pass"],
         "iterations": len(iters),
         "accepted": sum(r["decision"] == "accept" for r in iters),
         "rejected": sum(r["decision"].startswith("reject") for r in iters),
         "final_best_dev_pass": iters[-1]["best_dev_pass"] if iters else base["dev_pass"],
         "log": [{k: r.get(k) for k in ("iter", "candidate", "decision", "dev_pass", "dev_n", "best_dev_pass",
                                        "delta_dev_pass", "meta_tokens", "parse_ok", "seconds")} for r in log],
         "meta_tokens": sum(r.get("meta_tokens", 0) for r in iters)}
    best_iter = 0
    for r in iters:
        if r["decision"] == "accept":
            best_iter = r["iter"]
    s["best_iter"] = best_iter
    return s


def usage_totals(out: Path) -> dict:
    """Logical token use of one arm: every evaluated trace + meta-agent calls (cache replays included)."""
    pt = ct = calls = 0
    sec = 0.0
    for f in sorted((out / "traces").glob("*.jsonl")):
        for t in read_jsonl(f):
            pt += t["prompt_tokens"]; ct += t["completion_tokens"]; calls += t["llm_calls"]; sec += t["llm_seconds"]
    meta = sum(r.get("meta_tokens", 0) for r in read_jsonl(out / "meta_log.jsonl"))
    return {"agent_prompt_tokens": pt, "agent_completion_tokens": ct, "agent_llm_calls": calls,
            "meta_agent_tokens": meta, "total_tokens": pt + ct + meta, "llm_seconds_summed_over_workers": round(sec)}


def accepted_edits(out: Path) -> List[dict]:
    return [{"iter": r["iter"], "edits": r.get("edits_applied", [])} for r in read_jsonl(out / "meta_log.jsonl")
            if r.get("decision") == "accept"]


def build(out: Path) -> dict:
    summary: dict = {"main": summarize_run(out), "usage_main": usage_totals(out),
                     "accepted_edits_main": accepted_edits(out)}
    ev = out / "test_eval.json"
    if ev.exists():
        t = json.loads(ev.read_text())
        summary["test"] = t
        a = [x["hidden_pass"] for x in read_jsonl(out / "traces" / "test_baseline.jsonl")]
        b = [x["hidden_pass"] for x in read_jsonl(out / "traces" / "test_best.jsonl")]
        if a and len(a) == len(b):
            lo, hi = paired_bootstrap(a, b)
            gain, loss = mcnemar_counts(a, b)
            summary["test_paired"] = {"diff": sum(b) / len(b) - sum(a) / len(a), "diff_ci95_bootstrap": [lo, hi],
                                      "best_only_solved": gain, "baseline_only_solved": loss}
    ctrl = out.parent / (out.name + "_control_shuffled") if out.name == "results" else None
    if ctrl and ctrl.exists():
        summary["control_shuffled"] = summarize_run(ctrl)
        summary["usage_control"] = usage_totals(ctrl)
        summary["accepted_edits_control"] = accepted_edits(ctrl)
        if (ctrl / "test_eval.json").exists():
            summary["control_shuffled_test"] = json.loads((ctrl / "test_eval.json").read_text())
    return summary


def table(summary: dict) -> str:
    m, t = summary.get("main"), summary.get("test")
    L: List[str] = []
    if m:
        n = m["dev_n"]
        L += ["**Meta-loop log (DEV = acceptance gate)**", "",
              "| iter | candidate | decision | dev pass | best dev so far | Δ dev (problems) |",
              "|---|---|---|---|---|---|"]
        for r in m["log"]:
            L.append(f"| {r['iter']} | {r['candidate']} | {r['decision']} | {r['dev_pass']}/{r['dev_n']} | "
                     f"{r['best_dev_pass']}/{r['dev_n']} | {r.get('delta_dev_pass', '–') if r.get('delta_dev_pass') is not None else '–'} |")
        L.append("")
    if t:
        nt = t["n_test"]
        g = t["genomes"]
        L += [f"**Final evaluation on the untouched TEST split (n={nt}, each genome evaluated once; 95% Wilson CIs)**", "",
              "| genome | hidden-test pass (the real score) | visible-test pass | passes visible but fails hidden | prompt tok | completion tok | mean rounds | cost |",
              "|---|---|---|---|---|---|---|---|"]
        for name in ("baseline", "best"):
            x = g[name]
            L.append(f"| {name} | {ci_str(x['hidden_pass'], nt)} | {ci_str(x['visible_pass'], nt)} | "
                     f"{x['visible_but_hidden_fail']} ({pct(x['visible_but_hidden_fail'] / nt)}) | "
                     f"{x['prompt_tokens']:,} | {x['completion_tokens']:,} | {x['mean_rounds']:.2f} | $0.00 (local model) |")
        L.append("")
        if m:
            bd, bt = m["baseline_dev_pass"] / m["dev_n"], g["baseline"]["hidden_pass"] / nt
            fd, ft = m["final_best_dev_pass"] / m["dev_n"], g["best"]["hidden_pass"] / nt
            L += ["**Dev-to-test gap (does the dev gain survive?)**", "",
                  "| genome | dev | test | dev − test |", "|---|---|---|---|",
                  f"| baseline | {pct(bd)} | {pct(bt)} | {100 * (bd - bt):+.1f} pts |",
                  f"| best (selected on dev) | {pct(fd)} | {pct(ft)} | {100 * (fd - ft):+.1f} pts |", ""]
        p = summary.get("test_paired")
        if p:
            L += [f"**Best − baseline on test:** {100 * p['diff']:+.1f} pts, paired-bootstrap 95% CI "
                  f"[{100 * p['diff_ci95_bootstrap'][0]:+.1f}, {100 * p['diff_ci95_bootstrap'][1]:+.1f}] "
                  f"({p['best_only_solved']} problems solved only by best, {p['baseline_only_solved']} only by baseline).", ""]
    c = summary.get("control_shuffled")
    if c:
        L += ["**Control: meta-agent given shuffled (irrelevant) failures**", "",
              f"accepted {c['accepted']} / {c['iterations']} candidates; dev {c['baseline_dev_pass']}/{c['dev_n']} → "
              f"{c['final_best_dev_pass']}/{c['dev_n']}", ""]
        ct = summary.get("control_shuffled_test")
        if ct:
            x = ct["genomes"]["best"]
            L += [f"control-best on test: {ci_str(x['hidden_pass'], ct['n_test'])}", ""]
    return "\n".join(L)


def _edit_str(e: dict) -> str:
    op = e.get("op")
    if op == "add_skill":
        return f"add tip [{e.get('category')}]: \"{e.get('tip')}\""
    if op == "set_param":
        return f"set {e.get('name')} = {e.get('value')}"
    if op == "edit_skill":
        return f"edit {e.get('id')}: \"{e.get('tip')}\""
    if op == "set_system_prompt":
        return "rewrite the system prompt"
    return str(op)


def analysis(summary: dict) -> str:
    """Findings paragraph generated from the numbers, with conditional wording (never hand-typed)."""
    m, t, p = summary.get("main"), summary.get("test"), summary.get("test_paired")
    if not (m and t and p):
        return ""
    nt, n, g = t["n_test"], m["dev_n"], t["genomes"]
    L = ["### Findings (auto-generated from the results files)", ""]
    d_gain = m["final_best_dev_pass"] - m["baseline_dev_pass"]
    lo, hi = p["diff_ci95_bootstrap"]
    L.append(f"* **Real meta-loop.** {m['accepted']} of {m['iterations']} candidates were accepted. Dev hidden-test passes went "
             f"{m['baseline_dev_pass']}/{n} → {m['final_best_dev_pass']}/{n} ({d_gain:+d} problems). On the untouched test split the "
             f"selected genome scored {pct(g['best']['hidden_pass'] / nt)} vs {pct(g['baseline']['hidden_pass'] / nt)} for the baseline "
             f"({100 * p['diff']:+.1f} pts, paired-bootstrap 95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]).")
    if lo > 0:
        L.append("  The test CI excludes zero: in this run the dev gain **did transfer** to test.")
    elif hi < 0:
        L.append("  The test CI is entirely below zero: the selected genome was **worse** on test than the baseline.")
    elif d_gain > 0:
        L.append("  The test CI includes zero: **the dev gain did not demonstrably transfer to test** (a negative/null result).")
    else:
        L.append("  No dev gain was selected, and no test difference is detectable.")
    bd = m["baseline_dev_pass"] / n - g["baseline"]["hidden_pass"] / nt
    fd = m["final_best_dev_pass"] / n - g["best"]["hidden_pass"] / nt
    L.append(f"* **Dev-to-test gap.** Baseline dev−test = {100 * bd:+.1f} pts; selected genome dev−test = {100 * fd:+.1f} pts. "
             f"A larger gap for the selected genome is what selection on the dev set (winner's curse) predicts.")
    acc = summary.get("accepted_edits_main", [])
    if acc:
        L.append("* **What was accepted:** " + "; ".join(
            f"iter {a['iter']}: " + ", ".join(_edit_str(e) for e in a["edits"]) for a in acc) + ".")
    c, ct = summary.get("control_shuffled"), summary.get("control_shuffled_test")
    if c:
        cg = c["final_best_dev_pass"] - c["baseline_dev_pass"]
        rel = "as large or larger than" if cg >= d_gain else "smaller than"
        L.append(f"* **Control (shuffled failures).** The meta-agent shown *irrelevant* failures was accepted {c['accepted']} of "
                 f"{c['iterations']} times and moved dev {c['baseline_dev_pass']}/{c['dev_n']} → {c['final_best_dev_pass']}/{c['dev_n']} "
                 f"({cg:+d}), a dev gain {rel} the real arm's ({d_gain:+d}).")
        if cg >= d_gain and cg > 0:
            L.append("  So a dev gain of this size is reachable without any real signal in the failure traces: it is consistent with "
                     "selection on a 100-problem dev set picking up noise. (A single control run is suggestive, not conclusive.)")
        acc_c = summary.get("accepted_edits_control", [])
        if acc_c:
            L.append("  Control accepted edits: " + "; ".join(
                f"iter {a['iter']}: " + ", ".join(_edit_str(e) for e in a["edits"]) for a in acc_c) + ".")
        if ct:
            x = ct["genomes"]["best"]
            L.append(f"  Control-best on test: {ci_str(x['hidden_pass'], ct['n_test'])}.")
    b, w = g["baseline"], g["best"]
    L.append(f"* **Visible vs hidden.** Test problems passing the visible example but failing hidden asserts: baseline "
             f"{b['visible_but_hidden_fail']}/{nt} ({pct(b['visible_but_hidden_fail'] / nt)}), selected genome "
             f"{w['visible_but_hidden_fail']}/{nt} ({pct(w['visible_but_hidden_fail'] / nt)}). Visible-test pass rate overstates the real "
             f"score by {100 * (b['visible_pass'] - b['hidden_pass']) / nt:.1f} pts (baseline) and {100 * (w['visible_pass'] - w['hidden_pass']) / nt:.1f} pts (selected).")
    u, uc = summary.get("usage_main"), summary.get("usage_control")
    if u:
        s = f"* **Compute.** Main arm: {u['total_tokens']:,} tokens"
        if uc:
            s += f"; control arm: {uc['total_tokens']:,} tokens"
        L.append(s + " (agent + meta-agent, local model, $0.00). Token counts include responses replayed from the cache.")
    return "\n".join(L)


def plots(summary: dict, out: Path) -> None:
    pd = out / "plots"
    pd.mkdir(exist_ok=True)
    m, t = summary.get("main"), summary.get("test")
    if m:
        n = m["dev_n"]
        fig, ax = plt.subplots(figsize=(7, 4))
        its = [r["iter"] for r in m["log"]]
        ax.plot(its, [100 * r["dev_pass"] / n for r in m["log"]], "o--", color=GREY, label="candidate on DEV")
        ax.plot(its, [100 * r["best_dev_pass"] / n for r in m["log"]], "o-", color=BLUE, label="best-so-far on DEV (selected)")
        for r in m["log"][1:]:
            if r["decision"] == "accept":
                ax.annotate("accepted", (r["iter"], 100 * r["dev_pass"] / n), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        if t:
            nt = t["n_test"]
            for name, it in (("baseline", 0), ("best", m["best_iter"])):
                k = t["genomes"][name]["hidden_pass"]
                lo, hi = wilson(k, nt)
                ax.errorbar([it + 0.12], [100 * k / nt], yerr=[[100 * (k / nt - lo)], [100 * (hi - k / nt)]], fmt="s",
                            color=ORANGE, capsize=3, label="TEST (evaluated once, 95% CI)" if name == "baseline" else None)
        lo_y, hi_y = ax.get_ylim(); ax.set_ylim(lo_y, hi_y + 1.5)
        ax.set_xlabel("meta-iteration"); ax.set_ylabel("hidden-test pass rate (%)")
        ax.set_title("Dev score by iteration vs. final test score")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(pd / "dev_vs_test_by_iteration.png", dpi=150); plt.close(fig)
    if t and m:
        nt, g = t["n_test"], t["genomes"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i, name in enumerate(("baseline", "best")):
            k = g[name]["hidden_pass"]; lo, hi = wilson(k, nt)
            ax.bar(i - 0.2, 100 * k / nt, 0.38, yerr=[[100 * (k / nt - lo)], [100 * (hi - k / nt)]], capsize=4, color=BLUE,
                   label="hidden tests (real score)" if i == 0 else None)
            kv = g[name]["visible_pass"]
            ax.bar(i + 0.2, 100 * kv / nt, 0.38, color=ORANGE, alpha=0.75, label="visible test only" if i == 0 else None)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline", "best genome"])
        ax.set_ylabel("pass rate on TEST (%)"); ax.set_title(f"Baseline vs best on test (n={nt})")
        ax.set_ylim(0, 112); ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=False); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(pd / "baseline_vs_best_test.png", dpi=150); plt.close(fig)


def inject(readme: Path, md: str, tag: str = "RESULTS") -> bool:
    if not readme.exists():
        return False
    s = readme.read_text()
    a, b = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if a not in s or b not in s:
        return False
    readme.write_text(s[:s.index(a) + len(a)] + "\n" + md + "\n" + s[s.index(b):])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results"); ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()
    out = ROOT / args.out
    summary = build(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    md = table(summary)
    (out / "results_table.md").write_text(md)
    plots(summary, out)
    print(md)
    an = analysis(summary)
    (out / "analysis.md").write_text(an)
    print(an)
    print("README updated:", inject(ROOT / args.readme, md), inject(ROOT / args.readme, an, "ANALYSIS"))


if __name__ == "__main__":
    main()
