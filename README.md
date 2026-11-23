# rsi-mini: when an agent improves its own harness, is the gain real?

A small, fully reproducible experiment on **validator overfitting in self-improving coding agents**.
A meta-agent edits the "genome" (prompt, learned tips/skills, a few hyperparameters) of a small
USEagent-style coding agent. Edits are accepted only if they improve a **dev** score. I then measure
whether the dev gain survives on a **test** split the meta-agent never touched, and how often the agent
"passes" only because it satisfied the one visible test.

> **Read the results section as a case study, not a benchmark claim.** One benchmark (MBPP), one small
> local model, one seed, ~100-problem splits. Wide confidence intervals are part of the finding.

## Motivation

The project is motivated by *recursive self-improving (RSI) agents*: agents that learn from experience and
improve their own skills, prompts, search strategies and harnesses, possibly guided by a meta-agent that
decides what to improve. Any such loop needs a validator to decide whether an edit helped. That validator
is a finite set of tasks, so repeatedly selecting edits against it invites **overfitting to the validator**.

This is the same failure mode that [USEagent (Applis et al., ICSE 2026)](https://www.cs.purdue.edu/homes/lintan/publications/USEagent-icse26.pdf)
reports as open problems. USEagent is a Meta-Agent running a ReAct-style loop over coarse, intent-based
actions (CodeRetrieval, TestRetrieval, Reproduction, ExecuteTests, EditCode, ReviewPatch, Finish) with shared
task-state memory. Its discussion of open challenges (Section 6, RQ4) names patch **overfitting** (patches that
pass the given tests but miss the real requirement) and the lack of **backtracking**; its threats to validity
(Section 8) name possible **data leakage / memorization**. The paper also suggests generating additional tests as
a mitigation for overfitting. `rsi-mini` asks the question one level up: *if the agent's own harness is being
optimised against a validator, how do we know the improvement is real?*

## Design

```
                 TRAIN failures (traces)             DEV pass-rate gate
   meta-agent  ───────────────────────►  candidate genome  ───────────►  accept iff dev gain >= margin
        ▲                                      │                                   │
        └──────────── best genome ◄────────────┴───────────────────────────────────┘
                                   TEST: untouched until the very end, one evaluation per genome
```

* **Data** (`src/data.py`): MBPP *sanitized* (427 problems, all HF splits merged), shuffled with a fixed seed and split
  100 train / 100 dev / 227 test. The agent sees **only the problem text and the first assert** (the *visible test*).
  Grading runs **all** asserts (the *hidden tests*). Reporting visible-pass vs hidden-pass measures overfitting to the
  visible test (e.g. hard-coding the example).
* **Agent** (`src/agent.py`): a ReAct-style loop (thought → action → observation, with the trace as shared memory) over
  intent-based actions: `WriteCode`, optional `WriteTests` (the agent writes its own extra asserts), `RunVisibleTest`,
  `Revise`, `Finish`. Simplification versus USEagent: the LLM chooses *content*; the next *action* follows a fixed rule
  (pass → Finish, fail → Revise). Its behaviour is fully defined by a JSON **genome**: system prompt, a categorised list
  of learned tips ("skills", retrieved by keyword/category match), `max_rounds`, `top_k_tips`, `self_tests`.
* **Meta-agent** (`src/meta.py`): reads failure traces from the **train** split only (the same LLM plays this role) and
  returns JSON edits (add/edit/remove a skill, set the system prompt, change a parameter). Edits are validated
  (length/range caps, max 3 per proposal). A candidate is **accepted only if dev hidden-test passes improve by at least
  3 problems** (`accept_margin_problems`), otherwise rejected. Every candidate, its dev score and the decision are logged
  to `results/meta_log.jsonl`.
* **Final evaluation** (`src/evaluate.py final`): the baseline genome and the best genome are each run on the test split
  **once**; the command refuses to run again if `results/test_eval.json` exists. Because of this, there is no
  test-score-per-iteration curve (that would burn the test set); the iteration plot shows dev by iteration plus the two
  final test points.
* **Control** (stretch): the same loop, but the failure examples shown to the meta-agent are shuffled so each problem is
  paired with another problem's code and error. If this arm "improves" too, gains come from selection noise, not learning.
* **Sandbox** (`src/sandbox.py`): all generated code runs in a subprocess with a hard timeout (process group killed), in a
  fresh temp directory, with a minimal environment (no keys), stdin closed, CPU/file-size limits and, on macOS, a Seatbelt
  (`sandbox-exec`) profile denying network access, writes outside the temp dir and reads under `/Users`. Unit tests
  (`tests/test_sandbox.py`) check that an infinite loop, a file-delete attempt, a home-directory read, network access and
  env-var leakage are all contained, and that the grader passes all 427 MBPP reference solutions. Docker was not used.
  Not covered: memory caps (macOS does not enforce `RLIMIT_AS`) and fork bombs; the timeout is the backstop.
* **LLM layer** (`src/llm.py`): thin provider-agnostic wrapper (local Ollama by default; any OpenAI-compatible endpoint via
  `LLM_API_KEY` in the environment or a git-ignored `.env`), an on-disk response cache for reproducibility, and a cost/token
  tracker with a hard cap. Keys are never logged or cached.

## How to run

```bash
# 1. free local model (no API key, no cost): installs Ollama under ~/.local/ollama and pulls qwen2.5-coder:7b (~4.7 GB)
bash scripts/setup_ollama.sh

# 2. everything: tests -> meta loop -> test evaluation -> shuffled-failure control -> report (writes the table below)
bash scripts/reproduce.sh
```

Individual steps: `python -m src.evaluate smoke --n 10`, `python -m src.evaluate meta`, `python -m src.evaluate final`,
`python -m src.report`. Config is in `configs/default.json` (seed, split sizes, model, iterations, margin, budget).

**Reproducibility.** Seeds are fixed (split seed 0, LLM seed 0, temperature 0 for the coding agent). Every LLM response is
cached by a hash of (model, messages, sampling settings); the full cache from the reported run is shipped as
`results/llm_cache.jsonl.gz`, and `scripts/reproduce.sh` unpacks and replays it, so the reported numbers can be regenerated
exactly with no model running. A genuinely fresh run (empty cache) on other hardware may differ slightly because local LLM
inference is not bit-exact across machines. Because the test split must be used once, the script refuses to re-run the test
evaluation if `results/test_eval.json` exists (see the comments in the script).

## Results

All numbers below are generated by `python -m src.report` from the files in `results/`; nothing is typed by hand.

<!-- RESULTS:START -->
**Meta-loop log (DEV = acceptance gate)**

| iter | candidate | decision | dev pass | best dev so far | Δ dev (problems) |
|---|---|---|---|---|---|
| 0 | baseline | baseline | 85/100 | 85/100 | – |
| 1 | iter1 | reject | 84/100 | 85/100 | -1 |
| 2 | iter2 | reject | 87/100 | 85/100 | 2 |
| 3 | iter3 | accept | 89/100 | 89/100 | 4 |
| 4 | iter4 | reject | 86/100 | 89/100 | -3 |
| 5 | iter5 | reject | 89/100 | 89/100 | 0 |
| 6 | iter6 | reject | 86/100 | 89/100 | -3 |
| 7 | iter7 | reject | 86/100 | 89/100 | -3 |
| 8 | iter8 | reject | 89/100 | 89/100 | 0 |

**Final evaluation on the untouched TEST split (n=227, each genome evaluated once; 95% Wilson CIs)**

| genome | hidden-test pass (the real score) | visible-test pass | passes visible but fails hidden | prompt tok | completion tok | mean rounds | cost |
|---|---|---|---|---|---|---|---|
| baseline | 83.7% [78.3%, 87.9%] | 87.7% [82.8%, 91.3%] | 9 (4.0%) | 52,753 | 28,659 | 1.29 | $0.00 (local model) |
| best | 83.7% [78.3%, 87.9%] | 89.0% [84.2%, 92.4%] | 12 (5.3%) | 76,024 | 31,837 | 1.37 | $0.00 (local model) |

**Dev-to-test gap (does the dev gain survive?)**

| genome | dev | test | dev − test |
|---|---|---|---|
| baseline | 85.0% | 83.7% | +1.3 pts |
| best (selected on dev) | 89.0% | 83.7% | +5.3 pts |

**Best − baseline on test:** +0.0 pts, paired-bootstrap 95% CI [-2.6, +2.6] (4 problems solved only by best, 4 only by baseline).

**Control: meta-agent given shuffled (irrelevant) failures**

accepted 1 / 8 candidates; dev 85/100 → 91/100

control-best on test: 85.5% [80.3%, 89.5%]

<!-- RESULTS:END -->

Plots: `results/plots/dev_vs_test_by_iteration.png` and `results/plots/baseline_vs_best_test.png`.

<!-- ANALYSIS:START -->
### Findings (auto-generated from the results files)

* **Real meta-loop.** 1 of 8 candidates were accepted. Dev hidden-test passes went 85/100 → 89/100 (+4 problems). On the untouched test split the selected genome scored 83.7% vs 83.7% for the baseline (+0.0 pts, paired-bootstrap 95% CI [-2.6, +2.6]).
  The test CI includes zero: **the dev gain did not demonstrably transfer to test** (a negative/null result).
* **Dev-to-test gap.** Baseline dev−test = +1.3 pts; selected genome dev−test = +5.3 pts. A larger gap for the selected genome is what selection on the dev set (winner's curse) predicts.
* **What was accepted:** iter 3: add tip [general]: "Consider edge cases and hidden tests during your solution design.", edit s1: "Always test your function with a variety of inputs, including edge cases.", set max_rounds = 4.
* **Control (shuffled failures).** The meta-agent shown *irrelevant* failures was accepted 1 of 8 times and moved dev 85/100 → 91/100 (+6), a dev gain as large or larger than the real arm's (+4).
  So a dev gain of this size is reachable without any real signal in the failure traces: it is consistent with selection on a 100-problem dev set picking up noise. (A single control run is suggestive, not conclusive.)
  Control accepted edits: iter 3: add tip [general]: "Always define a function with a clear docstring and handle all edge cases in your code.", set max_rounds = 4, rewrite the system prompt.
  Control-best on test: 85.5% [80.3%, 89.5%].
* **Visible vs hidden.** Test problems passing the visible example but failing hidden asserts: baseline 9/227 (4.0%), selected genome 12/227 (5.3%). Visible-test pass rate overstates the real score by 4.0 pts (baseline) and 5.3 pts (selected).
* **Compute.** Main arm: 814,263 tokens; control arm: 1,006,510 tokens (agent + meta-agent, local model, $0.00). Token counts include responses replayed from the cache.
<!-- ANALYSIS:END -->

## Limitations

* **Small samples.** 100 dev and 227 test problems. A 3-problem dev margin is within sampling noise (a single
  proportion on 100 problems has a 95% CI roughly ±8 points), so "accepted" edits are partly selection on noise
  ("winner's curse") by construction.
* **One benchmark** (MBPP: short, well-known functions; contamination of the model's training data is possible and was not
  measured), **one model** (a 7B local model acting as both coder and meta-agent) and **one seed**. No claims about other
  models or tasks.
* The visible test is the first MBPP assert, so "visible pass" is a weak proxy for the real spec; the hidden tests are
  MBPP's own, which are also imperfect (some MBPP specs are ambiguous).
* The meta-agent is the same 7B model as the coder, and in the reported run the edits it proposed (and that were accepted)
  were generic advice ("consider edge cases", more rounds) in **both** the real and the shuffled-failure arm, i.e. it did not
  visibly extract problem-specific lessons from the traces. A stronger meta-model might behave differently; this was not tested.
* The control arm was run once; it shows that a dev gain of this size is reachable from noise, not how often.
* The meta-agent's instructions tell it the agent is graded on hidden asserts it never sees (so a tip may mention "hidden
  tests"); it is never shown any dev/test problem or any hidden assert from dev/test.
* The meta-agent can only make small, bounded edits (a few tips, a prompt, three parameters). This is far from a fully
  self-modifying agent and is not USEagent-scale.
* The test split is evaluated once per genome by design; there is no per-iteration test curve.
* Sandbox: Seatbelt/subprocess isolation on macOS, not a container; see the Design section for known gaps.

## Process

The code, tests and this README were written with an AI coding assistant (Claude Code) and reviewed by me
(Anvita Ghosh). The results tables are produced by scripts from the logged runs.

## License

MIT, see `LICENSE`.
