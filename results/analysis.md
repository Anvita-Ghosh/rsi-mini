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