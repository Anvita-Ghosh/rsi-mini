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
