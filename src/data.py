"""MBPP (sanitized) loading and fixed-seed train/dev/test splitting.

The agent is only ever shown the problem text and the FIRST assert (the
"visible test"). Grading uses ALL asserts (the "hidden tests").
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

HF_NAME = "google-research-datasets/mbpp"
HF_CONFIG = "sanitized"


@dataclass(frozen=True)
class Problem:
    task_id: int
    text: str
    tests: List[str]          # ALL asserts (hidden grading set)
    imports: List[str]        # test_imports from the dataset
    reference: str            # reference solution (never shown to any agent)

    @property
    def visible_test(self) -> str:
        return self.tests[0]


def load_mbpp() -> List[Problem]:
    """Load every MBPP-sanitized problem (all HF splits merged), sorted by task_id."""
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, HF_CONFIG)
    problems: Dict[int, Problem] = {}
    for split in ds.values():
        for r in split:
            problems[int(r["task_id"])] = Problem(
                task_id=int(r["task_id"]),
                text=r["prompt"].strip(),
                tests=list(r["test_list"]),
                imports=list(r["test_imports"]),
                reference=r["code"],
            )
    return [problems[k] for k in sorted(problems)]


def make_splits(problems: List[Problem], seed: int = 0, n_train: int = 100,
                n_dev: int = 100) -> Dict[str, List[Problem]]:
    """Shuffle with a fixed seed and cut into train / dev / test (the remainder)."""
    rng = random.Random(seed)
    order = sorted(problems, key=lambda p: p.task_id)
    rng.shuffle(order)
    return {
        "train": order[:n_train],
        "dev": order[n_train:n_train + n_dev],
        "test": order[n_train + n_dev:],
    }


def load_splits(seed: int = 0, n_train: int = 100, n_dev: int = 100) -> Dict[str, List[Problem]]:
    return make_splits(load_mbpp(), seed, n_train, n_dev)


def save_split_manifest(splits: Dict[str, List[Problem]], path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "dataset": f"{HF_NAME}/{HF_CONFIG}",
                "sizes": {k: len(v) for k, v in splits.items()},
                "task_ids": {k: [p.task_id for p in v] for k, v in splits.items()}}
    path.write_text(json.dumps(manifest, indent=1))
