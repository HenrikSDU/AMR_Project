from __future__ import annotations

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable


# Replace these imports with your real project imports
# from src.tracker.tracker_runner import run_tracker_once


RESULTS_PATH = Path("results/association_runs.csv")


@dataclass
class RunResult:
    method: str
    scenario: str
    seed: int
    num_scans: int
    avg_assoc_runtime_ms: float
    motp: float
    ce: float


def ensure_results_dir() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_csv(path: Path) -> None:
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "scenario",
            "seed",
            "num_scans",
            "avg_assoc_runtime_ms",
            "motp",
            "ce",
        ])


def append_result(path: Path, result: RunResult) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            result.method,
            result.scenario,
            result.seed,
            result.num_scans,
            f"{result.avg_assoc_runtime_ms:.6f}",
            f"{result.motp:.6f}",
            f"{result.ce:.6f}",
        ])


def run_tracker_once(method: str, scenario: str, seed: int) -> RunResult:
    """
    Replace this stub with your real tracker execution.

    Expected behavior:
    - loads scenario D or E with the given seed
    - runs tracker with association method = method
    - measures association runtime per scan
    - computes MOTP and CE
    - returns one RunResult
    """

    # ---------------------------
    # STUB VALUES FOR TESTING ONLY
    # ---------------------------
    if method == "NN" and scenario == "D":
        return RunResult(method, scenario, seed, 120, 1.8 + 0.01 * seed, 18.0 + 0.1 * seed, 0.82 + 0.01 * seed)

    if method == "GNN" and scenario == "D":
        return RunResult(method, scenario, seed, 120, 4.8 + 0.02 * seed, 11.2 + 0.05 * seed, 0.38 + 0.005 * seed)

    if method == "NN" and scenario == "E":
        return RunResult(method, scenario, seed, 180, 2.0 + 0.01 * seed, 21.0 + 0.12 * seed, 1.08 + 0.01 * seed)

    if method == "GNN" and scenario == "E":
        return RunResult(method, scenario, seed, 180, 5.3 + 0.02 * seed, 14.1 + 0.06 * seed, 0.57 + 0.006 * seed)

    raise ValueError(f"Unsupported method/scenario pair: {method=}, {scenario=}")


def run_all(
    methods: Iterable[str] = ("NN", "GNN"),
    scenarios: Iterable[str] = ("D", "E"),
    seeds: Iterable[int] = range(10),
) -> None:
    ensure_results_dir()
    init_csv(RESULTS_PATH)

    for method in methods:
        for scenario in scenarios:
            for seed in seeds:
                result = run_tracker_once(method=method, scenario=scenario, seed=seed)
                append_result(RESULTS_PATH, result)
                print(
                    f"Saved: method={result.method}, scenario={result.scenario}, "
                    f"seed={result.seed}, runtime={result.avg_assoc_runtime_ms:.3f} ms, "
                    f"MOTP={result.motp:.3f}, CE={result.ce:.3f}"
                )


if __name__ == "__main__":
    run_all()