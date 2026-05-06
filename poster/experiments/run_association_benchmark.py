from __future__ import annotations

import csv
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim_tracking import sim_tracking


RESULTS_PATH = ROOT / "results" / "association_runs.csv"


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


def init_csv(path: Path, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
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


def extract_cardinality_error(result: dict) -> float:
    """
    Simple cardinality-error proxy for the poster.

    CE = missed ground-truth targets + false confirmed tracks

    Lower is better.
    """
    gt_count = len(result["ground_truth"])

    evaluation = result["evaluation"]
    matched_gt_count = len(evaluation["representatives"])
    false_confirmed = evaluation["confirmed_alive_without_gt"]

    missed_gt = max(gt_count - matched_gt_count, 0)

    return float(missed_gt + false_confirmed)


def run_tracker_once(method: str, scenario: str, seed: int) -> RunResult:
    """
    Runs the real Scenario D/E tracker and extracts the metrics needed
    for the poster benchmark.

    Current intended use:
        scenario = "D"

    Later, when Scenario E is implemented in sim_tracking.py, add E to
    the scenarios list in run_all().
    """
    np.random.seed(seed)

    json_file = ROOT / "harbour_sim_output" / f"scenario_{scenario}.json"

    if not json_file.exists():
        raise FileNotFoundError(f"Missing scenario file: {json_file}")

    result = sim_tracking(
        json_file=str(json_file),
        scenario=scenario,
        mode="radar",
        assoc_method=method.lower(),
    )

    stats = result["stats"]
    evaluation = result["evaluation"]

    num_scans = int(stats.get("num_scans", 0))
    runtime_s = float(stats.get("association_runtime_s", 0.0))

    if num_scans <= 0:
        raise ValueError("num_scans was not recorded. Did you add timing stats to sim_tracking.py?")

    avg_assoc_runtime_ms = 1000.0 * runtime_s / num_scans

    avg_total_rmse = evaluation["avg_total_rmse"]
    if avg_total_rmse is None:
        # Penalize total tracking failure.
        motp = 1e6
    else:
        motp = float(avg_total_rmse)

    ce = extract_cardinality_error(result)

    return RunResult(
        method=method.upper(),
        scenario=scenario,
        seed=seed,
        num_scans=num_scans,
        avg_assoc_runtime_ms=avg_assoc_runtime_ms,
        motp=motp,
        ce=ce,
    )


def run_all(
    methods: Iterable[str] = ("NN", "GNN"),
    scenarios: Iterable[str] = ("D",),
    seeds: Iterable[int] = range(10),
) -> None:
    """
    Runs benchmark repetitions.

    Note:
    - The current harbour JSON files are probably deterministic.
    - Multiple seeds are still useful for runtime averaging.
    - Scenario E can be added later by changing scenarios=("D", "E").
    """
    ensure_results_dir()
    init_csv(RESULTS_PATH, overwrite=True)

    for scenario in scenarios:
        for method in methods:
            for seed in seeds:
                result = run_tracker_once(
                    method=method,
                    scenario=scenario,
                    seed=seed,
                )

                append_result(RESULTS_PATH, result)

                print(
                    f"Saved: method={result.method}, scenario={result.scenario}, "
                    f"seed={result.seed}, scans={result.num_scans}, "
                    f"runtime={result.avg_assoc_runtime_ms:.6f} ms/scan, "
                    f"MOTP/RMSE={result.motp:.3f}, CE={result.ce:.3f}"
                )

    print(f"\nSaved raw benchmark results to:\n{RESULTS_PATH}")


if __name__ == "__main__":
    run_all()