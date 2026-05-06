from __future__ import annotations

import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "results" / "association_runs.csv"
SUMMARY_PATH = ROOT / "results" / "association_summary.csv"


def normalize_series(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    s_min = series.min()
    s_max = series.max()

    if abs(s_max - s_min) < eps:
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - s_min) / (s_max - s_min)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw results file: {RAW_PATH}\n"
            "Run poster/experiments/run_association_benchmark.py first."
        )

    df = pd.read_csv(RAW_PATH)

    summary = (
        df.groupby(["method", "scenario"], as_index=False)
        .agg(
            runtime_ms_mean=("avg_assoc_runtime_ms", "mean"),
            runtime_ms_std=("avg_assoc_runtime_ms", "std"),
            rmse_mean=("motp", "mean"),
            rmse_std=("motp", "std"),
            ce_mean=("ce", "mean"),
            ce_std=("ce", "std"),
            num_scans_mean=("num_scans", "mean"),
        )
    )

    # Optional normalized score, mainly useful when Scenario E is added.
    rmse_norm = normalize_series(summary["rmse_mean"])
    ce_norm = normalize_series(summary["ce_mean"])

    summary["tracking_score_norm"] = 1.0 - (0.7 * rmse_norm + 0.3 * ce_norm)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Saved summary to:", SUMMARY_PATH)
    print(summary)


if __name__ == "__main__":
    main()