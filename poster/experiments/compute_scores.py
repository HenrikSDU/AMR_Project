from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("results/association_runs.csv")
SUMMARY_PATH = Path("results/association_summary.csv")


def normalize_series(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    return (series - s_min) / (s_max - s_min + eps)


def compute_tracking_score(motp_norm: pd.Series, ce_norm: pd.Series) -> pd.Series:
    """
    Higher is better.
    Lower MOTP and CE should produce higher score.
    """
    return 1.0 - (0.7 * motp_norm + 0.3 * ce_norm)


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    summary = (
        df.groupby(["method", "scenario"], as_index=False)
        .agg(
            runtime_ms_mean=("avg_assoc_runtime_ms", "mean"),
            runtime_ms_std=("avg_assoc_runtime_ms", "std"),
            motp_mean=("motp", "mean"),
            motp_std=("motp", "std"),
            ce_mean=("ce", "mean"),
            ce_std=("ce", "std"),
        )
    )

    summary["motp_norm"] = normalize_series(summary["motp_mean"])
    summary["ce_norm"] = normalize_series(summary["ce_mean"])
    summary["score"] = compute_tracking_score(summary["motp_norm"], summary["ce_norm"])

    summary.to_csv(SUMMARY_PATH, index=False)

    print("Saved summary to:", SUMMARY_PATH)
    print(summary)


if __name__ == "__main__":
    main()