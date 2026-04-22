from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


SUMMARY_PATH = Path("results/association_summary.csv")


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)

    marker_map = {
        "D": "o",
        "E": "^",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        x = row["runtime_ms_mean"]
        y = row["score"]
        label = f'{row["method"]}-{row["scenario"]}'
        marker = marker_map[row["scenario"]]

        ax.scatter(x, y, s=180, marker=marker, zorder=3)
        ax.text(x + 0.03, y + 0.01, label, fontsize=11)

    ax.set_xlabel("Average association runtime per scan [ms]")
    ax.set_ylabel("Tracking score (higher is better)")
    ax.set_title("Measured association accuracy vs computational cost")
    ax.grid(True, alpha=0.3)

    # Manual legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, label="Scenario D"),
        plt.Line2D([0], [0], marker="^", linestyle="", markersize=8, label="Scenario E"),
    ]
    ax.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()