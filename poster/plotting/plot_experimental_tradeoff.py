from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

SUMMARY_PATH = ROOT / "results" / "association_summary.csv"
OUT_DIR = ROOT / "poster" / "figures"

OUT_PNG = OUT_DIR / "association_tradeoff_combined.png"
OUT_SVG = OUT_DIR / "association_tradeoff_combined.svg"


def plot_manual_tradeoff(ax) -> None:
    """
    Manual qualitative/empirical subplot for literature-style comparison.

    These numbers are relative placements, not measured from our simulator.
    """
    methods = ["NN", "GNN", "JPDA", "MHT"]

    cost = [1.0, 2.2, 3.6, 4.8]
    accuracy = [1.2, 2.8, 4.0, 4.8]

    ax.axvspan(0.5, 1.7, alpha=0.08)
    ax.axvspan(1.7, 3.0, alpha=0.05)
    ax.axvspan(3.0, 5.3, alpha=0.03)

    ax.scatter(cost, accuracy, s=180, zorder=3)
    ax.plot(cost, accuracy, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    for x, y, label in zip(cost, accuracy, methods):
        ax.text(x + 0.08, y + 0.08, label, fontsize=10, weight="bold")

    ax.set_xlim(0.5, 5.3)
    ax.set_ylim(0.7, 5.3)
    ax.set_xlabel("Computational cost, qualitative")
    ax.set_ylabel("Association quality, qualitative")
    ax.set_title("(a) Literature-based qualitative tradeoff")
    ax.grid(True, alpha=0.25)

    ax.text(
        0.55,
        5.05,
        "Relative placement only",
        fontsize=9,
        alpha=0.8,
    )

def plot_measured_tradeoff(ax, df: pd.DataFrame) -> None:
    """
    Real measured subplot from Scenario D/E benchmark CSV.

    Uses inverse RMSE as tracking quality, so higher y-values are better.
    """
    marker_map = {
        "D": "o",
        "E": "^",
    }

    color_map = {
        "NN": "tab:orange",
        "GNN": "tab:blue",
    }

    df = df.copy()
    df["tracking_quality"] = 1.0 / df["rmse_mean"]

    for _, row in df.iterrows():
        scenario = str(row["scenario"])
        method = str(row["method"])

        x = row["runtime_ms_mean"]
        y = row["tracking_quality"]

        marker = marker_map.get(scenario, "s")
        color = color_map.get(method, "tab:gray")
        label = f"{method}-{scenario}"

        ax.scatter(
            x,
            y,
            s=180,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        if method == "GNN":
            xytext = (8, -18)
        elif method == "NN":
            xytext = (8, 10)
        else:
            xytext = (8, 8)

        ax.annotate(
            label,
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            ha="left",
            va="center",
        )

        if "runtime_ms_std" in row and pd.notna(row["runtime_ms_std"]):
            ax.errorbar(
                x,
                y,
                xerr=row["runtime_ms_std"],
                fmt="none",
                ecolor=color,
                capsize=3,
                alpha=0.6,
                zorder=2,
            )

    ax.set_xlim(8.5, 11.8)
    ax.set_ylim(0.270, 0.290)   

    if method == "GNN":
        xytext = (8, -12)
    elif method == "NN":
        xytext = (8, 8)
    else:
        xytext = (8, 8)

    ax.set_xlabel("Average association runtime per scan [ms]")
    ax.set_ylabel(r"Tracking quality, $1/\mathrm{RMSE}$ [1/m]")
    ax.set_title("(b) Measured harbour simulation tradeoff")
    ax.grid(True, alpha=0.25)

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=color_map["NN"],
            markeredgecolor="black",
            label="NN",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=color_map["GNN"],
            markeredgecolor="black",
            label="GNN",
        ),
    ]

    scenario_handles = []
    for scenario, marker in marker_map.items():
        if scenario in set(df["scenario"].astype(str)):
            scenario_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="",
                    color="black",
                    markersize=8,
                    label=f"Scenario {scenario}",
                )
            )

    ax.legend(
        handles=method_handles + scenario_handles,
        loc="best",
        fontsize=9,
    )

def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing summary file: {SUMMARY_PATH}\n"
            "Run these first:\n"
            "  python poster/experiments/run_association_benchmark.py\n"
            "  python poster/experiments/compute_scores.py"
        )

    df = pd.read_csv(SUMMARY_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_manual_tradeoff(axes[0])
    plot_measured_tradeoff(axes[1], df)

    fig.suptitle(
    "Data association tradeoff: qualitative expectation vs measured tracking performance",
    fontsize=14,
    y=1.04,
)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")

    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved SVG: {OUT_SVG}")

    plt.show()


if __name__ == "__main__":
    main()