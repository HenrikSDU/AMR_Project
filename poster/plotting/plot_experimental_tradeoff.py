from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

SUMMARY_PATH = ROOT / "results" / "association_summary.csv"
OUT_DIR = ROOT / "poster" / "figures"

OUT_PNG = OUT_DIR / "association_tradeoff_combined.png"
OUT_SVG = OUT_DIR / "association_tradeoff_combined.svg"
OUT_PDF = OUT_DIR / "association_tradeoff_combined.pdf"


def plot_qualitative_tradeoff(ax) -> None:
    """
    Literature-based qualitative tradeoff.

    The coordinates are relative visual placements, not measured values.
    """
    methods = ["NN", "GNN", "JPDA", "MHT"]

    cost = [1.0, 2.2, 3.6, 4.8]
    quality = [1.2, 2.8, 4.0, 4.8]

    ax.axvspan(0.5, 1.7, alpha=0.08)
    ax.axvspan(1.7, 3.0, alpha=0.05)
    ax.axvspan(3.0, 5.3, alpha=0.03)

    ax.scatter(cost, quality, s=180, zorder=3)
    ax.plot(cost, quality, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    for x, y, label in zip(cost, quality, methods):
        ax.text(
            x + 0.08,
            y + 0.08,
            label,
            fontsize=10,
            weight="bold",
        )

    ax.annotate(
        "Higher cost",
        xy=(5.05, 0.85),
        xytext=(3.85, 0.85),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=9,
        va="center",
    )

    ax.annotate(
        "Better ambiguity handling",
        xy=(0.75, 5.05),
        xytext=(0.75, 3.45),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=9,
        rotation=90,
        ha="center",
        va="center",
    )

    ax.text(
        0.58,
        5.15,
        "Relative placement only",
        fontsize=9,
        alpha=0.8,
    )

    ax.set_xlim(0.5, 5.3)
    ax.set_ylim(0.7, 5.35)
    ax.set_xlabel("Computational cost (qualitative)")
    ax.set_ylabel("Association quality / ambiguity handling (qualitative)")
    ax.set_title("(a) Literature-based qualitative tradeoff")
    ax.grid(True, alpha=0.25)


def plot_measured_tradeoff(ax, df: pd.DataFrame) -> None:
    """
    Measured benchmark tradeoff from results/association_summary.csv.

    Uses the score computed from MOTP and CE:
        higher score = better tracking performance.
    """
    marker_map = {
        "D": "o",
        "E": "^",
    }

    color_map = {
        "NN": "tab:orange",
        "GNN": "tab:blue",
    }

    for _, row in df.iterrows():
        method = str(row["method"])
        scenario = str(row["scenario"])

        x = float(row["runtime_ms_mean"])
        y = float(row["score"])

        marker = marker_map.get(scenario, "s")
        color = color_map.get(method, "tab:gray")
        label = f"{method}-{scenario}"

        ax.scatter(
            x,
            y,
            s=170,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            ha="left",
            va="bottom",
        )

        if "runtime_ms_std" in row and pd.notna(row["runtime_ms_std"]):
            ax.errorbar(
                x,
                y,
                xerr=float(row["runtime_ms_std"]),
                fmt="none",
                ecolor=color,
                capsize=3,
                alpha=0.6,
                zorder=2,
            )

    ax.set_xlabel("Average association runtime per scan [ms]")
    ax.set_ylabel("Tracking score from MOTP and CE (higher is better)")
    ax.set_title("(b) Measured harbour simulation tradeoff")
    ax.grid(True, alpha=0.25)

    # Add reasonable padding.
    x_min = df["runtime_ms_mean"].min()
    x_max = df["runtime_ms_mean"].max()
    y_min = df["score"].min()
    y_max = df["score"].max()

    x_pad = max((x_max - x_min) * 0.15, 0.25)
    y_pad = max((y_max - y_min) * 0.15, 0.08)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.05, y_max + y_pad))

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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    plot_qualitative_tradeoff(axes[0])
    plot_measured_tradeoff(axes[1], df)

    fig.suptitle(
        "Data association tradeoff: qualitative expectation vs measured tracking performance",
        fontsize=14,
        y=1.02,
    )

    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved SVG: {OUT_SVG}")
    print(f"Saved PDF: {OUT_PDF}")

    plt.show()


if __name__ == "__main__":
    main()