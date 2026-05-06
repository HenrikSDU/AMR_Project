from __future__ import annotations

from pathlib import Path

from real_data_adapter import load_real_dataset
from sim_tracking import (
    group_measurements_by_scan,
    run_multitarget_tracking,
    plot_multitarget_trajectories,
)


DATA_DIR = Path("AMS_project_2026/AMS_project_2026/Experimental data")


def run_real_tracking(assoc_method: str = "gnn", show_plots: bool = True):
    measurements = load_real_dataset(DATA_DIR)

    scans = group_measurements_by_scan(
        measurements,
        sensor_ids=("radar", "camera", "ais", "gnss"),
    )

    tracks, innov_hist, S_hist, stats = run_multitarget_tracking(
        scans,
        assoc_method=assoc_method,
        gt_multi=None,
        initiation_sensors=("radar", "ais"),
        lifecycle_sensor_ids=("radar", "ais"),
    )

    print("\nReal-data tracking complete")
    print(f"Number of tracks: {len(tracks)}")
    print(f"Confirmed/coasting tracks: {sum(t.status in {'confirmed', 'coasting'} for t in tracks)}")

    if show_plots:
        # No ground truth is available here, so pass empty GT.
        plot_multitarget_trajectories(
            gt_multi={},
            tracks=tracks,
            measurements=measurements,
            scenario="Real data",
            show_ended_tracks=True,
            ended_track_min_lifetime_s=20.0,
        )

    return {
        "tracks": tracks,
        "measurements": measurements,
        "innov": innov_hist,
        "S": S_hist,
        "stats": stats,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run real harbour dataset tracking.")
    parser.add_argument(
        "--assoc",
        choices=["gnn", "nn"],
        default="gnn",
        help="Association method.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run without opening figures.",
    )

    args = parser.parse_args()

    run_real_tracking(
        assoc_method=args.assoc,
        show_plots=not args.no_plots,
    )