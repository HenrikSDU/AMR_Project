from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from Coordinate_Frame_Manager import CoordinateFrameManager
from Target_EKF import Target_EKF
from map_background import WebMapConfig
from real_data_adapter import load_real_dataset
from sim_tracking import (
    ManagedTrack,
    group_measurements_by_scan,
    run_multitarget_tracking,
    plot_multitarget_trajectories,
)


DATA_DIR = Path("AMS_project_2026/AMS_project_2026/Experimental data")
REAL_DATA_ORIGIN_LAT_DEG = 55.69014690
REAL_DATA_ORIGIN_LON_DEG = 12.59998830
DEFAULT_REAL_DATA_MAP_CONFIG = WebMapConfig(
    origin_lat_deg=REAL_DATA_ORIGIN_LAT_DEG,
    origin_lon_deg=REAL_DATA_ORIGIN_LON_DEG,
)


class RealDataCoordinateFrameManager(CoordinateFrameManager):
    """Calibration/noise settings for the provided real harbour CSVs."""

    def __init__(self):
        super().__init__(
            radar_pos_ned=np.array([0.0, 0.0], dtype=float),
            camera_pos_ned=np.array([0.0, 0.0], dtype=float),
            camera_boresight=np.deg2rad(28.0),
            camera_fov_half=np.deg2rad(90.0),
            camera_max_range=500.0,
            radar_max_range=1000.0,
            sigma_r_radar=8.0,
            sigma_phi_radar=np.deg2rad(2.0),
            sigma_r_camera=60.0,
            sigma_phi_camera=np.deg2rad(5.0),
            sigma_pos_ais=6.0,
            sigma_pos_gnss=6.0,
        )


def split_ais_segments(
    ais_measurements: list[dict],
    max_gap_s: float = 180.0,
    max_speed_mps: float = 25.0,
) -> list[list[dict]]:
    segments = []
    current = []

    for measurement in sorted(ais_measurements, key=lambda m: m["time"]):
        if not current:
            current.append(measurement)
            continue

        prev = current[-1]
        dt = float(measurement["time"]) - float(prev["time"])
        if dt <= 0:
            continue

        dn = float(measurement["north_m"]) - float(prev["north_m"])
        de = float(measurement["east_m"]) - float(prev["east_m"])
        speed = float(np.hypot(dn, de) / dt)

        if dt > max_gap_s or speed > max_speed_mps:
            segments.append(current)
            current = [measurement]
        else:
            current.append(measurement)

    if current:
        segments.append(current)

    return segments


def state_history_from_ais_segment(segment: list[dict]) -> list[tuple[float, np.ndarray]]:
    states = []

    for idx, measurement in enumerate(segment):
        pos = np.array([measurement["north_m"], measurement["east_m"]], dtype=float)

        if len(segment) == 1:
            vel = np.zeros(2, dtype=float)
        elif idx == 0:
            next_pos = np.array([segment[1]["north_m"], segment[1]["east_m"]], dtype=float)
            dt = max(float(segment[1]["time"]) - float(measurement["time"]), 1e-6)
            vel = (next_pos - pos) / dt
        else:
            prev_pos = np.array([segment[idx - 1]["north_m"], segment[idx - 1]["east_m"]], dtype=float)
            dt = max(float(measurement["time"]) - float(segment[idx - 1]["time"]), 1e-6)
            vel = (pos - prev_pos) / dt

        states.append((
            float(measurement["time"]),
            np.array([pos[0], pos[1], vel[0], vel[1]], dtype=float),
        ))

    return states


def make_ais_only_tracks(
    measurements: list[dict],
    start_track_id: int,
    min_reports: int = 5,
    min_duration_s: float = 30.0,
    min_median_range_m: float = 1000.0,
) -> list[ManagedTrack]:
    by_mmsi = defaultdict(list)
    for measurement in measurements:
        if measurement["sensor_id"] != "ais":
            continue
        mmsi = int(measurement.get("mmsi", -1))
        if mmsi <= 0:
            continue
        by_mmsi[mmsi].append(measurement)

    tracks = []
    next_track_id = start_track_id

    for mmsi, ais_measurements in sorted(by_mmsi.items()):
        for segment in split_ais_segments(ais_measurements):
            if len(segment) < min_reports:
                continue

            duration_s = float(segment[-1]["time"]) - float(segment[0]["time"])
            if duration_s < min_duration_s:
                continue

            positions = np.array(
                [[m["north_m"], m["east_m"]] for m in segment],
                dtype=float,
            )
            median_range = float(np.median(np.linalg.norm(positions, axis=1)))
            if median_range < min_median_range_m:
                continue

            state_history = state_history_from_ais_segment(segment)
            x_final = state_history[-1][1].copy()
            ekf = Target_EKF(
                x_final,
                np.diag([36.0, 36.0, 100.0, 100.0]),
                dt=1.0,
            )

            track = ManagedTrack(
                track_id=next_track_id,
                ekf=ekf,
                status="confirmed",
                last_update_time=float(segment[-1]["time"]),
                hits=len(segment),
                misses=0,
                update_counts=Counter({"ais": len(segment)}),
                state_history=state_history,
                measurement_history=[
                    {
                        "time": float(m["time"]),
                        "sensor_id": "ais",
                        "is_false_alarm": False,
                        "target_id": -1,
                    }
                    for m in segment
                ],
                birth_sensor="ais",
                first_detection_position=state_history[0][1][:2].copy(),
                first_detection_time=float(segment[0]["time"]),
                velocity_initialized=True,
                lifecycle_sensors={"ais"},
                was_confirmed=True,
            )
            tracks.append(track)
            next_track_id += 1

    return tracks


def run_real_tracking(
    assoc_method: str = "gnn",
    show_plots: bool = True,
    map_config: WebMapConfig | None = DEFAULT_REAL_DATA_MAP_CONFIG,
):
    measurements = load_real_dataset(DATA_DIR)
    cfm = RealDataCoordinateFrameManager()

    scans = group_measurements_by_scan(
        measurements,
        sensor_ids=("radar", "camera", "ais", "gnss"),
    )

    tracks, innov_hist, S_hist, stats = run_multitarget_tracking(
        scans,
        assoc_method=assoc_method,
        gt_multi=None,
        cfm=cfm,
        initiation_sensors=("radar",),
        lifecycle_sensor_ids=("radar",),
        tentative_delete_misses=2,
        delete_misses=8,
    )

    next_track_id = max((track.track_id for track in tracks), default=-1) + 1
    ais_only_tracks = make_ais_only_tracks(
        measurements,
        start_track_id=next_track_id,
    )
    tracks.extend(ais_only_tracks)

    print("\nReal-data tracking complete")
    print(f"Number of tracks: {len(tracks)}")
    print(f"Confirmed/coasting tracks: {sum(t.status in {'confirmed', 'coasting'} for t in tracks)}")
    print(f"AIS-only MMSI tracks: {len(ais_only_tracks)}")

    if show_plots:
        # No ground truth is available here, so pass empty GT.
        plot_multitarget_trajectories(
            gt_multi={},
            tracks=tracks,
            measurements=measurements,
            scenario="Real data",
            show_ended_tracks=False,
            show_unassigned_tracks=True,
            ended_track_min_lifetime_s=None,
            track_min_states=5,
            track_min_lifetime_s=50.0,
            cfm=cfm,
            map_config=map_config,
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
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Disable the OSM background for the real-data trajectory plot.",
    )
    parser.add_argument(
        "--map-origin-lat",
        type=float,
        default=REAL_DATA_ORIGIN_LAT_DEG,
        help="Latitude of the WGS84 point anchored to the local map origin.",
    )
    parser.add_argument(
        "--map-origin-lon",
        type=float,
        default=REAL_DATA_ORIGIN_LON_DEG,
        help="Longitude of the WGS84 point anchored to the local map origin.",
    )
    parser.add_argument(
        "--map-origin-north",
        type=float,
        default=0.0,
        help="Local North coordinate in metres for the WGS84 anchor point.",
    )
    parser.add_argument(
        "--map-origin-east",
        type=float,
        default=0.0,
        help="Local East coordinate in metres for the WGS84 anchor point.",
    )
    parser.add_argument(
        "--map-zoom",
        type=int,
        default=16,
        help="Requested OSM zoom level. It is reduced automatically if too many tiles are needed.",
    )
    parser.add_argument(
        "--map-alpha",
        type=float,
        default=0.82,
        help="Map background opacity.",
    )
    parser.add_argument(
        "--map-cache-dir",
        type=Path,
        default=Path(".tile_cache/osm"),
        help="Directory used to cache downloaded map tiles.",
    )
    parser.add_argument(
        "--map-tile-url",
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        help="Tile URL template with {z}, {x}, and {y} placeholders.",
    )

    args = parser.parse_args()

    map_config = None
    if not args.no_map:
        map_config = WebMapConfig(
            origin_lat_deg=args.map_origin_lat,
            origin_lon_deg=args.map_origin_lon,
            origin_north_m=args.map_origin_north,
            origin_east_m=args.map_origin_east,
            zoom=args.map_zoom,
            alpha=args.map_alpha,
            cache_dir=args.map_cache_dir,
            tile_url_template=args.map_tile_url,
        )

    run_real_tracking(
        assoc_method=args.assoc,
        show_plots=not args.no_plots,
        map_config=map_config,
    )
