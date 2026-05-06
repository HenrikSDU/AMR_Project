from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# The CSV radar bearing uses a sensor convention where this offset aligns
# radar points with the camera/AIS NED positions in the real dataset.
RADAR_ROT_DEG = -(90 - 16)
CAMERA_ROT_DEG = 28.0


def rotation_matrix_deg(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ],
        dtype=float,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_time(row: dict[str, str]) -> float:
    possible_time_cols = ["time", "timestamp", "t", "Time", "seconds"]

    for col in possible_time_cols:
        if col in row:
            return float(row[col])

    raise ValueError(f"No recognizable time column found. Columns: {list(row)}")


def optional_float(row: dict[str, str], key: str) -> float:
    if key not in row or row[key] == "":
        return np.nan
    return float(row[key])


def optional_int(row: dict[str, str], key: str, default: int = -1) -> int:
    if key not in row or row[key] == "":
        return default
    return int(float(row[key]))


def load_radar_measurements(path: Path) -> list[dict]:
    """
    mm_wave_radar.csv expected fields:
    - time
    - cluster_id
    - range
    - bearing
    - cov_range
    - cov_range_bearing
    - cov_bearing

    bearing is assumed to be in degrees unless values look like radians.
    """
    rows = read_csv_rows(path)

    measurements = []

    for row in rows:
        r = float(row["range"])
        bearing_raw = float(row["bearing"])

        # Heuristic: if bearing magnitude is bigger than ~2pi, treat as degrees.
        if abs(bearing_raw) > 2 * np.pi:
            bearing_rad = np.deg2rad(bearing_raw)
        else:
            bearing_rad = bearing_raw

        bearing_ned = bearing_rad + np.deg2rad(RADAR_ROT_DEG)

        measurements.append(
            {
                "time": row_time(row),
                "sensor_id": "radar",
                "range_m": r,
                "bearing_rad": bearing_ned,
                "north_m": np.nan,
                "east_m": np.nan,
                "is_false_alarm": False,
                "target_id": -1,
                "source_id": optional_int(row, "cluster_id"),
                "cov_range": optional_float(row, "cov_range"),
                "cov_range_bearing": optional_float(row, "cov_range_bearing"),
                "cov_bearing": optional_float(row, "cov_bearing"),
            }
        )

    return measurements


def load_camera_measurements(path: Path) -> list[dict]:
    """
    camera.csv expected fields:
    - time
    - ID
    - X
    - Z
    - sigma_x
    - sigma_z

    Converts camera-frame position to NED-frame position using 28 deg rotation,
    then converts NED position to range/bearing because your EKF camera model
    currently expects polar range-bearing.
    """
    rows = read_csv_rows(path)

    R_cam_to_ned = rotation_matrix_deg(CAMERA_ROT_DEG)

    measurements = []

    for row in rows:
        x_cam = float(row["X"])
        z_cam = float(row["Z"])

        # Assumption: Z = camera-forward, X = camera-right.
        p_cam = np.array([z_cam, x_cam], dtype=float)
        p_ned = R_cam_to_ned @ p_cam

        north = float(p_ned[0])
        east = float(p_ned[1])

        range_m = float(np.hypot(north, east))
        bearing_rad = float(np.arctan2(east, north))

        measurements.append(
            {
                "time": row_time(row),
                "sensor_id": "camera",
                "range_m": range_m,
                "bearing_rad": bearing_rad,
                "north_m": north,
                "east_m": east,
                "is_false_alarm": False,
                "target_id": -1,
                "source_id": optional_int(row, "ID"),
                "sigma_x": optional_float(row, "sigma_x"),
                "sigma_z": optional_float(row, "sigma_z"),
            }
        )

    return measurements


def load_ais_measurements(path: Path) -> list[dict]:
    """
    ais.csv expected fields:
    - time
    - N
    - E
    - heading
    - mmsi
    - ais_id

    AIS is already in NED position.
    """
    rows = read_csv_rows(path)

    measurements = []

    for row in rows:
        mmsi = optional_int(row, "mmsi")
        measurements.append(
            {
                "time": row_time(row),
                "sensor_id": "ais",
                "range_m": np.nan,
                "bearing_rad": np.nan,
                "north_m": float(row["N"]),
                "east_m": float(row["E"]),
                "heading_deg": optional_float(row, "heading"),
                "is_false_alarm": False,
                "target_id": -1,
                "source_id": mmsi,
                "ais_id": optional_int(row, "ais_id"),
                "mmsi": mmsi,
            }
        )

    return measurements


def load_gnss_measurements(path: Path) -> list[dict]:
    """
    gnss.csv expected fields:
    - time
    - N
    - E
    - heading

    GNSS is own-vessel NED position.
    """
    rows = read_csv_rows(path)

    measurements = []

    for row in rows:
        measurements.append(
            {
                "time": row_time(row),
                "sensor_id": "gnss",
                "range_m": np.nan,
                "bearing_rad": np.nan,
                "north_m": float(row["N"]),
                "east_m": float(row["E"]),
                "heading_deg": optional_float(row, "heading"),
                "is_false_alarm": False,
                "target_id": -1,
                "source_id": -1,
            }
        )

    return measurements


def load_real_dataset(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)

    measurements = []

    measurements.extend(load_radar_measurements(data_dir / "mm_wave_radar.csv"))
    measurements.extend(load_camera_measurements(data_dir / "camera.csv"))
    measurements.extend(load_ais_measurements(data_dir / "ais.csv"))
    measurements.extend(load_gnss_measurements(data_dir / "gnss.csv"))

    measurements.sort(key=lambda m: float(m["time"]))

    return measurements
