from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RADAR_ROT_DEG = -(90-16)
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


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to standardize the time column name to 'time'.
    Adjust this if your CSV uses a different exact name.
    """
    df = df.copy()

    possible_time_cols = ["time", "timestamp", "t", "Time", "seconds"]

    for col in possible_time_cols:
        if col in df.columns:
            df = df.rename(columns={col: "time"})
            return df

    raise ValueError(f"No recognizable time column found. Columns: {list(df.columns)}")


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
    df = pd.read_csv(path)
    df = normalize_time_column(df)

    measurements = []

    for _, row in df.iterrows():
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
                "time": float(row["time"]),
                "sensor_id": "radar",
                "range_m": r,
                "bearing_rad": bearing_ned,
                "north_m": np.nan,
                "east_m": np.nan,
                "is_false_alarm": False,
                "target_id": -1,
                "source_id": int(row["cluster_id"]) if "cluster_id" in row else -1,
                "cov_range": float(row["cov_range"]) if "cov_range" in row else np.nan,
                "cov_range_bearing": float(row["cov_range_bearing"]) if "cov_range_bearing" in row else np.nan,
                "cov_bearing": float(row["cov_bearing"]) if "cov_bearing" in row else np.nan,
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
    df = pd.read_csv(path)
    df = normalize_time_column(df)

    R_cam_to_ned = rotation_matrix_deg(CAMERA_ROT_DEG)

    measurements = []

    for _, row in df.iterrows():
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
                "time": float(row["time"]),
                "sensor_id": "camera",
                "range_m": range_m,
                "bearing_rad": bearing_rad,
                "north_m": north,
                "east_m": east,
                "is_false_alarm": False,
                "target_id": -1,
                "source_id": int(row["ID"]) if "ID" in row else -1,
                "sigma_x": float(row["sigma_x"]) if "sigma_x" in row else np.nan,
                "sigma_z": float(row["sigma_z"]) if "sigma_z" in row else np.nan,
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
    df = pd.read_csv(path)
    df = normalize_time_column(df)

    measurements = []

    for _, row in df.iterrows():
        measurements.append(
            {
                "time": float(row["time"]),
                "sensor_id": "ais",
                "range_m": np.nan,
                "bearing_rad": np.nan,
                "north_m": float(row["N"]),
                "east_m": float(row["E"]),
                "heading_deg": float(row["heading"]) if "heading" in row else np.nan,
                "is_false_alarm": False,
                "target_id": -1,
                "source_id": int(row["ais_id"]) if "ais_id" in row else -1,
                "mmsi": int(row["mmsi"]) if "mmsi" in row else -1,
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
    df = pd.read_csv(path)
    df = normalize_time_column(df)

    measurements = []

    for _, row in df.iterrows():
        measurements.append(
            {
                "time": float(row["time"]),
                "sensor_id": "gnss",
                "range_m": np.nan,
                "bearing_rad": np.nan,
                "north_m": float(row["N"]),
                "east_m": float(row["E"]),
                "heading_deg": float(row["heading"]) if "heading" in row else np.nan,
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