from dataclasses import dataclass, field
from typing import Optional

import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.optimize import linear_sum_assignment

from Coordinate_Frame_Manager import CoordinateFrameManager
from Target_EKF import Target_EKF
from data_association import associate, associate_multisensor_slots


@dataclass
class ManagedTrack:
    track_id: int
    ekf: Target_EKF
    status: str
    last_update_time: float
    hits: int = 0
    misses: int = 0
    hit_history: list[bool] = field(default_factory=list)
    update_counts: Counter = field(default_factory=Counter)
    missed_detection_flags: Counter = field(default_factory=Counter)
    state_history: list[tuple[float, np.ndarray]] = field(default_factory=list)
    measurement_history: list[dict] = field(default_factory=list)
    deleted_at: Optional[float] = None
    birth_sensor: str = "radar"
    first_detection_position: Optional[np.ndarray] = None
    first_detection_time: Optional[float] = None
    velocity_initialized: bool = False
    coast_scans: int = 0
    lifecycle_sensors: set[str] = field(default_factory=lambda: {"radar"})
    was_confirmed: bool = False


# HELPER FUNCTIONS

def extract_ground_truth(data):
    gt = data["ground_truth"]["0"]
    gt_times = np.array([row[0] for row in gt], dtype=float)
    gt_states = np.array([row[1:] for row in gt], dtype=float)
    return gt_times, gt_states


def extract_ground_truth_multi(data):
    gt_multi = {}
    for key, rows in data["ground_truth"].items():
        gt_multi[int(key)] = (
            np.array([row[0] for row in rows], dtype=float),
            np.array([row[1:] for row in rows], dtype=float),
        )
    return gt_multi


def interpolate_gt(gt_times, gt_states, ts):
    gt_interp = np.zeros((len(ts), 4))
    for i in range(4):
        gt_interp[:, i] = np.interp(ts, gt_times, gt_states[:, i])
    return gt_interp


def compute_rmse(x_est, gt_interp):
    pos_err = x_est[:, :2] - gt_interp[:, :2]
    rmse = np.sqrt(np.mean(pos_err**2, axis=0))
    total = np.linalg.norm(rmse)
    print(f"RMSE [N, E]: {rmse}")
    print(f"Total RMSE: {total:.2f} m")
    return rmse


def compute_nis(innov_hist, S_hist):
    N = len(innov_hist)
    nis = np.zeros(N)

    for k in range(N):
        y = innov_hist[k]
        S = S_hist[k]
        nis[k] = y @ inv(S) @ y

    return nis


def measurement_vector(m):
    sensor = m["sensor_id"]
    if sensor in ["radar", "camera"]:
        return np.array([m["range_m"], m["bearing_rad"]], dtype=float)
    if sensor == "ais":
        return np.array([m["north_m"], m["east_m"]], dtype=float)
    raise ValueError(f"Unsupported measurement sensor: {sensor}")


def measurement_array(measurements, sensor_id):
    rows = []
    for m in measurements:
        if m["sensor_id"] != sensor_id:
            continue
        if sensor_id in ["radar", "camera"]:
            rows.append((m["time"], m["range_m"], m["bearing_rad"]))
        elif sensor_id == "ais":
            rows.append((m["time"], m["north_m"], m["east_m"]))
    return np.array(rows, dtype=float) if rows else np.empty((0, 3), dtype=float)


def group_measurements_by_event(measurements):
    sensor_order = {"radar": 0, "camera": 1, "ais": 2, "gnss": 3}
    grouped = defaultdict(list)

    for m in measurements:
        grouped[(float(m["time"]), m["sensor_id"])].append(m)

    events = []
    for (time_s, sensor_id), items in grouped.items():
        events.append({
            "time": time_s,
            "sensor_id": sensor_id,
            "measurements": items,
        })

    events.sort(key=lambda e: (e["time"], sensor_order.get(e["sensor_id"], 99)))
    return events


def group_measurements_by_scan(measurements, sensor_ids=("radar", "camera")):
    grouped = defaultdict(list)

    for m in measurements:
        grouped[float(m["time"])].append(m)

    scans = []
    for time_s, items in grouped.items():
        measurements_by_sensor = {sensor_id: [] for sensor_id in sensor_ids}
        for m in items:
            if m["sensor_id"] in measurements_by_sensor:
                measurements_by_sensor[m["sensor_id"]].append(m)

        scans.append({
            "time": time_s,
            "measurements_by_sensor": measurements_by_sensor,
            "sensor_available": {
                sensor_id: len(measurements_by_sensor[sensor_id]) > 0
                for sensor_id in sensor_ids
            },
        })

    scans.sort(key=lambda scan: scan["time"])
    return scans


def polar_to_ned(z, sensor_id, cfm):
    sensor_pos = cfm.get_sensor_position(sensor_id)
    north = sensor_pos[0] + z[0] * np.cos(z[1])
    east = sensor_pos[1] + z[0] * np.sin(z[1])
    return np.array([north, east], dtype=float)


def append_track_state(track, time_s):
    track.state_history.append((float(time_s), track.ekf.x.copy()))


def append_track_measurement(track, measurement):
    track.measurement_history.append({
        "time": float(measurement["time"]),
        "sensor_id": measurement["sensor_id"],
        "is_false_alarm": bool(measurement["is_false_alarm"]),
        "target_id": int(measurement.get("target_id", -1)),
    })


def flatten_scan_detections(scan):
    detections = []

    for sensor_id, measurements in scan["measurements_by_sensor"].items():
        if sensor_id == "gnss":
            continue
        if not scan["sensor_available"].get(sensor_id, False):
            continue

        for det_idx, measurement in enumerate(measurements):
            detections.append({
                "sensor_id": sensor_id,
                "measurement": measurement,
                "sensor_det_idx": det_idx,
                "z": measurement_vector(measurement),
            })

    return detections


def active_tracks(tracks):
    return [track for track in tracks if track.status != "deleted"]


def reportable_tracks(tracks):
    return [track for track in active_tracks(tracks) if track.status in {"confirmed", "coasting"}]


def measurement_position_ned(measurement, sensor_id, cfm):
    z = measurement_vector(measurement)
    if sensor_id in {"radar", "camera"}:
        return polar_to_ned(z, sensor_id, cfm)
    if sensor_id == "ais":
        return z
    raise ValueError(f"Cannot initialise target from sensor: {sensor_id}")


def position_covariance_from_measurement(measurement, sensor_id, cfm):
    if sensor_id == "ais":
        return cfm.R("ais")

    z = measurement_vector(measurement)
    r = float(z[0])
    phi = float(z[1])
    J = np.array(
        [
            [np.cos(phi), -r * np.sin(phi)],
            [np.sin(phi), r * np.cos(phi)],
        ],
        dtype=float,
    )
    return J @ cfm.R(sensor_id) @ J.T


def initiate_track_from_measurement(track_id, measurement, sensor_id, cfm=None, init_dt=1.0):
    if cfm is None:
        cfm = CoordinateFrameManager()

    pos = measurement_position_ned(measurement, sensor_id, cfm)
    x0 = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
    P_pos = position_covariance_from_measurement(measurement, sensor_id, cfm)
    P_pos = P_pos + np.diag([100.0, 100.0])
    P_vel = np.diag([10000.0, 10000.0])
    P0 = np.block([
        [P_pos, np.zeros((2, 2))],
        [np.zeros((2, 2)), P_vel],
    ])
    ekf = Target_EKF(x0, P0, dt=init_dt)
    ekf.cfm = cfm

    track = ManagedTrack(
        track_id=track_id,
        ekf=ekf,
        status="tentative",
        last_update_time=float(measurement["time"]),
        hits=1,
        misses=0,
        birth_sensor=sensor_id,
        first_detection_position=pos.copy(),
        first_detection_time=float(measurement["time"]),
        lifecycle_sensors={"radar", "ais"} if sensor_id == "ais" else {"radar"},
    )
    track.hit_history.append(True)
    track.update_counts[sensor_id] += 1
    append_track_state(track, measurement["time"])
    append_track_measurement(track, measurement)
    return track


def initiate_track_from_radar(track_id, measurement, init_dt=1.0):
    return initiate_track_from_measurement(track_id, measurement, "radar", init_dt=init_dt)


def maybe_initialize_track_velocity(track, measurement, sensor_id, time_s):
    if track.velocity_initialized:
        return
    if track.first_detection_position is None or track.first_detection_time is None:
        return

    dt = float(time_s) - float(track.first_detection_time)
    if dt <= 1e-6:
        return

    pos = measurement_position_ned(measurement, sensor_id, track.ekf.cfm)
    vel = (pos - track.first_detection_position) / dt
    track.ekf.x[2] = vel[0]
    track.ekf.x[3] = vel[1]
    track.velocity_initialized = True


def update_track_lifecycle(
    track,
    got_hit,
    time_s,
    miss_opportunity=True,
    confirmation_hits=3, # M
    confirmation_window=5, # N
    tentative_delete_misses=1,
    delete_misses=10,
):
    if track.status == "deleted":
        return

    if not got_hit and not miss_opportunity:
        return

    track.hit_history.append(bool(got_hit))
    if len(track.hit_history) > confirmation_window:
        track.hit_history = track.hit_history[-confirmation_window:]

    if got_hit:
        track.hits += 1
        track.misses = 0
        track.coast_scans = 0
        if track.status == "coasting":
            track.status = "confirmed"
            track.was_confirmed = True
        if track.status == "tentative" and sum(track.hit_history[-confirmation_window:]) >= confirmation_hits:
            track.status = "confirmed"
            track.was_confirmed = True
    else:
        track.misses += 1
        if track.status == "tentative" and track.misses >= tentative_delete_misses:
            track.status = "deleted"
            track.deleted_at = float(time_s)
        elif track.status == "confirmed":
            track.status = "coasting"
            track.coast_scans = 1
        elif track.status == "coasting":
            track.coast_scans += 1
            if track.misses >= delete_misses:
                track.status = "deleted"
                track.deleted_at = float(time_s)


def update_track_lifecycle_radar_only(track, got_radar_hit, radar_available, time_s):
    """
    Radar-driven track lifecycle:
    - no radar available => no hit/miss opportunity this scan
    - radar available + hit => hit
    - radar available + no hit => miss
    """
    if not radar_available:
        return
    update_track_lifecycle(track, got_radar_hit, time_s, miss_opportunity=True)


def summarize_track_assignment(track):
    assigned = [m["target_id"] for m in track.measurement_history if m["target_id"] >= 0]
    if not assigned:
        return None, 0
    counts = Counter(assigned)
    target_id, votes = counts.most_common(1)[0]
    return target_id, votes


def evaluate_tracks(all_tracks, gt_multi):
    representatives = {}
    all_summaries = []

    for track in all_tracks:
        if track.status not in {"confirmed", "coasting", "deleted"}:
            continue
        if not track.state_history:
            continue

        assigned_gt_id, votes = summarize_track_assignment(track)
        if assigned_gt_id is None or assigned_gt_id not in gt_multi:
            continue

        times = np.array([t for t, _ in track.state_history], dtype=float)
        states = np.array([x for _, x in track.state_history], dtype=float)
        gt_times, gt_states = gt_multi[assigned_gt_id]
        gt_interp = interpolate_gt(gt_times, gt_states, times)

        pos_err = states[:, :2] - gt_interp[:, :2]
        rmse_ne = np.sqrt(np.mean(pos_err**2, axis=0))
        rmse_total = float(np.linalg.norm(rmse_ne))

        summary = {
            "track_id": track.track_id,
            "gt_id": assigned_gt_id,
            "votes": votes,
            "rmse_ne": rmse_ne,
            "rmse_total": rmse_total,
            "num_states": len(times),
            "status": track.status,
        }
        all_summaries.append(summary)

        best = representatives.get(assigned_gt_id)
        if best is None or rmse_total < best["rmse_total"]:
            representatives[assigned_gt_id] = summary

    print("Per-target confirmed-track RMSE:")
    if not representatives:
        print("  No confirmed tracks could be matched to ground truth targets.")
        return {"representatives": {}, "all_summaries": all_summaries, "avg_total_rmse": None}

    total_rmses = []
    for gt_id in sorted(representatives):
        summary = representatives[gt_id]
        total_rmses.append(summary["rmse_total"])
        print(
            f"  GT {gt_id}: track {summary['track_id']} "
            f"RMSE [N, E]={summary['rmse_ne']} total={summary['rmse_total']:.2f} m "
            f"(votes={summary['votes']}, states={summary['num_states']})"
        )

    avg_total_rmse = float(np.mean(total_rmses))
    print(f"Average total RMSE across matched targets: {avg_total_rmse:.2f} m")

    return {
        "representatives": representatives,
        "all_summaries": all_summaries,
        "avg_total_rmse": avg_total_rmse,
    }


def active_true_positions(gt_multi, time_s, tol=0.5):
    positions = []
    ids = []

    for gt_id, (times, states) in gt_multi.items():
        if times[0] - tol <= time_s <= times[-1] + tol:
            north = float(np.interp(time_s, times, states[:, 0]))
            east = float(np.interp(time_s, times, states[:, 1]))
            positions.append([north, east])
            ids.append(gt_id)

    if not positions:
        return np.empty((0, 2), dtype=float), ids
    return np.array(positions, dtype=float), ids


def make_scan_record(time_s, tracks, gt_multi):
    confirmed = reportable_tracks(tracks)
    confirmed_positions = (
        np.array([track.ekf.x[:2].copy() for track in confirmed], dtype=float)
        if confirmed
        else np.empty((0, 2), dtype=float)
    )
    true_positions, true_ids = active_true_positions(gt_multi, time_s)

    return {
        "timestamp": float(time_s),
        "confirmed_positions": confirmed_positions,
        "confirmed_track_ids": [track.track_id for track in confirmed],
        "true_positions": true_positions,
        "true_ids": true_ids,
    }


def compute_motp_ce(scan_records):
    timestamps = []
    motp_series = []
    ce_series = []
    total_distance = 0.0
    total_matches = 0

    for record in scan_records:
        confirmed_positions = record["confirmed_positions"]
        true_positions = record["true_positions"]
        n_confirmed = confirmed_positions.shape[0]
        n_true = true_positions.shape[0]

        ce = float(abs(n_confirmed - n_true))
        if n_confirmed > 0 and n_true > 0:
            distances = np.linalg.norm(
                confirmed_positions[:, None, :] - true_positions[None, :, :],
                axis=2,
            )
            row_ind, col_ind = linear_sum_assignment(distances)
            matched_distances = distances[row_ind, col_ind]
            motp_scan = float(np.mean(matched_distances))
            total_distance += float(np.sum(matched_distances))
            total_matches += len(matched_distances)
        else:
            motp_scan = float("nan")

        timestamps.append(float(record["timestamp"]))
        motp_series.append(motp_scan)
        ce_series.append(ce)

    motp_avg = total_distance / total_matches if total_matches > 0 else float("nan")
    ce_avg = float(np.mean(ce_series)) if ce_series else float("nan")

    return {
        "timestamps": np.array(timestamps, dtype=float),
        "motp_series": np.array(motp_series, dtype=float),
        "ce_series": np.array(ce_series, dtype=float),
        "motp_avg": motp_avg,
        "ce_avg": ce_avg,
        "total_matches": total_matches,
    }


def print_motp_ce(metrics):
    print("Track-management metrics:")
    print(f"  MOTP avg: {metrics['motp_avg']:.2f} m")
    print(f"  CE avg: {metrics['ce_avg']:.2f}")
    print(f"  MOTP/CE matched pairs: {metrics['total_matches']}")


def merge_duplicate_tracks(tracks, time_s, threshold=5.991):
    reportable = reportable_tracks(tracks)
    delete_ids = set()

    for i in range(len(reportable)):
        for j in range(i + 1, len(reportable)):
            first = reportable[i]
            second = reportable[j]
            if first.track_id in delete_ids or second.track_id in delete_ids:
                continue

            dx = first.ekf.x[:2] - second.ekf.x[:2]
            S = first.ekf.P[:2, :2] + second.ekf.P[:2, :2]
            d2 = float(dx @ np.linalg.pinv(S) @ dx)
            if d2 <= threshold:
                delete_ids.add(max(first.track_id, second.track_id))

    for track in tracks:
        if track.track_id in delete_ids:
            track.status = "deleted"
            track.deleted_at = float(time_s)

    return len(delete_ids)


# EKF RUNNING FUNCTIONS

def run_ekf_radar_only(ts, zs_radar, x0):
    P0 = np.diag([25, 25, 2500, 2500])
    ekf = Target_EKF(x0, P0, dt=1.0)

    N = len(ts)
    x_est = np.zeros((N, 4))
    innov_hist, S_hist = [], []

    last_t = ts[0]
    for k in range(N):
        dt = ts[k] - last_t
        if dt > 0:
            ekf.predict(dt)
        z = zs_radar[k]
        innov, S = ekf.update_sensor(z, ["radar"])
        innov_hist.append(innov)
        S_hist.append(S)
        x_est[k] = ekf.x
        last_t = ts[k]

    return x_est, innov_hist, S_hist


def run_ekf_sequential(ts_radar, ts_camera, zs_radar, zs_camera, x0):
    P0 = np.diag([25, 25, 2500, 2500])
    ekf = Target_EKF(x0, P0, dt=None)

    # 1. Combine all measurements into one list and sort by timestamp
    # Format: (timestamp, sensor_type, measurement_value)
    measurements = []
    for t, z in zip(ts_radar, zs_radar):
        measurements.append((t, "radar", z))
    for t, z in zip(ts_camera, zs_camera):
        measurements.append((t, "camera", z))
    
    measurements.sort(key=lambda x: x[0])

    # Store timestamps for RMSE interpolation
    ts_est = np.array([m[0] for m in measurements])

    x_est = []
    innov_hist, S_hist = [], []
    last_t = measurements[0][0]

    count_radar = 0
    count_camera = 0

    for t, sensor_type, z in measurements:
        # 2. Predict step based on the actual time elapsed
        dt = t - last_t
        if dt > 0:
            ekf.predict(dt=dt)
        
        # 3. Update step for the specific sensor
        innov, S = ekf.update_sensor(z, [sensor_type])
        if sensor_type == "radar":
            count_radar += 1
        else:
            count_camera += 1
        
        # 4. Storage
        x_est.append(ekf.x.copy())
        innov_hist.append(innov)
        S_hist.append(S)
        last_t = t

    print(f"Processed {count_radar} radar measurements and {count_camera} camera measurements.")

    total_radar = len(ts_radar) if 'ts_radar' in locals() or 'ts_radar' in globals() else sum(1 for m in measurements if m[1]=='radar')
    total_camera = len(ts_camera) if 'ts_camera' in locals() or 'ts_camera' in globals() else sum(1 for m in measurements if m[1]=='camera')
    print(f"Sensor usage summary:")
    print(f"  Radar used {count_radar} times out of {total_radar} radar measurements")
    print(f"  Camera used {count_camera} times out of {total_camera} camera measurements")

    return np.array(x_est), np.array(ts_est), innov_hist, S_hist


def run_ekf_joint(ts_radar, ts_camera, zs_radar, zs_camera, x0):
    P0 = np.diag([25, 25, 2500, 2500])
    ekf = Target_EKF(x0, P0, dt=None)
    
    N = len(ts_radar)
    x_est = np.zeros((N, 4))
    innov_hist, S_hist = [], []
    
    cam_idx = 0
    last_t = ts_radar[0]

    count_radar = 0
    count_camera = 0

    for k in range(N):
        t_k = ts_radar[k]
        z_r = zs_radar[k]

        # 1. Process any camera measurements that happened BEFORE this radar scan
        # but AFTER the last radar scan (Sequential processing)
        while cam_idx < len(ts_camera) and ts_camera[cam_idx] < t_k:
            t_c = ts_camera[cam_idx]
            dt_c = t_c - last_t
            if dt_c > 0:
                ekf.predict(dt=dt_c)
            
            ekf.update_sensor(zs_camera[cam_idx], ["camera"])
            count_camera += 1
            last_t = t_c
            cam_idx += 1

        # 2. Predict up to the current Radar timestamp
        dt_r = t_k - last_t
        if dt_r > 0:
            ekf.predict(dt=dt_r)
        last_t = t_k

        # 3. Check if there is a camera measurement exactly at (or very close to) t_k
        # this is your "Closest in time" Joint Update
        if cam_idx < len(ts_camera) and abs(ts_camera[cam_idx] - t_k) < 1e-6:
            z_joint = np.hstack([z_r, zs_camera[cam_idx]])
            innov, S = ekf.update_sensor(z_joint, ["radar", "camera"])
            count_radar += 1
            count_camera += 1
            cam_idx += 1
        else:
            # Radar-only update
            innov, S = ekf.update_sensor(z_r, ["radar"])
            count_radar += 1

        innov_hist.append(innov)
        S_hist.append(S)
        x_est[k] = ekf.x.copy()

    print(f"Processed {count_radar} radar measurements and {count_camera} camera measurements.")

    total_radar = len(ts_radar)
    total_camera = len(ts_camera)
    print("Sensor usage summary:")
    print(f"  Radar used {count_radar} times out of {total_radar} radar measurements")
    print(f"  Camera used {count_camera} times out of {total_camera} camera measurements")

    return x_est, innov_hist, S_hist


def run_ekf_async_fusion(measurements, x0):
    P0 = np.diag([25.0, 25.0, 2500.0, 2500.0])
    ekf = Target_EKF(x0, P0, dt=1.0)

    x_est = []
    ts_est = []
    innov_hist, S_hist = [], []
    update_counts = Counter()
    rejected_counts = Counter()

    if not measurements:
        return np.empty((0, 4)), np.array([]), innov_hist, S_hist, update_counts, rejected_counts

    last_t = measurements[0]["time"]

    for m in measurements:
        sensor = m["sensor_id"]
        current_t = m["time"]
        dt = current_t - last_t

        if dt > 0:
            ekf.predict(dt)
            last_t = current_t

        z = measurement_vector(m)
        use_measurement = True
        if sensor == "camera":
            use_measurement = ekf.cfm.is_in_fov(ekf.x, "camera")

        if use_measurement:
            h = ekf.cfm.h(ekf.x, sensor)
            H = ekf.cfm.H(ekf.x, sensor)
            R = ekf.cfm.R(sensor, x=ekf.x)
            use_measurement, _ = ekf.compute_gating_distance(z, h, H, R, sensor)

        if use_measurement:
            innov, S = ekf.update_sensor(z, [sensor])
            innov_hist.append(innov)
            S_hist.append(S)
            update_counts[sensor] += 1
        else:
            rejected_counts[sensor] += 1

        x_est.append(ekf.x.copy())
        ts_est.append(current_t)

    print(
        "Accepted updates: "
        f"radar={update_counts['radar']}, "
        f"camera={update_counts['camera']}, "
        f"ais={update_counts['ais']}"
    )
    print(
        "Rejected updates: "
        f"radar={rejected_counts['radar']}, "
        f"camera={rejected_counts['camera']}, "
        f"ais={rejected_counts['ais']}"
    )

    return np.array(x_est), np.array(ts_est), innov_hist, S_hist, update_counts, rejected_counts


def format_sensor_counter(counter, sensor_ids):
    return ", ".join(f"{sensor_id}={counter[sensor_id]}" for sensor_id in sensor_ids)


def run_multitarget_tracking(
    scans,
    assoc_method="gnn",
    gate_probability=0.99,
    gt_multi=None,
    initiation_sensors=("radar", "ais"),
    lifecycle_sensor_ids=("radar", "ais"),
    confirmation_hits=3,
    confirmation_window=5,
    tentative_delete_misses=1,
    delete_misses=5,
    coast_gate_growth=2.0,
    merge_threshold=5.991,
):
    tracks = []
    next_track_id = 0
    innov_hist, S_hist = [], []
    cfm = CoordinateFrameManager()
    scan_records = []
    target_sensor_ids = ("radar", "camera", "ais")
    stats = {
        "matched_updates": Counter(),
        "unmatched_detections": Counter(),
        "total_detections": Counter(),
        "false_alarms_presented": Counter(),
        "new_tracks": 0,
        "new_tracks_by_sensor": Counter(),
        "confirmed_tracks": 0,
        "deleted_tracks": 0,
        "merged_tracks": 0,
        "sensor_available_scans": Counter(),
        "scan_records": scan_records,
    }

    if not scans:
        return tracks, innov_hist, S_hist, stats

    last_time = float(scans[0]["time"])

    for scan in scans:
        time_s = float(scan["time"])
        for gnss in scan["measurements_by_sensor"].get("gnss", []):
            cfm.update_gnss(
                np.array([gnss["north_m"], gnss["east_m"]], dtype=float),
                time_s,
            )

        detections = flatten_scan_detections(scan)
        association_sensor_available = {
            sensor_id: available
            for sensor_id, available in scan["sensor_available"].items()
            if sensor_id in target_sensor_ids
        }

        for sensor_id, measurements in scan["measurements_by_sensor"].items():
            if scan["sensor_available"].get(sensor_id, False):
                stats["sensor_available_scans"][sensor_id] += 1
            stats["total_detections"][sensor_id] += len(measurements)
            stats["false_alarms_presented"][sensor_id] += sum(
                1 for m in measurements if m["is_false_alarm"]
            )

        dt = time_s - last_time
        if dt > 0:
            for track in active_tracks(tracks):
                track.ekf.predict(dt)
            last_time = time_s

        active = active_tracks(tracks)
        for track in active:
            track.ekf.gate_extra = coast_gate_growth * track.coast_scans if track.status == "coasting" else 0.0

        if active and detections:
            matches_local, unmatched_slot_indices, unmatched_dets, _, slots = associate_multisensor_slots(
                tracks=[track.ekf for track in active],
                detections=detections,
                sensor_available=association_sensor_available,
                method=assoc_method,
                timestamp_s=time_s,
                gate_probability=gate_probability,
            )
        else:
            matches_local = []
            slots = []
            unmatched_slot_indices = []
            unmatched_dets = list(range(len(detections)))

        matched_track_ids = set()
        matched_detection_indices_by_sensor = defaultdict(set)

        for slot_idx, det_idx in matches_local:
            slot = slots[slot_idx]
            track = active[slot["track_idx"]]
            det = detections[det_idx]
            sensor_id = det["sensor_id"]
            measurement = det["measurement"]
            maybe_initialize_track_velocity(track, measurement, sensor_id, time_s)
            innov, S = track.ekf.update_sensor(det["z"], [sensor_id])
            innov_hist.append(innov)
            S_hist.append(S)
            track.last_update_time = time_s
            track.update_counts[sensor_id] += 1
            append_track_state(track, time_s)
            append_track_measurement(track, measurement)
            stats["matched_updates"][sensor_id] += 1
            matched_track_ids.add(track.track_id)
            matched_detection_indices_by_sensor[sensor_id].add(det["sensor_det_idx"])
            if sensor_id in lifecycle_sensor_ids:
                track.lifecycle_sensors.add(sensor_id)

        for slot_idx in unmatched_slot_indices:
            slot = slots[slot_idx]
            track = active[slot["track_idx"]]
            track.missed_detection_flags[slot["sensor_id"]] += 1

        for track in active:
            status_before = track.status
            got_hit = track.track_id in matched_track_ids
            miss_opportunity = any(
                association_sensor_available.get(sensor_id, False)
                for sensor_id in track.lifecycle_sensors
            )
            if (
                not miss_opportunity
                and association_sensor_available.get("camera", False)
                and track.ekf.cfm.is_in_fov(track.ekf.x, "camera", timestamp_s=time_s)
            ):
                miss_opportunity = True

            update_track_lifecycle(
                track,
                got_hit,
                time_s,
                miss_opportunity=miss_opportunity,
                confirmation_hits=confirmation_hits,
                confirmation_window=confirmation_window,
                tentative_delete_misses=tentative_delete_misses,
                delete_misses=delete_misses,
            )
            if status_before == "tentative" and track.status == "confirmed":
                stats["confirmed_tracks"] += 1
            if status_before != "deleted" and track.status == "deleted":
                stats["deleted_tracks"] += 1

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            sensor_id = det["sensor_id"]
            stats["unmatched_detections"][sensor_id] += 1

            if sensor_id not in initiation_sensors:
                continue

            if det["sensor_det_idx"] in matched_detection_indices_by_sensor[sensor_id]:
                continue

            new_track = initiate_track_from_measurement(
                next_track_id,
                det["measurement"],
                sensor_id,
                cfm=cfm,
            )
            tracks.append(new_track)
            next_track_id += 1
            stats["new_tracks"] += 1
            stats["new_tracks_by_sensor"][sensor_id] += 1

        stats["merged_tracks"] += merge_duplicate_tracks(
            tracks,
            time_s,
            threshold=merge_threshold,
        )

        if gt_multi is not None:
            scan_records.append(make_scan_record(time_s, tracks, gt_multi))

    for track in tracks:
        if track.status in {"confirmed", "coasting"} and not track.state_history:
            append_track_state(track, track.last_update_time)

    confirmed_alive = sum(1 for track in tracks if track.status == "confirmed")
    coasting_alive = sum(1 for track in tracks if track.status == "coasting")
    tentative_alive = sum(1 for track in tracks if track.status == "tentative")
    sensor_ids_print = [
        sensor_id for sensor_id in target_sensor_ids
        if (
            stats["total_detections"][sensor_id]
            or stats["matched_updates"][sensor_id]
            or stats["sensor_available_scans"][sensor_id]
        )
    ]
    print(f"Association method: {assoc_method.upper()}")
    print("Matched updates: " + format_sensor_counter(stats["matched_updates"], sensor_ids_print))
    print("Unmatched detections: " + format_sensor_counter(stats["unmatched_detections"], sensor_ids_print))
    print("False alarms presented: " + format_sensor_counter(stats["false_alarms_presented"], sensor_ids_print))
    print("Sensor available scans: " + format_sensor_counter(stats["sensor_available_scans"], sensor_ids_print))
    print(f"Confirmed tracks alive: {confirmed_alive}")
    print(f"Coasting tracks alive: {coasting_alive}")
    print(f"Confirmed track promotions: {stats['confirmed_tracks']}")
    print(
        f"Tracks created={stats['new_tracks']}, "
        f"created by sensor=({format_sensor_counter(stats['new_tracks_by_sensor'], sensor_ids_print)}), "
        f"tentative now={tentative_alive}, "
        f"deleted={stats['deleted_tracks']}, "
        f"merged={stats['merged_tracks']}"
    )

    if gt_multi is not None:
        stats["motp_ce"] = compute_motp_ce(scan_records)
        print_motp_ce(stats["motp_ce"])

    return tracks, innov_hist, S_hist, stats


# MAIN SIMULATION FUNCTION

def sim_tracking(json_file, scenario="A", mode="radar", assoc_method="gnn"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_times, gt_states = extract_ground_truth(data)
    print(f"Simulating Scenario {scenario} with mode {mode}")

    match scenario:
        case "A":
            meas = [
                (m["time"], m["range_m"], m["bearing_rad"])
                for m in data["measurements"]
                if m["sensor_id"] == "radar" and not m["is_false_alarm"]
            ]

            meas = np.array(meas)
            ts = meas[:, 0]
            zs_radar = meas[:, 1:]
            x_est, innov_hist, S_hist = run_ekf_radar_only(
                ts,
                zs_radar,
                x0=np.array([800, 600, -1, -2]),
            )
            gt_interp = interpolate_gt(gt_times, gt_states, ts)
            return x_est, gt_interp, innov_hist, S_hist, meas

        case "B":
            radar = [
                (m["time"], m["range_m"], m["bearing_rad"])
                for m in data["measurements"]
                if m["sensor_id"] == "radar" and not m["is_false_alarm"]
            ]
            camera = [
                (m["time"], m["range_m"], m["bearing_rad"])
                for m in data["measurements"]
                if m["sensor_id"] == "camera" and not m["is_false_alarm"]
            ]

            radar = np.array(radar)
            camera = np.array(camera)
            ts_radar = radar[:, 0]
            zs_radar = radar[:, 1:]
            ts_camera = camera[:, 0]
            zs_camera = camera[:, 1:]

            if mode == "joint":
                x_est, innov_hist, S_hist = run_ekf_joint(
                    ts_radar,
                    ts_camera,
                    zs_radar,
                    zs_camera,
                    x0=np.array([400, 80, 1.2, 2.2]),
                )
                ts_est = ts_radar
            elif mode == "radar":
                x_est, innov_hist, S_hist = run_ekf_radar_only(
                    ts_radar,
                    zs_radar,
                    x0=np.array([400, 80, 1.2, 2.2]),
                )
                ts_est = ts_radar
            else:
                x_est, ts_est, innov_hist, S_hist = run_ekf_sequential(
                    ts_radar,
                    ts_camera,
                    zs_radar,
                    zs_camera,
                    x0=np.array([400, 80, 1.2, 2.2]),
                )

            gt_interp = interpolate_gt(gt_times, gt_states, ts_est)
            return x_est, gt_interp, innov_hist, S_hist, radar, camera

        case "C":
            if mode == "sequential":
                valid_sensors = {"radar", "camera"}
            elif mode == "ais":
                valid_sensors = {"radar", "camera", "ais"}
            else:
                raise ValueError(
                    "Scenario C supports mode='sequential' for radar+camera "
                    "or mode='ais' for radar+camera+AIS."
                )

            measurements = [
                m for m in data["measurements"]
                if (
                    m["sensor_id"] in valid_sensors
                    and not m["is_false_alarm"]
                    and m.get("target_id", 0) == 0
                )
            ]
            measurements.sort(key=lambda m: m["time"])

            x0 = np.array(gt_states[0], dtype=float)
            x_est, ts_est, innov_hist, S_hist, _, _ = run_ekf_async_fusion(
                measurements,
                x0=x0,
            )
            gt_interp = interpolate_gt(gt_times, gt_states, ts_est)
            radar = measurement_array(measurements, "radar")
            camera = measurement_array(measurements, "camera")
            ais = measurement_array(measurements, "ais")
            return x_est, gt_interp, innov_hist, S_hist, radar, camera, ais, ts_est

        case "D":
            measurements = [
                m for m in data["measurements"]
                if m["sensor_id"] in {"radar", "camera"}
            ]
            scans = group_measurements_by_scan(measurements)
            gt_multi = extract_ground_truth_multi(data)
            tracks, innov_hist, S_hist, stats = run_multitarget_tracking(
                scans,
                assoc_method=assoc_method,
                gt_multi=gt_multi,
                initiation_sensors=("radar",),
                lifecycle_sensor_ids=("radar",),
            )
            evaluation = evaluate_tracks(tracks, gt_multi)
            return {
                "tracks": tracks,
                "ground_truth": gt_multi,
                "measurements": measurements,
                "innov": innov_hist,
                "S": S_hist,
                "stats": stats,
                "evaluation": evaluation,
            }

        case "E":
            measurements = [
                m for m in data["measurements"]
                if m["sensor_id"] in {"radar", "camera", "ais", "gnss"}
            ]
            scans = group_measurements_by_scan(
                measurements,
                sensor_ids=("radar", "camera", "ais", "gnss"),
            )
            gt_multi = extract_ground_truth_multi(data)
            tracks, innov_hist, S_hist, stats = run_multitarget_tracking(
                scans,
                assoc_method=assoc_method,
                gt_multi=gt_multi,
                initiation_sensors=("radar", "ais"),
                lifecycle_sensor_ids=("radar", "ais"),
            )
            evaluation = evaluate_tracks(tracks, gt_multi)
            return {
                "tracks": tracks,
                "ground_truth": gt_multi,
                "measurements": measurements,
                "innov": innov_hist,
                "S": S_hist,
                "stats": stats,
                "evaluation": evaluation,
            }

    raise ValueError(f"Unsupported scenario: {scenario}")


# PLOTTING

def plot_trajectories(gt, results, labels, measurements=None):
    plt.figure(figsize=(8, 6))
    plt.plot(gt[:, 0], gt[:, 1], "k--", linewidth=2, label="Ground Truth")

    for x_est, label in zip(results, labels):
        plt.plot(x_est[:, 0], x_est[:, 1], linewidth=1.5, label=label)

    sensor_origins = {
        "Radar": np.array([0.0, 0.0]),
        "Camera": np.array([-80.0, 120.0]),
    }

    if measurements is not None:
        if isinstance(measurements, list):
            for meas, meas_label in measurements:
                if len(meas) == 0:
                    continue
                if meas_label == "AIS":
                    mN, mE = meas[:, 1], meas[:, 2]
                else:
                    origin = sensor_origins.get(meas_label, np.array([0.0, 0.0]))
                    r, phi = meas[:, 1], meas[:, 2]
                    mN = origin[0] + r * np.cos(phi)
                    mE = origin[1] + r * np.sin(phi)
                plt.scatter(mN, mE, alpha=0.5, s=20, label=meas_label)
        else:
            r, phi = measurements[:, 1], measurements[:, 2]
            plt.scatter(r * np.cos(phi), r * np.sin(phi), alpha=0.5, s=20, label="Measurements")

    plt.xlabel("North (m)")
    plt.ylabel("East (m)")
    plt.title("Target Tracking Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def measurement_positions_ned(measurements, sensor_id, include_false_alarms=False):
    cfm = CoordinateFrameManager()
    positions = []

    for measurement in measurements:
        if measurement["sensor_id"] != sensor_id:
            continue
        if measurement["is_false_alarm"] and not include_false_alarms:
            continue

        if sensor_id in {"radar", "camera"}:
            positions.append(polar_to_ned(measurement_vector(measurement), sensor_id, cfm))
        elif sensor_id == "ais":
            positions.append(np.array([measurement["north_m"], measurement["east_m"]], dtype=float))

    return np.array(positions, dtype=float) if positions else np.empty((0, 2), dtype=float)


def track_lifetime_s(track):
    if not track.state_history:
        return 0.0

    start_time = float(track.state_history[0][0])
    if track.status == "deleted" and track.deleted_at is not None:
        end_time = float(track.deleted_at)
    else:
        end_time = float(track.state_history[-1][0])

    return max(0.0, end_time - start_time)


def plot_multitarget_trajectories(
    gt_multi,
    tracks,
    measurements,
    scenario="D",
    show_ended_tracks=False,
    show_unassigned_tracks=False,
    ended_track_min_lifetime_s=None,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10.colors
    gt_color = {
        gt_id: colors[idx % len(colors)]
        for idx, gt_id in enumerate(sorted(gt_multi))
    }
    bounds_points = []

    for gt_id in sorted(gt_multi):
        _, gt_states = gt_multi[gt_id]
        ax.plot(
            gt_states[:, 0],
            gt_states[:, 1],
            linestyle="--",
            linewidth=1.8,
            color=gt_color[gt_id],
            label=f"GT {gt_id}",
        )
        bounds_points.append(gt_states[:, :2])

    sensor_styles = {
        "radar": {"s": 10, "alpha": 0.12, "color": "#4C78A8", "label": "Radar detections"},
        "camera": {"s": 10, "alpha": 0.18, "color": "#F58518", "label": "Camera detections"},
        "ais": {"s": 16, "alpha": 0.35, "color": "#54A24B", "label": "AIS reports"},
    }
    for sensor_id, style in sensor_styles.items():
        points = measurement_positions_ned(measurements, sensor_id, include_false_alarms=False)
        if len(points) == 0:
            continue
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=style["s"],
            alpha=style["alpha"],
            color=style["color"],
            label=style["label"],
            zorder=1,
        )
        bounds_points.append(points)

    for track in tracks:
        if not track.state_history:
            continue
        lifetime_s = track_lifetime_s(track)
        show_ended_by_lifetime = (
            track.status == "deleted"
            and ended_track_min_lifetime_s is not None
            and lifetime_s >= ended_track_min_lifetime_s
        )
        show_deleted_track = show_ended_tracks or show_ended_by_lifetime
        if track.status == "deleted" and not show_deleted_track:
            continue
        if not (
            track.was_confirmed
            or track.status in {"confirmed", "coasting"}
            or show_ended_by_lifetime
        ):
            continue

        states = np.array([x for _, x in track.state_history], dtype=float)
        assigned_gt_id, votes = summarize_track_assignment(track)
        if assigned_gt_id is None and not show_unassigned_tracks:
            continue

        color = gt_color.get(assigned_gt_id, "0.35")
        label = f"Track {track.track_id}"
        if assigned_gt_id is not None:
            label += f" -> GT {assigned_gt_id}"
        if track.status == "deleted" and show_deleted_track:
            label += f" (ended, {lifetime_s:.0f}s)"

        ax.plot(
            states[:, 0],
            states[:, 1],
            linewidth=2.3,
            color=color,
            alpha=0.95,
            label=label,
            zorder=3,
        )
        ax.scatter(states[0, 0], states[0, 1], color=color, marker="o", s=28, zorder=4)
        ax.scatter(states[-1, 0], states[-1, 1], color=color, marker="s", s=28, zorder=4)
        bounds_points.append(states[:, :2])

    if bounds_points:
        all_points = np.vstack(bounds_points)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        span = np.maximum(maxs - mins, 1.0)
        margin = np.maximum(span * 0.12, 50.0)
        ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
        ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_title(f"Scenario {scenario} Multi-Target Tracking")
    ax.legend(ncol=2, fontsize=8, loc="best")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


def plot_position_error(gt, x_est_list, labels):
    plt.figure(figsize=(8, 5))

    for x_est, label in zip(x_est_list, labels):
        err = np.linalg.norm(x_est[:, :2] - gt[:, :2], axis=1)
        plt.plot(err, label=label)

    plt.xlabel("Time step")
    plt.ylabel("Position error (m)")
    plt.title("Tracking Error Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_nis(nis, label):
    nz = 2
    try:
        from scipy.stats import chi2

        lower = chi2.ppf(0.025, df=nz)
        upper = chi2.ppf(0.975, df=nz)
    except ModuleNotFoundError:
        lower = 0.0506
        upper = 7.3778

    plt.figure(figsize=(8, 4))
    plt.plot(nis, label="NIS")
    plt.axhline(lower, linestyle="--", color="red", label="95% bounds")
    plt.axhline(upper, linestyle="--", color="red")
    plt.xlabel("Time step")
    plt.ylabel("NIS")
    plt.title(f"NIS Consistency: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_track_management_metrics(metrics, scenario):
    ts = metrics["timestamps"]
    motp = metrics["motp_series"]
    ce = metrics["ce_series"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Scenario {scenario} Track Management Metrics")

    axes[0].plot(ts, motp, linewidth=1.5, label="MOTP per scan")
    axes[0].axhline(
        metrics["motp_avg"],
        linestyle="--",
        linewidth=1.0,
        label=f"Mean MOTP = {metrics['motp_avg']:.2f} m",
    )
    axes[0].set_ylabel("MOTP (m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].step(ts, ce, where="post", linewidth=1.5, label="CE per scan")
    axes[1].axhline(
        metrics["ce_avg"],
        linestyle="--",
        linewidth=1.0,
        label=f"Mean CE = {metrics['ce_avg']:.2f}",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Cardinality error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_rmse_bar(rmse_values, labels):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, rmse_values)
    plt.ylabel("RMSE (m)")
    plt.title("Position RMSE Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def run_demo(
    scenario="C",
    mode="ais",
    assoc_method="gnn",
    show_plots=True,
    show_ended_tracks=False,
    ended_track_min_lifetime_s=None,
):
    json_file = f"harbour_sim_output/scenario_{scenario}.json"
    result = sim_tracking(json_file, scenario=scenario, mode=mode, assoc_method=assoc_method)

    if scenario == "A":
        x_est, gt, innov, S, meas = result
        measurements = [(meas, "Radar")]
        label = "Radar-only"
        if len(innov) > 0:
            nis = compute_nis(innov, S)
            if show_plots:
                plot_nis(nis, label)
        compute_rmse(x_est, gt)
        if show_plots:
            plot_trajectories(gt, [x_est], [label], measurements=measurements)
            plot_position_error(gt, [x_est], [label])
        return

    if scenario == "B":
        x_est, gt, innov, S, radar, camera = result
        measurements = [(radar, "Radar")]
        if mode != "radar":
            measurements.append((camera, "Camera"))
        label = {
            "radar": "Radar-only",
            "sequential": "Sequential Fusion",
            "joint": "Joint Fusion",
        }.get(mode, mode)
        if len(innov) > 0:
            nis = compute_nis(innov, S)
            if show_plots:
                plot_nis(nis, label)
        compute_rmse(x_est, gt)
        if show_plots:
            plot_trajectories(gt, [x_est], [label], measurements=measurements)
            plot_position_error(gt, [x_est], [label])
        return

    if scenario == "C":
        x_est, gt, innov, S, radar, camera, ais, _ = result
        measurements = [(radar, "Radar"), (camera, "Camera")]
        if mode == "ais":
            measurements.append((ais, "AIS"))
            label = "Radar + Camera + AIS"
        else:
            label = "Radar + Camera"
        if len(innov) > 0:
            nis = compute_nis(innov, S)
            if show_plots:
                plot_nis(nis, label)
        compute_rmse(x_est, gt)
        if show_plots:
            plot_trajectories(gt, [x_est], [label], measurements=measurements)
            plot_position_error(gt, [x_est], [label])
        return

    if scenario in {"D", "E"}:
        if show_plots:
            plot_multitarget_trajectories(
                result["ground_truth"],
                result["tracks"],
                result["measurements"],
                scenario=scenario,
                show_ended_tracks=show_ended_tracks,
                ended_track_min_lifetime_s=ended_track_min_lifetime_s,
            )
            if "motp_ce" in result["stats"]:
                plot_track_management_metrics(result["stats"]["motp_ce"], scenario)
        return

    raise ValueError(f"Unsupported scenario: {scenario}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run harbour surveillance EKF simulations.")
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D", "E"],
        default="C",
        help="Scenario to run. Scenarios D/E are multi-target T6/T7 validation.",
    )
    parser.add_argument(
        "--mode",
        choices=["radar", "sequential", "joint", "ais"],
        default="ais",
        help="Fusion mode. D ignores mode and uses multi-target radar+camera.",
    )
    parser.add_argument(
        "--assoc",
        choices=["gnn", "nn"],
        default="gnn",
        help="Association method for Scenarios D/E.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run metrics without opening Matplotlib figures.",
    )
    parser.add_argument(
        "--show-ended-tracks",
        action="store_true",
        help="In Scenario D/E plots, include all deleted tracks that were confirmed.",
    )
    parser.add_argument(
        "--show-ended-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "In Scenario D/E plots, include deleted tracks whose lifetime is at "
            "least this many seconds."
        ),
    )
    args = parser.parse_args()

    run_demo(
        scenario=args.scenario,
        mode=args.mode,
        assoc_method=args.assoc,
        show_plots=not args.no_plots,
        show_ended_tracks=args.show_ended_tracks,
        ended_track_min_lifetime_s=args.show_ended_after,
    )
