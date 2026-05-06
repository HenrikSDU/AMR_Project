from dataclasses import dataclass, field
from typing import Optional

import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from Coordinate_Frame_Manager import CoordinateFrameManager
from Target_EKF import Target_EKF
from data_association import associate, associate_multisensor_slots

import time


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
    radar_scan_times: list[float] = field(default_factory=list)
    deleted_at: Optional[float] = None


@dataclass
class BirthCandidate:
    candidate_id: int
    first_seen_time: float
    last_seen_time: float
    pos_ned: np.ndarray
    measurement: dict
    hits: int = 1
    misses: int = 0
    radar_scan_times: list[float] = field(default_factory=list)


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
    if measurement["sensor_id"] == "radar":
        time_s = float(measurement["time"])
        if not track.radar_scan_times or not np.isclose(track.radar_scan_times[-1], time_s):
            track.radar_scan_times.append(time_s)


def flatten_scan_detections(scan):
    detections = []

    for sensor_id, measurements in scan["measurements_by_sensor"].items():
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


def initiate_track_from_radar(track_id, measurement, init_dt=1.0):
    cfm = CoordinateFrameManager()
    z = measurement_vector(measurement)
    pos = polar_to_ned(z, "radar", cfm)
    x0 = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
    P0 = np.diag([100.0, 100.0, 10000.0, 10000.0])
    ekf = Target_EKF(x0, P0, dt=init_dt)

    track = ManagedTrack(
        track_id=track_id,
        ekf=ekf,
        status="tentative",
        last_update_time=float(measurement["time"]),
        hits=1,
        misses=0,
    )
    track.hit_history.append(True)
    track.update_counts["radar"] += 1
    track.radar_scan_times.append(float(measurement["time"]))
    append_track_state(track, measurement["time"])
    append_track_measurement(track, measurement)
    return track


def initiate_track_from_candidate(track_id, candidate, init_dt=1.0):
    track = initiate_track_from_radar(track_id, candidate.measurement, init_dt=init_dt)
    track.hits = max(track.hits, candidate.hits)
    track.hit_history = [True] * min(candidate.hits, 4)
    track.radar_scan_times = list(candidate.radar_scan_times)
    track.last_update_time = float(candidate.last_seen_time)
    return track


def detection_near_active_track(pos_ned, tracks, distance_gate=60.0):
    for track in active_tracks(tracks):
        if np.linalg.norm(track.ekf.x[:2] - pos_ned) <= distance_gate:
            return True
    return False


def track_position(track):
    return np.asarray(track.ekf.x[:2], dtype=float)


def candidate_priority(candidate, reinforcing_distance):
    return (
        int(candidate.hits),
        -float(reinforcing_distance),
    )


def track_strength(track):
    return (
        1 if track.status == "confirmed" else 0,
        int(track.hits),
        len(track.state_history),
        -float(np.trace(track.ekf.P[:2, :2])),
    )


def track_near_stronger_track(track, tracks, distance_gate=50.0):
    for other in active_tracks(tracks):
        if other.track_id == track.track_id:
            continue
        if np.linalg.norm(track_position(track) - track_position(other)) > distance_gate:
            continue
        if track_strength(other) >= track_strength(track):
            return True, other.track_id
    return False, None


def should_confirm_track(
    track,
    got_radar_hit,
    *,
    min_total_radar_hits=4,
    min_recent_hits=3,
    recent_window=4,
    min_radar_scan_span=3,
    max_pos_cov_trace_for_confirmation=400.0,
):
    if track.status != "tentative" or not got_radar_hit:
        return False
    if sum(track.hit_history[-recent_window:]) < min_recent_hits:
        return False
    if int(track.hits) < min_total_radar_hits:
        return False
    if len(track.radar_scan_times) < min_radar_scan_span:
        return False
    pos_cov_trace = float(np.trace(track.ekf.P[:2, :2]))
    if pos_cov_trace > max_pos_cov_trace_for_confirmation:
        return False
    return True


def update_birth_candidates(
    birth_candidates,
    radar_measurements,
    tracks,
    next_track_id,
    next_candidate_id,
    time_s,
    candidate_gate=45.0,
    duplicate_gate=60.0,
    promotion_duplicate_gate=80.0,
):
    promoted_tracks = []
    stats = Counter()

    detections = []
    cfm = CoordinateFrameManager()
    for measurement in radar_measurements:
        z = measurement_vector(measurement)
        detections.append({
            "measurement": measurement,
            "pos_ned": polar_to_ned(z, "radar", cfm),
        })

    candidate_matches = {}
    used_detection_indices = set()
    scored_pairs = []

    for cand_idx, candidate in enumerate(birth_candidates):
        for det_idx, det in enumerate(detections):
            distance = np.linalg.norm(candidate.pos_ned - det["pos_ned"])
            if distance <= candidate_gate:
                scored_pairs.append((distance, cand_idx, det_idx))

    scored_pairs.sort(key=lambda item: item[0])
    used_candidates = set()
    for _, cand_idx, det_idx in scored_pairs:
        if cand_idx in used_candidates or det_idx in used_detection_indices:
            continue
        candidate_matches[cand_idx] = det_idx
        used_candidates.add(cand_idx)
        used_detection_indices.add(det_idx)

    survivors = []
    promotable = []
    for cand_idx, candidate in enumerate(birth_candidates):
        if cand_idx in candidate_matches:
            det = detections[candidate_matches[cand_idx]]
            candidate.pos_ned = det["pos_ned"]
            candidate.measurement = det["measurement"]
            candidate.last_seen_time = float(time_s)
            candidate.hits += 1
            candidate.misses = 0
            if (
                not candidate.radar_scan_times
                or not np.isclose(candidate.radar_scan_times[-1], float(time_s))
            ):
                candidate.radar_scan_times.append(float(time_s))

            if candidate.hits >= 2 and len(candidate.radar_scan_times) >= 2:
                if detection_near_active_track(
                    candidate.pos_ned,
                    tracks,
                    distance_gate=promotion_duplicate_gate,
                ):
                    stats["suppressed_promotions"] += 1
                else:
                    reinforcing_distance = np.linalg.norm(candidate.pos_ned - det["pos_ned"])
                    promotable.append((candidate_priority(candidate, reinforcing_distance), candidate))
            else:
                survivors.append(candidate)
        else:
            candidate.misses += 1
            if candidate.misses < 2:
                survivors.append(candidate)
            else:
                stats["expired_candidates"] += 1

    promotable.sort(key=lambda item: item[0], reverse=True)
    selected_promotions = []
    for _, candidate in promotable:
        if any(
            np.linalg.norm(candidate.pos_ned - other.pos_ned) <= promotion_duplicate_gate
            for other in selected_promotions
        ):
            stats["suppressed_same_scan_promotions"] += 1
            continue
        selected_promotions.append(candidate)

    for candidate in selected_promotions:
        promoted_tracks.append(initiate_track_from_candidate(next_track_id, candidate))
        next_track_id += 1
        stats["promoted_candidates"] += 1

    birth_candidates = survivors

    for det_idx, det in enumerate(detections):
        if det_idx in used_detection_indices:
            continue

        pos_ned = det["pos_ned"]
        if detection_near_active_track(pos_ned, tracks, distance_gate=duplicate_gate):
            stats["suppressed_births"] += 1
            continue

        if any(np.linalg.norm(candidate.pos_ned - pos_ned) <= duplicate_gate for candidate in birth_candidates):
            stats["suppressed_births"] += 1
            continue

        birth_candidates.append(
            BirthCandidate(
                candidate_id=next_candidate_id,
                first_seen_time=float(time_s),
                last_seen_time=float(time_s),
                pos_ned=pos_ned,
                measurement=det["measurement"],
                hits=1,
                misses=0,
                radar_scan_times=[float(time_s)],
            )
        )
        next_candidate_id += 1
        stats["new_candidates"] += 1

    return birth_candidates, promoted_tracks, next_track_id, next_candidate_id, stats


def update_track_lifecycle(track, got_hit, time_s):
    if track.status == "deleted":
        return

    track.hit_history.append(bool(got_hit))
    if len(track.hit_history) > 4:
        track.hit_history = track.hit_history[-4:]

    if got_hit:
        track.hits += 1
        track.misses = 0
    else:
        track.misses += 1
        if track.status == "tentative" and track.misses >= 1:
            track.status = "deleted"
            track.deleted_at = float(time_s)
        elif track.status == "confirmed" and track.misses >= 3:
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
    update_track_lifecycle(track, got_radar_hit, time_s)


def maybe_demote_duplicate_confirmation(track, tracks):
    is_duplicate, _ = track_near_stronger_track(track, tracks)
    if track.status == "confirmed" and is_duplicate:
        track.status = "deleted"
        return True
    return False


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
    confirmed_alive_summaries = []

    for track in all_tracks:
        if track.status not in {"confirmed", "deleted"}:
            continue
        if not track.state_history:
            continue

        assigned_gt_id, votes = summarize_track_assignment(track)
        times = np.array([t for t, _ in track.state_history], dtype=float)
        states = np.array([x for _, x in track.state_history], dtype=float)
        rmse_ne = None
        rmse_total = None
        if assigned_gt_id is not None and assigned_gt_id in gt_multi:
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
            "current_pos": states[-1, :2],
            "radar_updates": int(track.update_counts["radar"]),
            "camera_updates": int(track.update_counts["camera"]),
        }
        all_summaries.append(summary)
        if track.status == "confirmed":
            confirmed_alive_summaries.append(summary)

        if assigned_gt_id is not None and assigned_gt_id in gt_multi:
            best = representatives.get(assigned_gt_id)
            if best is None or rmse_total < best["rmse_total"]:
                representatives[assigned_gt_id] = summary

    print("Confirmed tracks alive (full debug):")
    if not confirmed_alive_summaries:
        print("  None")
    else:
        for summary in sorted(confirmed_alive_summaries, key=lambda item: item["track_id"]):
            pos = summary["current_pos"]
            gt_label = "None" if summary["gt_id"] is None else str(summary["gt_id"])
            print(
                f"  track {summary['track_id']}: "
                f"GT={gt_label} votes={summary['votes']} states={summary['num_states']} "
                f"pos=[{pos[0]:.1f}, {pos[1]:.1f}] "
                f"radar_updates={summary['radar_updates']} "
                f"camera_updates={summary['camera_updates']}"
            )

        grouped = defaultdict(list)
        for summary in confirmed_alive_summaries:
            if summary["gt_id"] is not None:
                grouped[summary["gt_id"]].append(summary["track_id"])
        for gt_id in sorted(grouped):
            if len(grouped[gt_id]) > 1:
                print(f"Duplicate confirmed tracks for GT {gt_id}: {grouped[gt_id]}")

    confirmed_alive_total = len(confirmed_alive_summaries)
    confirmed_alive_with_gt = sum(1 for summary in confirmed_alive_summaries if summary["gt_id"] is not None)
    confirmed_alive_without_gt = confirmed_alive_total - confirmed_alive_with_gt
    print(
        "Confirmed alive summary: "
        f"total={confirmed_alive_total}, "
        f"with_gt={confirmed_alive_with_gt}, "
        f"without_gt={confirmed_alive_without_gt}"
    )

    print("Per-target confirmed-track RMSE:")
    if not representatives:
        print("  No confirmed tracks could be matched to ground truth targets.")
        return {
            "representatives": {},
            "all_summaries": all_summaries,
            "confirmed_alive_summaries": confirmed_alive_summaries,
            "confirmed_alive_total": confirmed_alive_total,
            "confirmed_alive_with_gt": confirmed_alive_with_gt,
            "confirmed_alive_without_gt": confirmed_alive_without_gt,
            "avg_total_rmse": None,
        }

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
        "confirmed_alive_summaries": confirmed_alive_summaries,
        "confirmed_alive_total": confirmed_alive_total,
        "confirmed_alive_with_gt": confirmed_alive_with_gt,
        "confirmed_alive_without_gt": confirmed_alive_without_gt,
        "avg_total_rmse": avg_total_rmse,
    }


# EKF RUNNING FUNCTIONS

def run_ekf_radar_only(ts, zs_radar, x0):
    dt = np.mean(np.diff(ts))
    P0 = np.diag([25, 25, 2500, 2500])
    ekf = Target_EKF(x0, P0, dt=dt)

    N = len(ts)
    x_est = np.zeros((N, 4))
    innov_hist, S_hist = [], []

    for k in range(N):
        ekf.predict()
        z = zs_radar[k]
        innov, S = ekf.update_sensor(z, ["radar"])
        innov_hist.append(innov)
        S_hist.append(S)
        x_est[k] = ekf.x

    return x_est, innov_hist, S_hist


def run_ekf_sequential(ts_radar, ts_camera, zs_radar, zs_camera, x0):
    dt = np.mean(np.diff(ts_radar))
    P0 = np.diag([0.1, 0.1, 0.1, 0.1])
    ekf = Target_EKF(x0, P0, dt=dt)

    N = len(ts_radar)
    x_est = np.zeros((N, 4))
    innov_hist, S_hist = [], []
    cam_idx = 0

    for k in range(N):
        ekf.predict()
        t_k = ts_radar[k]

        zr = zs_radar[k]
        innov, S = ekf.update_sensor(zr, ["radar"])
        innov_hist.append(innov)
        S_hist.append(S)

        while cam_idx < len(ts_camera) and ts_camera[cam_idx] <= t_k + 1e-6:
            zc = zs_camera[cam_idx]
            if ekf.cfm.is_in_fov(ekf.x, "camera"):
                h_c = ekf.cfm.h(ekf.x, "camera")
                H_c = ekf.cfm.H(ekf.x, "camera")
                R_c = ekf.cfm.R("camera", x=ekf.x)

                in_gate, _ = ekf.compute_gating_distance(zc, h_c, H_c, R_c, "camera", threshold=9.21)
                if in_gate:
                    innov, S = ekf.update_sensor(zc, ["camera"])
                    innov_hist.append(innov)
                    S_hist.append(S)

            cam_idx += 1

        x_est[k] = ekf.x

    return x_est, innov_hist, S_hist


def run_ekf_joint(ts_radar, ts_camera, zs_radar, zs_camera, x0):
    dt = np.mean(np.diff(ts_radar))
    camera_dt = np.mean(np.diff(ts_camera)) if len(ts_camera) > 1 else dt
    max_staleness = camera_dt / 2.0

    P0 = np.diag([25, 25, 2500, 2500])
    ekf = Target_EKF(x0, P0, dt=dt)
    N = len(ts_radar)

    x_est = np.zeros((N, 4))
    innov_hist, S_hist = [], []
    cam_idx = 0
    count_cam_used = 0

    for k in range(N):
        ekf.predict()
        t_k = ts_radar[k]
        z_r = zs_radar[k]

        cam_at_this_step = None
        cam_ts = None
        while cam_idx < len(ts_camera) and ts_camera[cam_idx] <= t_k + 1e-6:
            cam_at_this_step = zs_camera[cam_idx]
            cam_ts = ts_camera[cam_idx]
            cam_idx += 1

        use_camera = False
        if cam_at_this_step is not None:
            age = t_k - cam_ts
            if age <= max_staleness:
                z_c = cam_at_this_step
                if ekf.cfm.is_in_fov(ekf.x, "camera"):
                    h_c = ekf.cfm.h(ekf.x, "camera")
                    H_c = ekf.cfm.H(ekf.x, "camera")
                    R_c = ekf.cfm.R("camera", x=ekf.x)
                    in_gate, _ = ekf.compute_gating_distance(z_c, h_c, H_c, R_c, "camera")
                    if in_gate:
                        use_camera = True
                        count_cam_used += 1

        if use_camera:
            z = np.hstack([z_r, z_c])
            innov, S = ekf.update_sensor(z, ["radar", "camera"])
        else:
            innov, S = ekf.update_sensor(z_r, ["radar"])

        innov_hist.append(innov)
        S_hist.append(S)
        x_est[k] = ekf.x

    print(f"Camera fused on {count_cam_used} / {N} radar scans")
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
            ekf.predict_dt(dt)
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


def run_multitarget_tracking(scans, assoc_method="gnn", gate_probability=0.99):
    tracks = []
    next_track_id = 0
    birth_candidates = []
    next_candidate_id = 0
    innov_hist, S_hist = [], []
    stats = {
        "matched_updates": Counter(),
        "unmatched_detections": Counter(),
        "total_detections": Counter(),
        "false_alarms_presented": Counter(),
        "new_tracks": 0,
        "confirmed_tracks": 0,
        "deleted_tracks": 0,
        "sensor_available_scans": Counter(),
        "new_candidates": 0,
        "promoted_candidates": 0,
        "expired_candidates": 0,
        "suppressed_births": 0,
        "suppressed_promotions": 0,
        "suppressed_same_scan_promotions": 0,
        "suppressed_confirmations": 0,
        "rejected_confirmations_quality": 0,

        # Benchmark timing
        "num_scans": 0,
        "association_runtime_s": 0.0,
        "association_calls": 0,
    }

    if not scans:
        return tracks, innov_hist, S_hist, stats

    last_time = float(scans[0]["time"])

    for scan in scans:
        stats["num_scans"] += 1
        time_s = float(scan["time"])
        detections = flatten_scan_detections(scan)
        radar_available = scan["sensor_available"].get("radar", False)

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
                track.ekf.predict_dt(dt)
            last_time = time_s

        active = active_tracks(tracks)
        assoc_t0 = time.perf_counter()
        if active and detections:
            matches_local, unmatched_slot_indices, unmatched_dets, _, slots = associate_multisensor_slots(
                tracks=[track.ekf for track in active],
                detections=detections,
                sensor_available=scan["sensor_available"],
                method=assoc_method,
                timestamp_s=time_s,
                gate_probability=gate_probability,
            )
        else:
            matches_local = []
            slots = []
            unmatched_slot_indices = []
            unmatched_dets = list(range(len(detections)))

        stats["association_runtime_s"] += time.perf_counter() - assoc_t0
        stats["association_calls"] += 1

        radar_matched_track_ids = set()
        matched_radar_detection_indices = set()

        for slot_idx, det_idx in matches_local:
            slot = slots[slot_idx]
            track = active[slot["track_idx"]]
            det = detections[det_idx]
            sensor_id = det["sensor_id"]
            measurement = det["measurement"]
            innov, S = track.ekf.update_sensor(det["z"], [sensor_id])
            innov_hist.append(innov)
            S_hist.append(S)
            track.last_update_time = time_s
            track.update_counts[sensor_id] += 1
            append_track_state(track, time_s)
            append_track_measurement(track, measurement)
            stats["matched_updates"][sensor_id] += 1
            if sensor_id == "radar":
                radar_matched_track_ids.add(track.track_id)
                matched_radar_detection_indices.add(det["sensor_det_idx"])

        for slot_idx in unmatched_slot_indices:
            slot = slots[slot_idx]
            track = active[slot["track_idx"]]
            track.missed_detection_flags[slot["sensor_id"]] += 1

        for track in active:
            status_before = track.status
            got_radar_hit = track.track_id in radar_matched_track_ids
            update_track_lifecycle_radar_only(track, got_radar_hit, radar_available, time_s)
            if (
                status_before == "tentative"
                and track.status == "tentative"
                and should_confirm_track(track, got_radar_hit)
            ):
                track.status = "confirmed"
            elif (
                status_before == "tentative"
                and track.status == "tentative"
                and got_radar_hit
                and sum(track.hit_history[-4:]) >= 3
            ):
                stats["rejected_confirmations_quality"] += 1

            if status_before != "confirmed" and track.status == "confirmed":
                if maybe_demote_duplicate_confirmation(track, tracks):
                    track.deleted_at = float(time_s)
                    stats["deleted_tracks"] += 1
                    stats["suppressed_confirmations"] += 1
                    continue
            if status_before != "confirmed" and track.status == "confirmed":
                stats["confirmed_tracks"] += 1
            if status_before != "deleted" and track.status == "deleted":
                stats["deleted_tracks"] += 1

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            sensor_id = det["sensor_id"]
            stats["unmatched_detections"][sensor_id] += 1

        if radar_available:
            unmatched_radar_measurements = [
                det["measurement"]
                for det_idx, det in enumerate(detections)
                if det_idx in unmatched_dets and det["sensor_id"] == "radar"
            ]
            (
                birth_candidates,
                promoted_tracks,
                next_track_id,
                next_candidate_id,
                birth_stats,
            ) = update_birth_candidates(
                birth_candidates,
                unmatched_radar_measurements,
                tracks,
                next_track_id,
                next_candidate_id,
                time_s,
            )
            for key, value in birth_stats.items():
                stats[key] += value
            for track in promoted_tracks:
                tracks.append(track)
                stats["new_tracks"] += 1

    for track in tracks:
        if track.status == "confirmed" and not track.state_history:
            append_track_state(track, track.last_update_time)

    confirmed_alive = sum(1 for track in tracks if track.status == "confirmed")
    tentative_alive = sum(1 for track in tracks if track.status == "tentative")
    print(f"Association method: {assoc_method.upper()}")
    print(
        "Matched updates: "
        f"radar={stats['matched_updates']['radar']}, "
        f"camera={stats['matched_updates']['camera']}"
    )
    print(
        "Unmatched detections: "
        f"radar={stats['unmatched_detections']['radar']}, "
        f"camera={stats['unmatched_detections']['camera']}"
    )
    print(
        "False alarms presented: "
        f"radar={stats['false_alarms_presented']['radar']}, "
        f"camera={stats['false_alarms_presented']['camera']}"
    )
    print(
        "Sensor available scans: "
        f"radar={stats['sensor_available_scans']['radar']}, "
        f"camera={stats['sensor_available_scans']['camera']}"
    )
    print(f"Confirmed tracks alive: {confirmed_alive}")
    print(f"Confirmed track promotions: {stats['confirmed_tracks']}")
    print(
        f"Tracks created={stats['new_tracks']}, "
        f"tentative now={tentative_alive}, "
        f"deleted={stats['deleted_tracks']}"
    )
    print(
        f"Birth candidates active={len(birth_candidates)}, "
        f"new={stats['new_candidates']}, "
        f"promoted={stats['promoted_candidates']}, "
        f"expired={stats['expired_candidates']}, "
        f"suppressed_births={stats['suppressed_births']}, "
        f"suppressed_promotions={stats['suppressed_promotions']}, "
        f"suppressed_same_scan_promotions={stats['suppressed_same_scan_promotions']}, "
        f"suppressed_confirmations={stats['suppressed_confirmations']}, "
        f"rejected_confirmations_quality={stats['rejected_confirmations_quality']}"
    )

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
            elif mode == "radar":
                x_est, innov_hist, S_hist = run_ekf_radar_only(
                    ts_radar,
                    zs_radar,
                    x0=np.array([400, 80, 1.2, 2.2]),
                )
            else:
                x_est, innov_hist, S_hist = run_ekf_sequential(
                    ts_radar,
                    ts_camera,
                    zs_radar,
                    zs_camera,
                    x0=np.array([400, 80, 1.2, 2.2]),
                )

            gt_interp = interpolate_gt(gt_times, gt_states, ts_radar)
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


def plot_multitarget_trajectories(gt_multi, tracks, measurements):
    plt.figure(figsize=(9, 7))

    for gt_id in sorted(gt_multi):
        _, gt_states = gt_multi[gt_id]
        plt.plot(
            gt_states[:, 0],
            gt_states[:, 1],
            linestyle="--",
            linewidth=1.5,
            label=f"GT {gt_id}",
        )

    radar = measurement_array(measurements, "radar")
    camera = measurement_array(measurements, "camera")
    if len(radar) > 0:
        plt.scatter(
            radar[:, 1] * np.cos(radar[:, 2]),
            radar[:, 1] * np.sin(radar[:, 2]),
            s=10,
            alpha=0.15,
            label="Radar detections",
        )
    if len(camera) > 0:
        origin = np.array([-80.0, 120.0])
        plt.scatter(
            origin[0] + camera[:, 1] * np.cos(camera[:, 2]),
            origin[1] + camera[:, 1] * np.sin(camera[:, 2]),
            s=10,
            alpha=0.15,
            label="Camera detections",
        )

    for track in tracks:
        if not track.state_history:
            continue
        states = np.array([x for _, x in track.state_history], dtype=float)
        if track.status == "confirmed":
            plt.plot(states[:, 0], states[:, 1], linewidth=2.0, label=f"Track {track.track_id}")

    plt.xlabel("North (m)")
    plt.ylabel("East (m)")
    plt.title("Scenario D Multi-Target Tracking")
    plt.legend(ncol=2, fontsize=8)
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
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


def plot_rmse_bar(rmse_values, labels):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, rmse_values)
    plt.ylabel("RMSE (m)")
    plt.title("Position RMSE Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def run_demo(scenario="C", mode="ais", assoc_method="gnn", show_plots=True):
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

    if scenario == "D":
        if show_plots:
            plot_multitarget_trajectories(
                result["ground_truth"],
                result["tracks"],
                result["measurements"],
            )
        return

    raise ValueError(f"Unsupported scenario: {scenario}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run harbour surveillance EKF simulations.")
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D"],
        default="C",
        help="Scenario to run. Scenario D is multi-target radar+camera T6.",
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
        help="Association method for Scenario D.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run metrics without opening Matplotlib figures.",
    )
    args = parser.parse_args()

    run_demo(
        scenario=args.scenario,
        mode=args.mode,
        assoc_method=args.assoc,
        show_plots=not args.no_plots,
    )
