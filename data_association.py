import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


def gate_threshold_for_detection(z, gate_probability=0.99):
    """
    Chi-square gating threshold for a measurement vector z of dimension n_z.
    """
    return float(chi2.ppf(gate_probability, df=len(z)))


def build_candidates(
    tracks,
    detections,
    sensor_id,
    timestamp_s=None,
    gate_threshold=None,
    gate_probability=0.99,
):
    """
    Build gated candidate associations for one sensor.

    detections is a list of measurement vectors z.
    """
    candidates = []

    for i, track in enumerate(tracks):
        h_x = track.cfm.h(track.x, sensor_id=sensor_id, timestamp_s=timestamp_s)
        H = track.cfm.H(track.x, sensor_id=sensor_id, timestamp_s=timestamp_s)
        R = track.cfm.R(sensor_id=sensor_id, timestamp_s=timestamp_s)

        for j, z in enumerate(detections):
            threshold = gate_threshold
            if threshold is None:
                threshold = gate_threshold_for_detection(z, gate_probability=gate_probability)
            threshold += float(getattr(track, "gate_extra", 0.0))

            ok, d2 = track.compute_gating_distance(
                z=z,
                h_x=h_x,
                H=H,
                R=R,
                sensor_id=sensor_id,
                threshold=threshold,
            )

            if ok:
                candidates.append({
                    "track_idx": i,
                    "det_idx": j,
                    "cost": float(d2),
                })

    return candidates


def build_track_sensor_slots(
    tracks,
    sensor_available,
):
    """
    Create one assignment slot per active track per available sensor.
    """
    slots = []

    for track_idx, track in enumerate(tracks):
        for sensor_id, available in sensor_available.items():
            if not available:
                continue
            slots.append({
                "slot_idx": len(slots),
                "track_idx": track_idx,
                "sensor_id": sensor_id,
            })

    return slots


def build_multisensor_slot_candidates(
    tracks,
    slots,
    detections,
    timestamp_s=None,
    gate_probability=0.99,
):
    """
    Build gated candidates for simultaneous scan-level multi-sensor assignment.

    Rows are (track, sensor) slots. Columns are concrete detections.
    """
    candidates = []

    for slot in slots:
        track = tracks[slot["track_idx"]]
        sensor_id = slot["sensor_id"]

        if sensor_id == "camera" and not track.cfm.is_in_fov(
            track.x,
            "camera",
            timestamp_s=timestamp_s,
        ):
            continue

        h_x = track.cfm.h(track.x, sensor_id=sensor_id, timestamp_s=timestamp_s)
        H = track.cfm.H(track.x, sensor_id=sensor_id, timestamp_s=timestamp_s)
        R = track.cfm.R(sensor_id=sensor_id, timestamp_s=timestamp_s)

        for det_idx, det in enumerate(detections):
            if det["sensor_id"] != sensor_id:
                continue

            z = det["z"]
            threshold = gate_threshold_for_detection(z, gate_probability=gate_probability)
            threshold += float(getattr(track, "gate_extra", 0.0))

            ok, d2 = track.compute_gating_distance(
                z=z,
                h_x=h_x,
                H=H,
                R=R,
                sensor_id=sensor_id,
                threshold=threshold,
            )

            if ok:
                candidates.append({
                    "track_idx": slot["slot_idx"],
                    "det_idx": det_idx,
                    "cost": float(d2),
                    "slot_idx": slot["slot_idx"],
                    "sensor_id": sensor_id,
                    "real_track_idx": slot["track_idx"],
                })

    return candidates


def build_cost_matrix(candidates, n_tracks, n_dets, inf_cost=1e9):
    cost_matrix = np.full((n_tracks, n_dets), inf_cost, dtype=float)

    for c in candidates:
        i = c["track_idx"]
        j = c["det_idx"]
        cost_matrix[i, j] = c["cost"]

    return cost_matrix


def associate_nn(candidates, n_tracks, n_dets):
    matches = []
    unmatched_tracks = set(range(n_tracks))
    unmatched_dets = set(range(n_dets))

    candidates_sorted = sorted(candidates, key=lambda c: c["cost"])

    for c in candidates_sorted:
        i = c["track_idx"]
        j = c["det_idx"]

        if i in unmatched_tracks and j in unmatched_dets:
            matches.append((i, j))
            unmatched_tracks.remove(i)
            unmatched_dets.remove(j)

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


def associate_gnn(candidates, n_tracks, n_dets, inf_cost=1e9):
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    cost_matrix = build_cost_matrix(candidates, n_tracks, n_dets, inf_cost=inf_cost)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_tracks = set()
    matched_dets = set()

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < inf_cost:
            matches.append((i, j))
            matched_tracks.add(i)
            matched_dets.add(j)

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


def associate(tracks, detections, sensor_id, method="GNN", timestamp_s=None, gate_threshold=None):
    n_tracks = len(tracks)
    n_dets = len(detections)

    candidates = build_candidates(
        tracks=tracks,
        detections=detections,
        sensor_id=sensor_id,
        timestamp_s=timestamp_s,
        gate_threshold=gate_threshold,
    )

    method = method.upper()

    if method == "NN":
        matches, unmatched_tracks, unmatched_dets = associate_nn(candidates, n_tracks, n_dets)
    elif method == "GNN":
        matches, unmatched_tracks, unmatched_dets = associate_gnn(candidates, n_tracks, n_dets)
    else:
        raise ValueError(f"Unknown association method: {method}")

    return matches, unmatched_tracks, unmatched_dets, candidates


def associate_multisensor_slots(
    tracks,
    detections,
    sensor_available,
    method="GNN",
    timestamp_s=None,
    gate_probability=0.99,
):
    """
    Joint assignment across all gated detections from all sensors in one scan.
    A track may match at most one detection per available sensor.
    """
    slots = build_track_sensor_slots(tracks, sensor_available)
    n_tracks = len(slots)
    n_dets = len(detections)

    candidates = build_multisensor_slot_candidates(
        tracks=tracks,
        slots=slots,
        detections=detections,
        timestamp_s=timestamp_s,
        gate_probability=gate_probability,
    )

    method = method.upper()

    if method == "NN":
        matches, unmatched_tracks, unmatched_dets = associate_nn(candidates, n_tracks, n_dets)
    elif method == "GNN":
        matches, unmatched_tracks, unmatched_dets = associate_gnn(candidates, n_tracks, n_dets)
    else:
        raise ValueError(f"Unknown association method: {method}")

    return matches, unmatched_tracks, unmatched_dets, candidates, slots
