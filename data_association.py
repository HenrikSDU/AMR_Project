## Moved mahalanobis gating to Target_EKF.py

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_candidates(tracks, detections, sensor_id, timestamp_s=None, gate_threshold=9.21):
    """
    Build gated candidate associations.

    Parameters
    ----------
    tracks : list[Target_EKF]
        Active tracks.
    detections : list[np.ndarray]
        List of measurement vectors z, e.g. [range, bearing].
    sensor_id : str
        "radar", "camera", "ais", ...
    timestamp_s : float | None
        Optional timestamp passed through if later needed.
    gate_threshold : float
        Chi-square gate threshold.

    Returns
    -------
    candidates : list[dict]
        Each dict has:
            {
                "track_idx": i,
                "det_idx": j,
                "cost": d2
            }
    """
    candidates = []

    for i, track in enumerate(tracks):
        h_x = track.cfm.h(track.x, sensor_id=sensor_id, timestamp_s=timestamp_s)
        H = track.cfm.H(track.x, sensor_id=sensor_id, timestamp_s=timestamp_s)
        R = track.cfm.R(sensor_id=sensor_id, timestamp_s=timestamp_s)

        for j, z in enumerate(detections):
            ok, d2 = track.compute_gating_distance(
                z=z,
                h_x=h_x,
                H=H,
                R=R,
                sensor_id=sensor_id,
                threshold=gate_threshold,
            )

            if ok:
                candidates.append({
                    "track_idx": i,
                    "det_idx": j,
                    "cost": float(d2),
                })

    return candidates


def build_cost_matrix(candidates, n_tracks, n_dets, inf_cost=1e9):
    """
    Convert gated candidates into a dense cost matrix.
    """
    cost_matrix = np.full((n_tracks, n_dets), inf_cost, dtype=float)

    for c in candidates:
        i = c["track_idx"]
        j = c["det_idx"]
        cost_matrix[i, j] = c["cost"]

    return cost_matrix


def associate_nn(candidates, n_tracks, n_dets):
    """
    Nearest-neighbour:
    sort all valid candidates by cost and accept the lowest-cost
    non-conflicting pairs first.
    """
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
    """
    Global nearest-neighbour using Hungarian assignment.
    """
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


def associate(tracks, detections, sensor_id, method="GNN", timestamp_s=None, gate_threshold=9.21):
    """
    Full association wrapper:
    1. build gated candidates
    2. run NN or GNN

    Returns
    -------
    matches : list[(track_idx, det_idx)]
    unmatched_tracks : list[int]
    unmatched_dets : list[int]
    candidates : list[dict]
    """
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