"""
track_manager.py
================
Full multi-target track lifecycle (Tentative → Confirmed → Coasting → Deleted)
plus MOTP / Cardinality Error metrics for Scenarios D and E.

States
------
TENTATIVE  – new EKF per unassigned detection; awaiting M-of-N confirmation.
CONFIRMED  – M-of-N achieved; reported to system output.
COASTING   – consecutive misses; predict-only EKF, covariance grows, gate widens.
DELETED    – K_del consecutive misses; removed from active list.

Public API
----------
TrackManager.step(detections, sensor_id, timestamp, dt)
    → List[Track]          (confirmed tracks this scan)

compute_motp_ce(scan_records)
    → dict with motp_series, ce_series, motp_avg, ce_avg

run_multi_target_scenario(json_file, sensor_id, cfm, **kwargs)
    → metrics dict + scan_records

plot_metrics(metrics)
    → matplotlib figures
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from Coordinate_Frame_Manager import CoordinateFrameManager
from Target_EKF import Target_EKF


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _polar_to_cart(r: float, phi: float, sensor_pos: np.ndarray) -> np.ndarray:
    """Convert polar measurement to Cartesian NED position."""
    return np.array([
        sensor_pos[0] + r * np.cos(phi),
        sensor_pos[1] + r * np.sin(phi),
    ])


def _polar_to_cart_jacobian(r: float, phi: float) -> np.ndarray:
    """
    Jacobian of (pN, pE) = (r cosφ, r sinφ)  w.r.t. [r, φ].
    Shape: (2, 2).  Sensor-offset cancels in partial derivatives.
    """
    return np.array([
        [np.cos(phi), -r * np.sin(phi)],
        [np.sin(phi),  r * np.cos(phi)],
    ])


# ──────────────────────────────────────────────────────────────────────────────
#  Track state enum
# ──────────────────────────────────────────────────────────────────────────────

class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    COASTING  = auto()
    DELETED   = auto()


_id_counter = itertools.count(1)


# ──────────────────────────────────────────────────────────────────────────────
#  Track
# ──────────────────────────────────────────────────────────────────────────────

class Track:
    """
    Single tracked object.

    Parameters
    ----------
    z_init          : First detection  [range_m, bearing_rad].
    t_init          : Timestamp of first detection (s).
    sensor_id       : Sensor that generated z_init.
    cfm             : Shared CoordinateFrameManager.
    sigma_a         : Process-noise acceleration std (m/s²).
    M, N            : M-of-N confirmation thresholds.
    K_del           : Consecutive misses before deletion.
    gate_base       : Chi-² gate threshold for healthy tracks (default 9.21 → 99 %, 2-DOF).
    coast_gate_growth : Additional gate per coast scan (absolute chi-² units).
    """

    def __init__(
        self,
        z_init: np.ndarray,
        t_init: float,
        sensor_id: str,
        cfm: CoordinateFrameManager,
        sigma_a: float = 0.05,
        M: int = 3,
        N: int = 5,
        K_del: int = 5,
        gate_base: float = 9.21,
        coast_gate_growth: float = 2.0,
    ):
        self.id    = next(_id_counter)
        self.state = TrackState.TENTATIVE
        self.cfm   = cfm

        # Config ──────────────────────────────────────────────────
        self.M                = M
        self.N                = N
        self.K_del            = K_del
        self.gate_base        = gate_base
        self.coast_gate_growth = coast_gate_growth

        # Counters ────────────────────────────────────────────────
        self.hit_history: List[bool] = []   # rolling window of N scans
        self.consecutive_misses: int = 0
        self.n_coast: int            = 0    # scans spent in COASTING
        self.age: int                = 0    # scans since creation

        # ── Initialise EKF from first polar detection ─────────────
        sensor_pos = cfm.get_sensor_position(sensor_id)
        r0, phi0   = float(z_init[0]), float(z_init[1])
        pos0       = _polar_to_cart(r0, phi0, sensor_pos)

        # Position covariance from sensor noise via Jacobian
        R_sensor = cfm.R(sensor_id=sensor_id)          # (2×2) polar noise
        J        = _polar_to_cart_jacobian(r0, phi0)    # (2×2)
        P_pos    = J @ R_sensor @ J.T

        # Velocity is unknown at birth → large prior
        P_vel = np.diag([400.0, 400.0])                 # ±20 m/s 1-σ

        x0 = np.array([pos0[0], pos0[1], 0.0, 0.0])
        P0 = np.block([
            [P_pos,            np.zeros((2, 2))],
            [np.zeros((2, 2)), P_vel           ],
        ])

        self.ekf      = Target_EKF(x0, P0, sigma_a=sigma_a)
        self.ekf.cfm  = cfm                             # inject shared CFM

        # Store first detection for finite-difference velocity init
        self._first_pos:  np.ndarray = pos0
        self._first_time: float      = t_init
        self._vel_init:   bool       = False

        # Record first scan as a hit
        self._record_hit(True)

    # ── Properties ────────────────────────────────────────────────

    @property
    def gate_threshold(self) -> float:
        """Gate threshold – widens linearly while coasting."""
        if self.state == TrackState.COASTING:
            return self.gate_base + self.coast_gate_growth * self.n_coast
        return self.gate_base

    @property
    def pos(self) -> np.ndarray:
        """Current position estimate [pN, pE]."""
        return self.ekf.x[:2].copy()

    @property
    def x(self) -> np.ndarray:
        return self.ekf.x.copy()

    @property
    def P(self) -> np.ndarray:
        return self.ekf.P.copy()

    # ── Lifecycle ─────────────────────────────────────────────────

    def on_predict(self, dt: float) -> None:
        """Predict step (called every scan before association)."""
        self.ekf.predict(dt)
        self.age += 1

    def on_hit(self, z: np.ndarray, sensor_id: str, timestamp: float) -> None:
        """
        EKF update with a matched detection.

        On the second hit of a TENTATIVE track the velocity is initialised
        via finite difference between the first and second Cartesian positions.
        """
        # ── Velocity initialisation on second detection ────────────
        if not self._vel_init:
            sensor_pos = self.cfm.get_sensor_position(sensor_id)
            pos1 = _polar_to_cart(float(z[0]), float(z[1]), sensor_pos)
            dt_init = timestamp - self._first_time
            if dt_init > 1e-6:
                v = (pos1 - self._first_pos) / dt_init
                self.ekf.x[2] = v[0]
                self.ekf.x[3] = v[1]
            self._vel_init = True

        # ── EKF measurement update ─────────────────────────────────
        self.ekf.update_sensor(z, [sensor_id])

        # ── State machine ──────────────────────────────────────────
        self.consecutive_misses = 0
        self.n_coast            = 0
        if self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED   # re-acquired

        self._record_hit(True)
        self._check_confirmation()

    def on_miss(self) -> None:
        """No detection associated this scan – predict-only coast or delete."""
        self.consecutive_misses += 1
        self._record_hit(False)

        if self.state == TrackState.TENTATIVE:
            # Tentative tracks die quickly on consecutive misses
            if self.consecutive_misses >= 2:
                self.state = TrackState.DELETED

        elif self.state == TrackState.CONFIRMED:
            self.state  = TrackState.COASTING
            self.n_coast = 1

        elif self.state == TrackState.COASTING:
            self.n_coast += 1
            if self.consecutive_misses >= self.K_del:
                self.state = TrackState.DELETED

    # ── Private ───────────────────────────────────────────────────

    def _record_hit(self, hit: bool) -> None:
        self.hit_history.append(hit)
        if len(self.hit_history) > self.N:
            self.hit_history.pop(0)

    def _check_confirmation(self) -> None:
        if self.state == TrackState.TENTATIVE:
            if sum(self.hit_history[-self.N:]) >= self.M:
                self.state = TrackState.CONFIRMED

    def __repr__(self) -> str:
        return (f"Track(id={self.id}, {self.state.name}, "
                f"pos=[{self.pos[0]:.1f}, {self.pos[1]:.1f}], "
                f"misses={self.consecutive_misses})")


# ──────────────────────────────────────────────────────────────────────────────
#  Track Manager
# ──────────────────────────────────────────────────────────────────────────────

class TrackManager:
    """
    Manages the full multi-target lifecycle for one sensor stream.

    Usage
    -----
    tm = TrackManager(cfm)
    for detections, sensor_id, timestamp, dt in scan_loop:
        confirmed = tm.step(detections, sensor_id, timestamp, dt)
    """

    def __init__(
        self,
        cfm: CoordinateFrameManager,
        sigma_a: float           = 0.05,
        M: int                   = 3,
        N: int                   = 5,
        K_del: int               = 5,
        gate_base: float         = 9.21,
        coast_gate_growth: float = 2.0,
        merge_threshold: float   = 5.991,   # chi²(0.95, 2-DOF)
        assoc_method: str        = "GNN",
    ):
        self.cfm               = cfm
        self.sigma_a           = sigma_a
        self.M                 = M
        self.N                 = N
        self.K_del             = K_del
        self.gate_base         = gate_base
        self.coast_gate_growth = coast_gate_growth
        self.merge_threshold   = merge_threshold
        self.assoc_method      = assoc_method.upper()

        self.tracks: List[Track] = []

    # ── Main per-scan step ─────────────────────────────────────────

    def step(
        self,
        detections: List[np.ndarray],
        sensor_id: str,
        timestamp: float,
        dt: float,
    ) -> List[Track]:
        """
        One full tracking cycle.

        1. Predict all active tracks.
        2. Associate detections to tracks (per-track gate width).
        3. Update matched tracks; miss unmatched tracks.
        4. Spawn TENTATIVE tracks for unmatched detections.
        5. Prune DELETED tracks.
        6. Merge duplicate confirmed/coasting tracks.

        Returns
        -------
        List of currently confirmed (and coasting) tracks.
        """
        active = [t for t in self.tracks if t.state != TrackState.DELETED]

        # 1 ── Predict ──────────────────────────────────────────────
        for track in active:
            track.on_predict(dt)

        # 2 ── Associate ────────────────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = self._associate(
            active, detections, sensor_id, timestamp
        )

        # 3 ── Update / Miss ────────────────────────────────────────
        for ti, di in matches:
            active[ti].on_hit(detections[di], sensor_id, timestamp)

        for ti in unmatched_tracks:
            active[ti].on_miss()

        # 4 ── Spawn tentative tracks ───────────────────────────────
        for di in unmatched_dets:
            self._spawn(detections[di], timestamp, sensor_id)

        # 5 ── Prune deleted ────────────────────────────────────────
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        # 6 ── Merge duplicates ─────────────────────────────────────
        self._merge_duplicates()

        return self.confirmed_tracks

    # ── Properties ────────────────────────────────────────────────

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Tracks in CONFIRMED or COASTING state (reported to output)."""
        return [t for t in self.tracks
                if t.state in (TrackState.CONFIRMED, TrackState.COASTING)]

    @property
    def all_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state != TrackState.DELETED]

    # ── Private ───────────────────────────────────────────────────

    def _associate(
        self,
        active: List[Track],
        detections: List[np.ndarray],
        sensor_id: str,
        timestamp: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Build per-track gated candidates (each track uses its own gate_threshold)
        then run GNN (Hungarian) or NN assignment.
        """
        from data_association import associate_gnn, associate_nn

        n_tracks = len(active)
        n_dets   = len(detections)

        if n_tracks == 0:
            return [], [], list(range(n_dets))
        if n_dets == 0:
            return [], list(range(n_tracks)), []

        # ── Build gated candidates ─────────────────────────────────
        candidates: List[Dict] = []
        for i, track in enumerate(active):
            h_x = track.cfm.h(track.ekf.x, sensor_id=sensor_id, timestamp_s=timestamp)
            H   = track.cfm.H(track.ekf.x, sensor_id=sensor_id, timestamp_s=timestamp)
            R   = track.cfm.R(sensor_id=sensor_id, timestamp_s=timestamp)

            for j, z in enumerate(detections):
                ok, d2 = track.ekf.compute_gating_distance(
                    z=z, h_x=h_x, H=H, R=R,
                    sensor_id=sensor_id,
                    threshold=track.gate_threshold,   # ← per-track, widens on coast
                )
                if ok:
                    candidates.append({"track_idx": i, "det_idx": j, "cost": float(d2)})

        # ── Assignment ────────────────────────────────────────────
        if self.assoc_method == "NN":
            return associate_nn(candidates, n_tracks, n_dets)
        else:
            return associate_gnn(candidates, n_tracks, n_dets)

    def _spawn(self, z: np.ndarray, timestamp: float, sensor_id: str) -> None:
        """Create a new TENTATIVE track from an unmatched detection."""
        track = Track(
            z_init=z,
            t_init=timestamp,
            sensor_id=sensor_id,
            cfm=self.cfm,
            sigma_a=self.sigma_a,
            M=self.M,
            N=self.N,
            K_del=self.K_del,
            gate_base=self.gate_base,
            coast_gate_growth=self.coast_gate_growth,
        )
        self.tracks.append(track)

    def _merge_duplicates(self) -> None:
        """
        Merge confirmed/coasting track pairs that are too close in state space.

        Criterion (position-only Mahalanobis to avoid velocity masking):
            d² = (xi − xj)ᵀ (Pi + Pj)⁻¹ (xi − xj)  < merge_threshold
        The newer track (higher id) is absorbed into the older one.
        """
        reportable = [t for t in self.tracks
                      if t.state in (TrackState.CONFIRMED, TrackState.COASTING)]
        to_delete: set = set()

        for i in range(len(reportable)):
            for j in range(i + 1, len(reportable)):
                ti_id = reportable[i].id
                tj_id = reportable[j].id
                if ti_id in to_delete or tj_id in to_delete:
                    continue

                ti, tj = reportable[i], reportable[j]
                dx = ti.ekf.x[:2] - tj.ekf.x[:2]
                S  = ti.ekf.P[:2, :2] + tj.ekf.P[:2, :2]
                try:
                    d2 = float(dx @ np.linalg.solve(S, dx))
                except np.linalg.LinAlgError:
                    continue

                if d2 < self.merge_threshold:
                    to_delete.add(tj_id)   # keep the older (lower id) track

        self.tracks = [t for t in self.tracks if t.id not in to_delete]


# ──────────────────────────────────────────────────────────────────────────────
#  Metrics: MOTP and Cardinality Error
# ──────────────────────────────────────────────────────────────────────────────

def compute_motp_ce(scan_records: List[Dict]) -> Dict:
    """
    Compute MOTP and Cardinality Error from per-scan records.

    Parameters
    ----------
    scan_records : list of dicts, one entry per scan:
        {
            "timestamp"           : float,
            "confirmed_positions" : np.ndarray (Nc, 2)  [pN, pE] per confirmed track,
            "true_positions"      : np.ndarray (Nt, 2)  [pN, pE] per active true target,
        }

    Returns
    -------
    dict:
        "timestamps"    : (K,) float array
        "motp_series"   : (K,) float – per-scan mean distance to nearest true target
                          (NaN when no matches possible)
        "ce_series"     : (K,) float – per-scan |Nc − Nt|
        "motp_avg"      : scalar – average MOTP over all matched pairs
        "ce_avg"        : scalar – mean CE across scans
        "total_matches" : int
    """
    timestamps: List[float]  = []
    motp_series: List[float] = []
    ce_series: List[float]   = []

    total_dist  = 0.0
    total_match = 0

    for rec in scan_records:
        t        = float(rec["timestamp"])
        conf_pos = np.atleast_2d(rec["confirmed_positions"])   # (Nc, 2)
        true_pos = np.atleast_2d(rec["true_positions"])        # (Nt, 2)

        # Handle empty arrays (atleast_2d of empty gives shape (1,0))
        if conf_pos.shape[1] == 0 or conf_pos.size == 0:
            conf_pos = np.empty((0, 2))
        if true_pos.shape[1] == 0 or true_pos.size == 0:
            true_pos = np.empty((0, 2))

        Nc = conf_pos.shape[0]
        Nt = true_pos.shape[0]

        # ── Cardinality Error ──────────────────────────────────────
        ce = float(abs(Nc - Nt))

        # ── MOTP via minimum-cost Hungarian matching ───────────────
        if Nc > 0 and Nt > 0:
            # Euclidean distance matrix (Nc × Nt)
            dist_mat = np.linalg.norm(
                conf_pos[:, None, :] - true_pos[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(dist_mat)
            matched_dists    = dist_mat[row_ind, col_ind]
            motp_scan        = float(np.mean(matched_dists))
            total_dist      += float(np.sum(matched_dists))
            total_match     += len(row_ind)
        else:
            motp_scan = float("nan")

        timestamps.append(t)
        motp_series.append(motp_scan)
        ce_series.append(ce)

    motp_avg = total_dist / total_match if total_match > 0 else float("nan")

    return {
        "timestamps"   : np.array(timestamps),
        "motp_series"  : np.array(motp_series),
        "ce_series"    : np.array(ce_series,   dtype=float),
        "motp_avg"     : motp_avg,
        "ce_avg"       : float(np.mean(ce_series)) if ce_series else float("nan"),
        "total_matches": total_match,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Ground-truth helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_multi_target_gt(data: Dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Parse ground_truth dict from the JSON scenario file.
    Handles both single-target {"0": [...]} and multi-target {"0":..., "1":...}.

    Returns
    -------
    gt : dict  target_id → (gt_times np.ndarray,  gt_states np.ndarray (K×4))
    """
    gt: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tid, rows in data["ground_truth"].items():
        arr         = np.array(rows)          # (K, 5): [t, pN, pE, vN, vE]
        gt[tid]     = (arr[:, 0], arr[:, 1:]) # times, states
    return gt


def _active_true_targets(
    gt: Dict[str, Tuple[np.ndarray, np.ndarray]],
    timestamp: float,
    tol: float = 2.0,
) -> np.ndarray:
    """
    Return (Nt, 2) array of ground-truth positions at *timestamp*.
    A target is considered active if its ground-truth spans this time.
    Interpolates linearly between ground-truth samples.
    """
    positions = []
    for tid, (times, states) in gt.items():
        if times[0] - tol <= timestamp <= times[-1] + tol:
            pN = float(np.interp(timestamp, times, states[:, 0]))
            pE = float(np.interp(timestamp, times, states[:, 1]))
            positions.append([pN, pE])
    return np.array(positions) if positions else np.empty((0, 2))


# ──────────────────────────────────────────────────────────────────────────────
#  Scenario runner (Scenarios D and E)
# ──────────────────────────────────────────────────────────────────────────────

def run_multi_target_scenario(
    json_file: str,
    sensor_id: str            = "radar",
    cfm: Optional[CoordinateFrameManager] = None,
    # TrackManager config
    sigma_a: float            = 0.05,
    M: int                    = 3,
    N: int                    = 5,
    K_del: int                = 5,
    gate_base: float          = 9.21,
    coast_gate_growth: float  = 2.0,
    merge_threshold: float    = 5.991,
    assoc_method: str         = "GNN",
    # Scan grouping tolerance
    scan_dt_tol: float        = 0.5,
) -> Tuple[Dict, List[Dict], TrackManager]:
    """
    Run the full track lifecycle over a multi-target scenario JSON file.

    Parameters
    ----------
    json_file         : Path to the scenario JSON (same format as Scenarios A-C).
    sensor_id         : Which sensor stream to process ("radar", "camera", …).
    cfm               : CoordinateFrameManager; a default one is created if None.
    M, N              : M-of-N confirmation (default 3-of-5).
    K_del             : Coast scans before deletion (default 5).
    gate_base         : Base Mahalanobis gate (default 9.21 → χ²₂, 99 %).
    coast_gate_growth : Extra gate chi-² per coast scan (default 2.0).
    merge_threshold   : Mahalanobis threshold for duplicate merge (default 5.991).
    assoc_method      : "GNN" (Hungarian) or "NN" (nearest-neighbour).
    scan_dt_tol       : Measurements within this many seconds form one scan.

    Returns
    -------
    metrics     : dict from compute_motp_ce (+ "scenario")
    scan_records: raw per-scan data for further analysis
    tm          : TrackManager instance (access tracks history)
    """
    if cfm is None:
        cfm = CoordinateFrameManager()

    with open(json_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # ── Ground truth ──────────────────────────────────────────────
    gt = _load_multi_target_gt(data)

    # ── Filter & sort measurements for the requested sensor ───────
    raw_meas = [
        m for m in data["measurements"]
        if m["sensor_id"] == sensor_id
        # NOTE: remove the is_false_alarm filter below once false-alarm
        #       handling (gating) is relied upon exclusively.
        and not m.get("is_false_alarm", False)
    ]
    raw_meas.sort(key=lambda m: m["time"])

    if not raw_meas:
        raise ValueError(f"No measurements found for sensor '{sensor_id}' in {json_file}")

    # ── Group into scans ──────────────────────────────────────────
    # A "scan" is a batch of measurements within scan_dt_tol seconds.
    scans: List[Tuple[float, List[np.ndarray]]] = []
    current_t    = raw_meas[0]["time"]
    current_dets: List[np.ndarray] = []

    for m in raw_meas:
        t = m["time"]
        z = np.array([m["range_m"], m["bearing_rad"]])
        if t - current_t <= scan_dt_tol:
            current_dets.append(z)
        else:
            scans.append((current_t, current_dets))
            current_t    = t
            current_dets = [z]
    scans.append((current_t, current_dets))  # flush last scan

    # ── TrackManager ──────────────────────────────────────────────
    tm = TrackManager(
        cfm=cfm,
        sigma_a=sigma_a,
        M=M, N=N,
        K_del=K_del,
        gate_base=gate_base,
        coast_gate_growth=coast_gate_growth,
        merge_threshold=merge_threshold,
        assoc_method=assoc_method,
    )

    scan_records: List[Dict] = []
    prev_t = scans[0][0]

    for scan_t, dets in scans:
        dt = max(scan_t - prev_t, 1e-3)   # guard against zero dt

        confirmed = tm.step(
            detections=dets,
            sensor_id=sensor_id,
            timestamp=scan_t,
            dt=dt,
        )
        prev_t = scan_t

        # ── Record for metrics ─────────────────────────────────────
        conf_pos = np.array([t.pos for t in confirmed]) if confirmed else np.empty((0, 2))
        true_pos = _active_true_targets(gt, scan_t)

        scan_records.append({
            "timestamp"           : scan_t,
            "n_confirmed"         : len(confirmed),
            "n_true"              : true_pos.shape[0],
            "confirmed_positions" : conf_pos,
            "true_positions"      : true_pos,
            "confirmed_tracks"    : list(confirmed),   # snapshot references
        })

    # ── Compute metrics ───────────────────────────────────────────
    metrics = compute_motp_ce(scan_records)
    metrics["scenario"] = json_file

    print(f"\n{'─'*55}")
    print(f"  Scenario : {json_file}")
    print(f"  Scans    : {len(scans)}")
    print(f"  MOTP avg : {metrics['motp_avg']:.2f} m")
    print(f"  CE avg   : {metrics['ce_avg']:.2f}")
    print(f"  Matches  : {metrics['total_matches']}")
    print(f"{'─'*55}\n")

    return metrics, scan_records, tm


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_metrics(metrics: Dict, title: str = "") -> None:
    """
    Plot MOTP and CE time series from a metrics dict.

    Parameters
    ----------
    metrics : output of compute_motp_ce / run_multi_target_scenario.
    title   : optional super-title string.
    """
    ts      = metrics["timestamps"]
    motp    = metrics["motp_series"]
    ce      = metrics["ce_series"]
    motp_a  = metrics["motp_avg"]
    ce_a    = metrics["ce_avg"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title or "Track Lifecycle Metrics", fontsize=13)

    # ── MOTP ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ts, motp, color="steelblue", linewidth=1.5, label="MOTP per scan")
    ax.axhline(motp_a, color="steelblue", linestyle="--", linewidth=1.0,
               label=f"Mean MOTP = {motp_a:.2f} m")
    ax.set_ylabel("MOTP (m)")
    ax.set_title("Multiple Object Tracking Precision")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # ── CE ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.step(ts, ce, color="darkorange", linewidth=1.5, where="post", label="CE per scan")
    ax.axhline(ce_a, color="darkorange", linestyle="--", linewidth=1.0,
               label=f"Mean CE = {ce_a:.2f}")
    ax.set_ylabel("|N_confirmed − N_true|")
    ax.set_xlabel("Time (s)")
    ax.set_title("Cardinality Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_scenario_trajectories(
    scan_records: List[Dict],
    gt: Optional[Dict] = None,
    title: str = "",
) -> None:
    """
    Plot estimated track trajectories against ground truth.

    Parameters
    ----------
    scan_records : output list from run_multi_target_scenario.
    gt           : optional dict from _load_multi_target_gt for plotting GT lines.
    title        : figure title.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title or "Track Trajectories")

    # ── Collect per-track position histories ──────────────────────
    track_histories: Dict[int, List[np.ndarray]] = {}
    for rec in scan_records:
        for t in rec["confirmed_tracks"]:
            track_histories.setdefault(t.id, []).append(t.pos.copy())

    colors = plt.cm.tab10.colors
    for k, (tid, pos_list) in enumerate(track_histories.items()):
        pts = np.array(pos_list)
        c   = colors[k % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], color=c, linewidth=1.5,
                label=f"Track {tid}")
        ax.scatter(pts[0, 0], pts[0, 1], color=c, marker="o", s=40, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=c, marker="s", s=40, zorder=5)

    # ── Ground truth ──────────────────────────────────────────────
    if gt is not None:
        for tid, (times, states) in gt.items():
            ax.plot(states[:, 0], states[:, 1], "k--", linewidth=1.2,
                    label=f"GT target {tid}" if tid == list(gt.keys())[0] else "_nolegend_")

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cardinality_comparison(
    metrics_D: Dict,
    metrics_E: Dict,
) -> None:
    """Side-by-side CE and MOTP bar chart comparing Scenarios D and E."""
    labels  = ["Scenario D", "Scenario E"]
    motps   = [metrics_D["motp_avg"], metrics_E["motp_avg"]]
    ces     = [metrics_D["ce_avg"],   metrics_E["ce_avg"]]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].bar(labels, motps, color=["steelblue", "tomato"])
    axes[0].set_ylabel("MOTP (m)")
    axes[0].set_title("Mean MOTP")
    axes[0].grid(True, axis="y", alpha=0.4)

    axes[1].bar(labels, ces, color=["steelblue", "tomato"])
    axes[1].set_ylabel("Mean |Nc − Nt|")
    axes[1].set_title("Mean Cardinality Error")
    axes[1].grid(True, axis="y", alpha=0.4)

    plt.suptitle("Scenario D vs E – Tracking Metrics", fontsize=12)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  Example entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Scenario D ────────────────────────────────────────────────
    metrics_D, records_D, tm_D = run_multi_target_scenario(
        json_file    = "harbour_sim_output/scenario_D.json",
        sensor_id    = "radar",
        M=3, N=5, K_del=5,
    )
    plot_metrics(metrics_D, title="Scenario D – MOTP & CE")
    plot_scenario_trajectories(records_D, title="Scenario D – Trajectories")

    # ── Scenario E ────────────────────────────────────────────────
    metrics_E, records_E, tm_E = run_multi_target_scenario(
        json_file    = "harbour_sim_output/scenario_E.json",
        sensor_id    = "radar",
        M=3, N=5, K_del=5,
    )
    plot_metrics(metrics_E, title="Scenario E – MOTP & CE")
    plot_scenario_trajectories(records_E, title="Scenario E – Trajectories")

    # ── Side-by-side comparison ───────────────────────────────────
    plot_cardinality_comparison(metrics_D, metrics_E)