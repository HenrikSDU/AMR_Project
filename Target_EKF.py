from typing import List, Literal

import numpy as np
from numpy.linalg import inv

from Coordinate_Frame_Manager import CoordinateFrameManager

try:
    from scipy.linalg import block_diag
except ModuleNotFoundError:
    def block_diag(*arrays):
        rows = sum(a.shape[0] for a in arrays)
        cols = sum(a.shape[1] for a in arrays)
        out = np.zeros((rows, cols), dtype=float)
        r = 0
        c = 0
        for a in arrays:
            rr, cc = a.shape
            out[r:r + rr, c:c + cc] = a
            r += rr
            c += cc
        return out


SensorId = Literal["radar", "camera", "ais", "gnss"]


def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class Target_EKF:
    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        dt: float = 1.0,
        sigma_a: float = 0.05,
    ):
        """
        Constant-velocity EKF for x = [p_N, p_E, v_N, v_E].
        """
        self.x = x0
        self.P = P0
        self.sigma_a = sigma_a
        self._set_motion_model(dt)
        self.cfm = CoordinateFrameManager()

    def _set_motion_model(self, dt: float) -> None:
        """Update F and Q for a given prediction interval."""
        self.dt = float(dt)
        dt2, dt3, dt4 = self.dt**2, self.dt**3, self.dt**4

        self.F = np.array(
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        q = self.sigma_a**2
        self.Q = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=float,
        )

    def predict(self) -> None:
        """Predict one fixed-rate step using the dt from initialization."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def predict_dt(self, dt: float) -> None:
        """Predict one asynchronous step. Used by Scenario C only."""
        old_dt = self.dt
        self._set_motion_model(dt)
        self.predict()
        self._set_motion_model(old_dt)

    def update_sensor(self, z: np.ndarray, sensor_ids: List[SensorId]):
        h_list = []
        H_list = []
        R_list = []

        angle_indices = []
        current_idx = 0

        for sensor in sensor_ids:
            h_i = self.cfm.h(self.x, sensor_id=sensor)
            H_i = self.cfm.H(self.x, sensor_id=sensor)
            R_i = self.cfm.R(sensor_id=sensor)

            h_list.append(h_i)
            H_list.append(H_i)
            R_list.append(R_i)

            if sensor in ["radar", "camera"]:
                angle_indices.append(current_idx + 1)

            current_idx += len(h_i)

        h = np.concatenate(h_list)
        H = np.vstack(H_list)
        R = block_diag(*R_list)

        innov = z - h

        for idx in angle_indices:
            innov[idx] = wrap_angle(innov[idx])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ inv(S)

        self.x = self.x + K @ innov

        I = np.eye(len(self.x))
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return innov, S

    def compute_gating_distance(
        self,
        z,
        h_x,
        H,
        R,
        sensor_id: SensorId,
        threshold: float,
    ):
        y = z - h_x


        if sensor_id in ["radar", "camera"]:
            y[1] = wrap_angle(y[1])

        S = H @ self.P @ H.T + R
        S_inv = np.linalg.pinv(S)
        d_squared = y.T @ S_inv @ y
        is_within_gate = d_squared <= threshold

        return is_within_gate, float(d_squared)
