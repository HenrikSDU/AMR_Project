import numpy as np
from numpy.linalg import inv
from Coordinate_Frame_Manager import CoordinateFrameManager
from scipy.linalg import block_diag

from typing import List, Literal

SensorId = Literal["radar", "camera", "ais", "gnss"]

def wrap_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

class Target_EKF:
    
    def __init__(self, x0: np.ndarray, P0: np.ndarray, cfm: CoordinateFrameManager, sigma_a: float = 0.05):
        """
        x = [p_N, p_E, v_N, v_E]
        """
        self.x = x0
        self.P = P0
        self.sigma_a  = sigma_a

        self.cfm = cfm

    def predict(self, dt: float):
        if dt <= 0:
            return # Prevent backwards or zero-time predictions
            
        # Recalculate matrices for the specific time gap
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        q = self.sigma_a**2
        self.Q = q * np.array([
            [dt4/4,     0, dt3/2,     0],
            [    0, dt4/4,     0, dt3/2],
            [dt3/2,     0,   dt2,     0],
            [    0, dt3/2,     0,   dt2],
        ])

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update_sensor(self, z: np.ndarray, sensor_ids: List[SensorId]):
        h_list = []
        H_list = []
        R_list = []
        
        # Track indices that require angle wrapping
        angle_indices = []
        current_idx = 0

        for sensor in sensor_ids:
            # Fetch components from CoordinateFrameManager
            h_i = self.cfm.h(self.x, sensor_id=sensor)
            H_i = self.cfm.H(self.x, sensor_id=sensor)
            R_i = self.cfm.R(sensor_id=sensor) # Ensure this returns a 2x2 matrix
            
            h_list.append(h_i)
            H_list.append(H_i)
            R_list.append(R_i)

            # Radar and Camera are [range, bearing] -> wrap index 1 of their block
            if sensor in ["radar", "camera"]:
                angle_indices.append(current_idx + 1)
            
            current_idx += len(h_i)

        # Stack measurements into a single large system
        h = np.concatenate(h_list)
        H = np.vstack(H_list)
        R = block_diag(*R_list)

        # Calculate Innovation
        innov = z - h
        
        # Selective Wrapping: Apply only to radar/camera bearing components
        for idx in angle_indices:
            innov[idx] = wrap_angle(innov[idx])

        # Standard EKF Update (Joseph Form for stability)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ inv(S)

        self.x = self.x + K @ innov
        
        I = np.eye(len(self.x))
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return innov, S
    
    def compute_gating_distance(self, z, h_x, H, R, sensor_id: SensorId, threshold=9.21):
        # mahalanobis gating
        y = z - h_x

        # Only wrap the angle if the sensor is polar (radar or camera)
        if sensor_id in ["radar", "camera"]:
            y[1] = wrap_angle(y[1])

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # compute squared Mahalanobis Distance (d^2)
        S_inv = np.linalg.pinv(S)
        d_squared = y.T @ S_inv @ y

        # Evaluate against the chi squared threshold
        is_within_gate = d_squared <= threshold

        return is_within_gate, float(d_squared)