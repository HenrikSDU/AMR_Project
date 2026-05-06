from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


SensorId = Literal["radar", "camera", "ais", "gnss"]
Array = NDArray[np.float64]


def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class CoordinateFrameManager:
    radar_pos_ned: Array = field(default_factory=lambda: np.array([0.0, 0.0]))
    camera_pos_ned: Array = field(default_factory=lambda: np.array([-80.0, 120.0]))
    camera_boresight: float = np.deg2rad(45.0)
    camera_fov_half: float = np.deg2rad(90.0)
    camera_max_range: float = 500.0
    radar_max_range: float = 1000.0

    latest_vessel_pos_ned: Optional[Array] = None
    latest_gnss_time: Optional[float] = None

    sigma_r_radar: float = 5.0
    sigma_phi_radar: float = np.deg2rad(0.3)

    sigma_r_camera: float = 8.0
    sigma_phi_camera: float = np.deg2rad(0.15)

    sigma_pos_ais: float = 4.0
    sigma_pos_gnss: float = 2.0

    def update_gnss(self, vessel_pos_ned: Array, timestamp_s: float) -> None:
        self.latest_vessel_pos_ned = vessel_pos_ned
        self.latest_gnss_time = timestamp_s

    def get_sensor_position(
        self,
        sensor_id: SensorId,
        timestamp_s: Optional[float] = None,
    ) -> Array:
        if sensor_id == "radar":
            return self.radar_pos_ned
        if sensor_id == "camera":
            return self.camera_pos_ned
        if sensor_id == "ais":
            if self.latest_vessel_pos_ned is None:
                raise ValueError("AIS vessel position is unavailable. Update GNSS first.")
            return self.latest_vessel_pos_ned
        raise ValueError(f"Unknown sensor_id: {sensor_id}")

    def is_in_fov(
        self,
        x: Array,
        sensor_id: SensorId,
        timestamp_s: Optional[float] = None,
    ) -> bool:
        """
        Returns True if target state x is within the sensor's FOV and range.
        Radar and non-polar sensors always return True. Camera applies a
        180 deg FOV centred on camera_boresight and a 500 m range gate.
        """
        if sensor_id != "camera":
            return True

        sensor_pos = self.get_sensor_position(sensor_id, timestamp_s)
        dN = x[0] - sensor_pos[0]
        dE = x[1] - sensor_pos[1]
        r = np.sqrt(dN**2 + dE**2)

        if r > self.camera_max_range:
            return False

        bearing = np.arctan2(dE, dN)
        rel = wrap_angle(bearing - self.camera_boresight)
        return abs(rel) <= self.camera_fov_half

    def h(
        self,
        x: Array,
        sensor_id: SensorId,
        timestamp_s: Optional[float] = None,
    ) -> Array:
        """
        Measurement function.

        Radar/camera measure polar [range, bearing]. AIS measures absolute
        Cartesian NED position [north_m, east_m].
        """
        if sensor_id == "ais":
            return np.array([x[0], x[1]], dtype=float)

        px, py = x[0], x[1]
        sensor_pos = self.get_sensor_position(sensor_id, timestamp_s)
        sx, sy = sensor_pos[0], sensor_pos[1]

        dN = px - sx
        dE = py - sy

        r = np.sqrt(dN**2 + dE**2)
        phi = np.arctan2(dE, dN)

        return np.array([r, phi], dtype=float)

    def H(
        self,
        x: Array,
        sensor_id: SensorId,
        timestamp_s: Optional[float] = None,
    ) -> Array:
        """Jacobian of h wrt x = [p_N, p_E, v_N, v_E]."""
        if sensor_id == "ais":
            return np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=float,
            )

        px, py = x[0], x[1]
        sensor_pos = self.get_sensor_position(sensor_id, timestamp_s)
        sx, sy = sensor_pos[0], sensor_pos[1]

        dN = px - sx
        dE = py - sy

        q = dN**2 + dE**2
        if q < 1e-12:
            raise ValueError("Jacobian undefined for target at sensor position.")

        r = np.sqrt(q)

        return np.array(
            [
                [dN / r, dE / r, 0.0, 0.0],
                [-dE / q, dN / q, 0.0, 0.0],
            ],
            dtype=float,
        )

    def R(
        self,
        sensor_id: SensorId,
        x: Optional[Array] = None,
        timestamp_s: Optional[float] = None,
    ) -> Array:
        """Measurement noise covariance for the given sensor."""
        if sensor_id == "radar":
            return np.diag([self.sigma_r_radar**2, self.sigma_phi_radar**2])

        if sensor_id == "camera":
            if x is not None:
                sensor_pos = self.get_sensor_position("camera")
                dN = x[0] - sensor_pos[0]
                dE = x[1] - sensor_pos[1]
                r = np.sqrt(dN**2 + dE**2)
                ref_range = 50.0
                sigma_r = self.sigma_r_camera * max((r / ref_range) ** 2, 1.0)
            else:
                sigma_r = self.sigma_r_camera
            return np.diag([sigma_r**2, self.sigma_phi_camera**2])

        if sensor_id == "ais":
            return np.diag([self.sigma_pos_ais**2, self.sigma_pos_ais**2])

        if sensor_id == "gnss":
            return np.diag([self.sigma_pos_gnss**2, self.sigma_pos_gnss**2])

        raise ValueError(f"Unknown sensor_id: {sensor_id}")

    def ais_position_to_measurement(
        self,
        ais_target_pos_ned: Array,
        timestamp_s: Optional[float] = None,
    ) -> Array:
        """Return AIS absolute NED target position as [N, E]."""
        return np.asarray(ais_target_pos_ned, dtype=float)
