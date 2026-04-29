from Target_EKF import Target_EKF
import numpy as np
from numpy.linalg import inv
import json
import matplotlib.pyplot as plt
from collections import Counter

# HELPER FUNCTIONS

def extract_ground_truth(data):
    gt = data["ground_truth"]["0"]
    gt_times = np.array([row[0] for row in gt])
    gt_states = np.array([row[1:] for row in gt])
    return gt_times, gt_states


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

# EKF RUNNING FUNCTIONS

def run_ekf_radar_only(ts, zs_radar, x0):

    dt = np.mean(np.diff(ts))

    P0 = np.diag([25, 25, 2500, 2500])

    ekf = Target_EKF(x0, P0, dt=dt)

    N = len(ts)
    x_est = np.zeros((N, 4))
    innov_hist, S_hist = [], []

    SensorId = ["radar"]

    for k in range(N):

        ekf.predict()

        z = zs_radar[k]
        innov, S = ekf.update_sensor(z, SensorId)

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

        # --- RADAR UPDATE ---
        zr = zs_radar[k]
        innov, S = ekf.update_sensor(zr, ["radar"])
        innov_hist.append(innov)
        S_hist.append(S)

        # --- CAMERA UPDATE (ONLY IF TIME MATCHES) ---
        while cam_idx < len(ts_camera) and ts_camera[cam_idx] <= t_k + 1e-6:
            
            zc = zs_camera[cam_idx]
            if ekf.cfm.is_in_fov(ekf.x, "camera"):
                h_c = ekf.cfm.h(ekf.x, "camera")
                H_c = ekf.cfm.H(ekf.x, "camera")
                R_c = ekf.cfm.R("camera", x=ekf.x)   # pass x for range-dependent noise
                
                in_gate, d2 = ekf.compute_gating_distance(zc, h_c, H_c, R_c, "camera")
                if in_gate:
                    innov, S = ekf.update_sensor(zc, ["camera"])
                    innov_hist.append(innov)
                    S_hist.append(S)

            cam_idx += 1

        x_est[k] = ekf.x

    return x_est, innov_hist, S_hist


def run_ekf_joint(ts_radar, ts_camera, zs_radar, zs_camera, x0):
    dt            = np.mean(np.diff(ts_radar))
    camera_dt     = np.mean(np.diff(ts_camera)) if len(ts_camera) > 1 else dt
    max_staleness = camera_dt / 2.0   # 1.0 s for 0.5 Hz camera

    P0  = np.diag([25, 25, 2500, 2500])
    ekf = Target_EKF(x0, P0, dt=dt)
    N   = len(ts_radar)

    x_est                    = np.zeros((N, 4))
    innov_hist, S_hist       = [], []
    cam_idx, count_cam_used  = 0, 0

    for k in range(N):
        ekf.predict()
        t_k = ts_radar[k]
        z_r = zs_radar[k]

        # Collect the most recent camera measurement up to t_k
        cam_at_this_step = None
        cam_ts           = None
        while cam_idx < len(ts_camera) and ts_camera[cam_idx] <= t_k + 1e-6:
            cam_at_this_step = zs_camera[cam_idx]
            cam_ts           = ts_camera[cam_idx]
            cam_idx         += 1

        # Only use if fresh, in FOV, and within gate
        use_camera = False
        if cam_at_this_step is not None:
            age = t_k - cam_ts
            if age <= max_staleness:
                z_c = cam_at_this_step
                if ekf.cfm.is_in_fov(ekf.x, "camera"):
                    h_c    = ekf.cfm.h(ekf.x, "camera")
                    H_c    = ekf.cfm.H(ekf.x, "camera")
                    R_c    = ekf.cfm.R("camera", x=ekf.x)
                    in_gate, d2 = ekf.compute_gating_distance(z_c, h_c, H_c, R_c, "camera")
                    if in_gate:
                        use_camera      = True
                        count_cam_used += 1

        if use_camera:
            z         = np.hstack([z_r, z_c])
            innov, S  = ekf.update_sensor(z, ["radar", "camera"])
        else:
            innov, S  = ekf.update_sensor(z_r, ["radar"])

        innov_hist.append(innov)
        S_hist.append(S)
        x_est[k] = ekf.x

    print(f"Camera fused on {count_cam_used} / {N} radar scans")
    return x_est, innov_hist, S_hist


def run_ekf_async_fusion(measurements, x0):
    """
    Event-driven fusion for Scenario C.

    Radar/camera are polar measurements. AIS is absolute Cartesian N/E.
    Measurements are already single-target and false-alarm filtered for T5.
    """
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
            ekf.predict(dt=dt)
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

    return (
        np.array(x_est),
        np.array(ts_est),
        innov_hist,
        S_hist,
        update_counts,
        rejected_counts,
    )

# MAIN SIMULATION FUNCTION

def sim_tracking(json_file, scenario="A", mode="radar"):

    data = json.load(open(json_file, "r", encoding="utf-8"))

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

            x_est, innov_hist, S_hist = run_ekf_radar_only(ts, zs_radar, x0 = np.array([800, 600, -1, -2]))

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

            # print("ts_radar:", ts_radar)
            # print("ts_camera:", ts_camera)

            if mode == "joint":
                x_est, innov_hist, S_hist = run_ekf_joint(ts_radar, ts_camera, zs_radar, zs_camera, x0 = np.array([400, 80, 1.2, 2.2]))
            elif mode == 'radar':
                x_est, innov_hist, S_hist = run_ekf_radar_only(ts_radar, zs_radar, x0 = np.array([400, 80, 1.2, 2.2]))
            else: # sequential
                x_est, innov_hist, S_hist = run_ekf_sequential(ts_radar, ts_camera, zs_radar, zs_camera, x0 = np.array([400, 80, 1.2, 2.2]))
            
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

    compute_rmse(x_est, gt_interp)

    return x_est, gt_interp, innov_hist, S_hist


# PLOTTING

def plot_trajectories(gt, results, labels, measurements=None):

    plt.figure(figsize=(8, 6))
    plt.plot(gt[:, 0], gt[:, 1], 'k--', linewidth=2, label="Ground Truth")

    for x_est, label in zip(results, labels):
        plt.plot(x_est[:, 0], x_est[:, 1], linewidth=1.5, label=label)

    sensor_origins = {
        "Radar":  np.array([0.0,   0.0]),
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


def run_demo(scenario="C", mode="ais", show_plots=True):
    json_file = f"harbour_sim_output/scenario_{scenario}.json"
    result = sim_tracking(json_file, scenario=scenario, mode=mode)

    if scenario == "A":
        x_est, gt, innov, S, meas = result
        measurements = [(meas, "Radar")]
        label = "Radar-only"
    elif scenario == "B":
        x_est, gt, innov, S, radar, camera = result
        measurements = [(radar, "Radar")]
        if mode != "radar":
            measurements.append((camera, "Camera"))
        label = {
            "radar": "Radar-only",
            "sequential": "Sequential Fusion",
            "joint": "Joint Fusion",
        }.get(mode, mode)
    elif scenario == "C":
        x_est, gt, innov, S, radar, camera, ais, _ = result
        measurements = [
            (radar, "Radar"),
            (camera, "Camera"),
        ]
        if mode == "ais":
            measurements.append((ais, "AIS"))
            label = "Radar + Camera + AIS"
        else:
            label = "Radar + Camera"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    compute_nis(innov, S)
    compute_rmse(x_est, gt)

    if show_plots:
        plot_trajectories(gt, [x_est], [label], measurements=measurements)
        plot_position_error(gt, [x_est], [label])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run harbour surveillance EKF simulations.")
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C"],
        default="C",
        help="Scenario to run. Scenario C includes AIS fusion.",
    )
    parser.add_argument(
        "--mode",
        choices=["radar", "sequential", "joint", "ais"],
        default="ais",
        help="Fusion mode. A uses radar, B uses radar/sequential/joint, C uses ais.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run metrics without opening Matplotlib figures.",
    )
    args = parser.parse_args()

    run_demo(scenario=args.scenario, mode=args.mode, show_plots=not args.no_plots)
