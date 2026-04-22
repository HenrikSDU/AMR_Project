from Target_EKF import Target_EKF
import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
import json
import matplotlib.pyplot as plt
from data_association import associate

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
        while cam_idx < len(ts_camera) and ts_camera[cam_idx] <= t_k:

            zc = zs_camera[cam_idx]
            innov, S = ekf.update_sensor(zc, ["camera"])
            innov_hist.append(innov)
            S_hist.append(S)

            cam_idx += 1

        x_est[k] = ekf.x

    return x_est, innov_hist, S_hist


def run_ekf_joint(ts_radar, ts_camera, zs_radar, zs_camera, x0):

    dt = np.mean(np.diff(ts_radar))

    P0 = np.diag([25, 25, 2500, 2500])

    ekf = Target_EKF(x0, P0, dt=dt)

    N = len(ts_radar)
    x_est = np.zeros((N, 4))
    innov_hist, S_hist = [], []

    cam_idx = 0

    for k in range(N):

        ekf.predict()

        t_k = ts_radar[k]

        # --- Always take radar ---
        z_r = zs_radar[k]

        # --- Check if camera is available at this time ---
        use_camera = False

        if cam_idx < len(ts_camera):
            if np.isclose(ts_camera[cam_idx], t_k, atol=1e-2):
                use_camera = True

        if use_camera:
            z_c = zs_camera[cam_idx]

            # Stack measurement
            z = np.hstack([z_r, z_c])

            innov, S = ekf.update_sensor(z, ["radar", "camera"])

            cam_idx += 1

        else:
            # Radar only
            innov, S = ekf.update_sensor(z_r, ["radar"])

        innov_hist.append(innov)
        S_hist.append(S)

        x_est[k] = ekf.x

    return x_est, innov_hist, S_hist

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

            if mode == "joint":
                x_est, innov_hist, S_hist = run_ekf_joint(ts_radar, ts_camera, zs_radar, zs_camera, x0 = np.array([400, 80, 1.2, 2.2]))
            elif mode == 'radar':
                x_est, innov_hist, S_hist = run_ekf_radar_only(ts_radar, zs_radar, x0 = np.array([400, 80, 1.2, 2.2]))
            else: # sequential
                x_est, innov_hist, S_hist = run_ekf_sequential(ts_radar, ts_camera, zs_radar, zs_camera, x0 = np.array([400, 80, 1.2, 2.2]))
            
            gt_interp = interpolate_gt(gt_times, gt_states, ts_radar)
            
            return x_est, gt_interp, innov_hist, S_hist, radar, camera

        case "C":
            pass

    compute_rmse(x_est, gt_interp)

    return x_est, gt_interp, innov_hist, S_hist


# PLOTTING

def plot_trajectories(gt, results, labels, measurements=None):

    plt.figure(figsize=(8, 6))

    plt.plot(gt[:, 0], gt[:, 1], 'k--', linewidth=2, label="Ground Truth")

    for x_est, label in zip(results, labels):
        plt.plot(x_est[:, 0], x_est[:, 1], linewidth=1.5, label=label)

    # Plot measurements if provided
    if measurements is not None:
        if isinstance(measurements, list):
            # Handle multiple measurement types (radar, camera, etc.)
            for meas, meas_label in measurements:
                # Convert polar to Cartesian coordinates
                meas_cart = np.column_stack([
                    meas[:, 1] * np.cos(meas[:, 2]),
                    meas[:, 1] * np.sin(meas[:, 2])
                ])
                plt.scatter(meas_cart[:, 0], meas_cart[:, 1], alpha=0.5, s=20, label=meas_label)
        else:
            # Single measurement array
            meas_cart = np.column_stack([
                measurements[:, 1] * np.cos(measurements[:, 2]),
                measurements[:, 1] * np.sin(measurements[:, 2])
            ])
            plt.scatter(meas_cart[:, 0], meas_cart[:, 1], alpha=0.5, s=20, label="Measurements")

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
    lower = chi2.ppf(0.025, df=nz)
    upper = chi2.ppf(0.975, df=nz)

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


# SCENARIO A
xA, gtA, innovA, SA, meas = sim_tracking("harbour_sim_output\scenario_A.json", scenario="A")
nisA = compute_nis(innovA, SA)
rmseA = compute_rmse(xA, gtA)

# # SCENARIO B sequential harbour_sim_output\scenario_B.json
xB, gtB, innovB, SB, radar, camera = sim_tracking("harbour_sim_output\scenario_B.json", scenario="B", mode='radar')
nisB = compute_nis(innovB, SB)
rmseB = compute_rmse(xB, gtB)

xB_seq, gtB, innovB_seq, SB_seq, radar, camera = sim_tracking("harbour_sim_output\scenario_B.json", scenario="B", mode='sequential')
nisB_seq = compute_nis(innovB_seq, SB_seq)
rmseB_seq = compute_rmse(xB_seq, gtB)

# # SCENARIO B centralized
xB_joint, gtB, innovB_j, SB_j, radar, camera = sim_tracking("harbour_sim_output\scenario_B.json", scenario="B", mode='joint')
nisB_j = compute_nis(innovB_j, SB_j)
rmseB_j = compute_rmse(xB_joint, gtB)

# plot_trajectories(
#     gtA,
#     [xA],
#     ["Radar-only"]
# )

# plot_position_error(
#     gtA,
#     [xA],
#     ["Radar-only"]
# )

plot_trajectories(
    gtB,
    [xB],
    ["Radar-only"],
    measurements=[
        (radar, "Radar")
    ]
)

plot_trajectories(
    gtB,
    [xB_joint],
    ["Joint Fusion"],
    measurements=[
        (radar, "Radar"),
        (camera, "Camera")
    ]
)

plot_trajectories(
    gtB,
    [xB_seq],
    ["Sequential Fusion"],
    measurements=[
        (radar, "Radar"),
        (camera, "Camera")
    ]
)

# plot_position_error(
#     gtB,
#     [xB_seq],
#     ["Sequential"]
# )

# plot_nis(nisA, "Radar-only")
# plot_nis(nisB_seq, "Sequential Fusion")
#plot_nis(nisB_j, "Joint Fusion")