from copy_Target_EKF import Target_EKF
import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
import json
import matplotlib.pyplot as plt

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

def run_ekf_async(all_measurements, coord_manager):
    # 1. Find the first valid target detection (Ignore GNSS, accept Radar/Camera/AIS)
    first_det = next(m for m in all_measurements if m["sensor_id"] != "gnss")
    sensor = first_det["sensor_id"]
    
    # 2. Initialize x0 based on the coordinate system of the sensor
    if sensor in ["radar", "camera"]:
        # Polar to Cartesian conversion
        r0, phi0 = first_det["range_m"], first_det["bearing_rad"]
        x0 = np.array([r0*np.cos(phi0), r0*np.sin(phi0), 0.0, 0.0])
    
    elif sensor == "ais":
        x0 = np.array([first_det["north_m"], first_det["east_m"], 0.0, 0.0])
    
    # initial uncertaint
    P0 = np.diag([25.0, 25.0, 2500.0, 2500.0])
    
    # Init EKF (dt will be provided dynamically in predict)
    ekf = Target_EKF(x0, P0) 

    # 2. Setup dynamic storage
    x_est = []
    ts_est = []
    innov_hist = []
    S_hist = []
    
    last_t = first_det["time"]

    # 3. The Event-Driven Loop
    for m in all_measurements:
        current_t = m["time"]
        sensor = m["sensor_id"]

        # Intercept GNSS: Update the vessel position, do NOT update the EKF
        if sensor == "gnss":
            coord_manager.update_vessel_position(m["north_m"], m["east_m"])
            continue 

        # Calculate dynamic time step
        dt = current_t - last_t
        
        # Only predict if time has actually moved forward
        if dt > 0:
            ekf.predict(dt)
            last_t = current_t

        # Format measurement vector based on sensor type
        if sensor in ["radar", "camera"]:
            z = np.array([m["range_m"], m["bearing_rad"]])
        elif sensor == "ais":
            z = np.array([m["north_m"], m["east_m"]])

        # Update EKF
        innov, S = ekf.update_sensor(z, [sensor])
        
        # Store data
        innov_hist.append(innov)
        S_hist.append(S)
        x_est.append(ekf.x.copy())
        ts_est.append(current_t)

    return np.array(ts_est), np.array(x_est), innov_hist, S_hist

# MAIN SIMULATION FUNCTION

def sim_tracking(json_file, coord_manager, scenario="A"):
    data = json.load(open(json_file, "r", encoding="utf-8"))
    gt_times, gt_states = extract_ground_truth(data)

    print(f"Simulating Scenario {scenario} (Asynchronous)")

    # 1. Pool all valid measurements
    # (Note: we keep the 'not false_alarm' cheat until T6)
    all_measurements = [
        m for m in data["measurements"] 
        if not m["is_false_alarm"]
    ]

    # 2. Sort them strictly by time
    all_measurements.sort(key=lambda x: x["time"])

    # 3. Pass to the universal event loop
    ts_est, x_est, innov_hist, S_hist = run_ekf_async(all_measurements, coord_manager)

    # 4. Interpolate Ground Truth to match your dynamic EKF update times
    gt_interp = interpolate_gt(gt_times, gt_states, ts_est)
    
    compute_rmse(x_est, gt_interp)

    return x_est, gt_interp, innov_hist, S_hist, ts_est


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
xB_seq1, gtB1, innovB_seq1, SB_seq1, radar1, camera1 = sim_tracking("harbour_sim_output\scenario_B.json", scenario="B", mode='radar')
nisB_seq1 = compute_nis(innovB_seq1, SB_seq1)
rmseB_seq1 = compute_rmse(xB_seq1, gtB1)
xB_seq, gtB, innovB_seq, SB_seq, radar, camera = sim_tracking("harbour_sim_output\scenario_B.json", scenario="B", mode='sequential')
nisB_seq = compute_nis(innovB_seq, SB_seq)
rmseB_seq = compute_rmse(xB_seq, gtB)

# # SCENARIO B centralized
# xB_joint, innovB_j, SB_j = sim_tracking("harbour_sim_output\scenario_B.json", scenario="B", mode='joint')
# nisB_j = compute_nis(innovB_j, SB_j)
# rmseB_j = compute_rmse(xB_joint, gtB)

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
    gtB1,
    [xB_seq1],
    ["Radar-only"],
    measurements=[
        (radar, "Radar")
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