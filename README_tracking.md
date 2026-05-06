# Harbour Tracking Simulation

This README explains how to run the EKF tracking code in this repository and how to use the new T5 AIS fusion implementation.

## Files

- `sim_tracking.py`: main simulation script for radar, camera, and AIS fusion.
- `Target_EKF.py`: constant-velocity extended Kalman filter.
- `Coordinate_Frame_Manager.py`: sensor measurement models, Jacobians, FOV checks, and noise matrices.
- `harbour_sim_output/scenario_*.json`: simulated harbour surveillance scenarios.
- `sim_tracking_copy.py`: older/reference copy; T5 changes are implemented in `sim_tracking.py`.

## Requirements

Install the Python packages used by the script:

```bash
pip install numpy matplotlib
```

`scipy` is optional. If installed, it is used for exact chi-square NIS plot bounds; otherwise the script falls back to fixed 95% bounds for 2D measurements.

Run commands from the repository root:

```bash
cd C:\Github\AMR_Project
```

## Running Simulations

Run Scenario C without AIS, using asynchronous radar + camera fusion:

```bash
python sim_tracking.py --scenario C --mode sequential
```

Run Scenario C with radar, camera, and AIS fusion:

```bash
python sim_tracking.py --scenario C --mode ais
```

Run without plots, useful for quick RMSE checks:

```bash
python sim_tracking.py --scenario C --mode sequential --no-plots
python sim_tracking.py --scenario C --mode ais --no-plots
```

Compare the printed `Total RMSE` values from those two runs to evaluate the AIS improvement. There is no automatic comparison helper.

Run the earlier scenarios:

```bash
python sim_tracking.py --scenario A --mode radar
python sim_tracking.py --scenario B --mode radar
python sim_tracking.py --scenario B --mode sequential
python sim_tracking.py --scenario B --mode joint
```

## Scenario Meaning

- Scenario A: radar-only baseline.
- Scenario B: radar plus camera fusion. Use `radar`, `sequential`, or `joint` mode.
- Scenario C: asynchronous fusion for T5. Use `sequential` for radar + camera only, or `ais` for radar + camera + AIS.

## AIS Fusion Model

AIS is treated as an absolute Cartesian NED position measurement:

```text
z_ais = [north_m, east_m]
h_ais(x) = [x_N, x_E]
H_ais = [[1, 0, 0, 0],
         [0, 1, 0, 0]]
```

Radar and camera remain polar sensors:

```text
z = [range_m, bearing_rad]
```

The Scenario C runner is asynchronous: it sorts the selected sensor messages by timestamp, predicts the EKF by the time gap since the last message, gates the measurement, and then updates with that one sensor.

## Output

The script prints:

- Accepted update counts for radar, camera, and AIS.
- Rejected update counts from gating/FOV checks.
- RMSE in north/east and total position RMSE.

When plots are enabled, it shows:

- Estimated trajectory against ground truth.
- Radar/camera measurement points converted from polar to N/E.
- AIS measurement points plotted directly in N/E.
- Position error over time.

## Current Assumptions

- T5 is still single-target tracking.
- False alarms are filtered out with `is_false_alarm == false`.
- Scenario C uses `target_id == 0`.
- GNSS is ignored for T5 because simulated AIS measurements already provide absolute N/E target positions.
- Full multi-target data association is left for T6.
