"""Run ONE robustness sample (7 scenarios) and append to CSV. Called from Slurm."""
import sys, os, csv, numpy as np

j = int(sys.argv[1])
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/robustness"

sys.path.insert(0, os.getcwd())
from src.sweeps.sensitivity import run_lap

samples = np.load(os.path.join(out_dir, "robust_samples.npy"))
param_names = ["yaw_gain","yaw_damping","tc_slip_target",
               "tc_aggressiveness","regen_ratio","brake_bias_front"]

row = samples[j]
params = dict(zip(param_names, row))

SCENARIOS = [
    ("nominal",    1.00, 0.0),
    ("low_grip",   0.85, 0.0),
    ("high_grip",  1.15, 0.0),
    ("cold_tires", 0.90, 0.0),
    ("driver_a",   1.00, 0.03),
    ("driver_b",   1.00, 0.03),
    ("driver_c",   1.00, 0.03),
]

row_data = {"sample_id": j}
scenario_laps = []
scenario_crashes = []
for s_name, grip, noise in SCENARIOS:
    r = run_lap(params, grip_scale=grip, driver_noise=noise)
    row_data[f"lap_{s_name}"] = r["lap_time"]
    row_data[f"crash_{s_name}"] = int(r["crashed"])
    scenario_laps.append(r["lap_time"])
    scenario_crashes.append(r["crashed"])

valid = [lt for lt, cr in zip(scenario_laps, scenario_crashes) if not cr]
mean_lt = np.mean(valid) if valid else 999.0
std_lt = np.std(valid) if valid else 0.0
p_crash = sum(scenario_crashes) / len(scenario_crashes)
J = mean_lt + 0.5 * std_lt + 10.0 * p_crash

row_data["J"] = J
row_data["mean_lap"] = mean_lt
row_data["std_lap"] = std_lt
row_data["p_crash"] = p_crash

out_path = os.path.join(out_dir, f"robust_{task_id:04d}.csv")
write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0

with open(out_path, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row_data.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row_data)

print(f"Sample {j} done — J={J:.3f}")
