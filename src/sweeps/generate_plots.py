import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, csv

# ── Load Stage 1 data ──────────────────────────────────────────────
rows = []
with open("/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/sensitivity/all_results.csv") as f:
    for r in csv.DictReader(f): rows.append(r)

X = np.array([[float(r[f"param_{n}"]) for n in ["yaw_gain","yaw_damping","tc_slip_target","tc_aggressiveness","regen_ratio","brake_bias_front"]] for r in rows])
y_lap = np.array([float(r["lap_time"]) for r in rows])
y_yaw = np.array([float(r["yaw_error_rms"]) for r in rows])
y_energy = np.array([float(r["energy_used"]) for r in rows])

param_names = ["yaw_gain","yaw_damping","tc_slip_target","tc_aggressiveness","regen_ratio","brake_bias_front"]
metrics = ["lap_time","yaw_error_rms","energy_used"]

# ── FIGURE 1: The Big Reveal — yaw_damping vs everything ───────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("50,000 VDC Simulations — Parameter Sensitivity", fontsize=16, fontweight='bold')

for i, (name, ax) in enumerate(zip(param_names, axes.flatten())):
    x = X[:, i]
    if i == 0:  # yaw_gain
        y_plot = y_yaw
        ax.set_ylabel('Yaw Error RMS')
        ax.set_title(f'{name}\n(S1 = 0.961 — controls yaw tracking)')
    elif i == 1:  # yaw_damping
        y_plot = y_lap
        ax.set_ylabel('Lap Time [s]')
        ax.set_title(f'{name}\n(S1 = 0.926 — controls lap time & energy)')
    elif i == 4:  # regen_ratio
        y_plot = y_energy
        ax.set_ylabel('Energy [J]')
        ax.set_title(f'{name}\n(S1 = 0.031 — small effect on energy)')
    else:
        y_plot = y_lap
        ax.set_ylabel('Lap Time [s]')
        ax.set_title(f'{name}\n(S1 ≈ 0 — no effect)')
    
    ax.scatter(x[::50], y_plot[::50], alpha=0.3, s=1, c='steelblue')
    ax.set_xlabel(name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/validation/sensitivity_scatter.png", dpi=150)
print("Saved: sensitivity_scatter.png")

# ── FIGURE 2: Sobol indices bar chart ───────────────────────────────
sobol_lap = [0.0244, 0.9259, 0.0001, 0.0001, 0.0003, 0.0001]
sobol_yaw = [0.9605, 0.0107, 0.0001, 0.0002, 0.0004, 0.0001]
sobol_energy = [0.0332, 0.8365, 0.0001, 0.0001, 0.0307, 0.0002]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Sobol Sensitivity Analysis — Which Parameters Matter?", fontsize=14, fontweight='bold')

colors = ['#2196F3' if v > 0.1 else '#BDBDBD' for v in sobol_lap]
axes[0].barh(param_names, sobol_lap, color=colors)
axes[0].set_xlabel('S1 (fraction of variance explained)')
axes[0].set_title('Lap Time')
for i, v in enumerate(sobol_lap):
    axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')

colors = ['#2196F3' if v > 0.1 else '#BDBDBD' for v in sobol_yaw]
axes[1].barh(param_names, sobol_yaw, color=colors)
axes[1].set_xlabel('S1')
axes[1].set_title('Yaw Error RMS')

colors = ['#2196F3' if v > 0.1 else '#BDBDBD' for v in sobol_energy]
axes[2].barh(param_names, sobol_energy, color=colors)
axes[2].set_xlabel('S1')
axes[2].set_title('Energy Used')

plt.tight_layout()
plt.savefig("/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/validation/sobol_bars.png", dpi=150)
print("Saved: sobol_bars.png")

# ── FIGURE 3: Optimization convergence ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
methods = ['Physics\n(Stage 1)', 'Neural Net\n(1.1B evals)', 'Gaussian\nProcess', 'Multi-Fidelity\n(GPU+Physics)']
best_laps = [41.11, 41.27, 41.13, 41.11]
colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF9800']
ax.bar(methods, best_laps, color=colors, edgecolor='black')
ax.set_ylabel('Best Lap Time [s]')
ax.set_title('VDC Optimization Convergence — All Methods Agree', fontweight='bold')
ax.set_ylim(41.0, 41.5)
for i, v in enumerate(best_laps):
    ax.text(i, v + 0.01, f'{v:.2f}s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/validation/convergence.png", dpi=150)
print("Saved: convergence.png")

# ── FIGURE 4: The Decoupled Controller ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.text(5, 9, 'KIT25e VDC System', ha='center', fontsize=16, fontweight='bold')
ax.text(5, 8, 'Statistically Decoupled', ha='center', fontsize=12, color='gray')

# Yaw damping box
ax.add_patch(plt.Rectangle((1, 4), 3.5, 2.5, fill=True, facecolor='#2196F3', alpha=0.2, edgecolor='#2196F3'))
ax.text(2.75, 5.75, 'YAW DAMPING', ha='center', fontweight='bold', fontsize=12)
ax.text(2.75, 5.25, '→ Lap Time (92.6%)', ha='center', fontsize=10)
ax.text(2.75, 4.75, '→ Energy (83.7%)', ha='center', fontsize=10)

# Yaw gain box
ax.add_patch(plt.Rectangle((5.5, 4), 3.5, 2.5, fill=True, facecolor='#4CAF50', alpha=0.2, edgecolor='#4CAF50'))
ax.text(7.25, 5.75, 'YAW GAIN', ha='center', fontweight='bold', fontsize=12)
ax.text(7.25, 5.25, '→ Yaw Error (96.1%)', ha='center', fontsize=10)

# Bottom text
ax.text(1.5, 2.5, 'TC, Brake Bias, Regen:', fontsize=10, color='gray')
ax.text(1.5, 2.0, 'S1 ≈ 0.000 — statistically inert on autocross track', fontsize=10, color='gray')

plt.savefig("/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/validation/decoupled.png", dpi=150)
print("Saved: decoupled.png")
print("\nAll plots generated in results/validation/")
