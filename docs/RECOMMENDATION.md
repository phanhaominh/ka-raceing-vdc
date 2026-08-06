# Robust VDC Calibration Study — KIT25e Reference Model

## Purpose

This document summarizes the best-performing VDC calibration identified using the HPC-based optimization framework developed for the KIT25e reference vehicle. The values represent candidate calibration regions obtained from simulation and should be validated through vehicle testing before deployment.

---

## Key Finding: The Speed-Consistency Trade-Off

The 1.4M physics simulations reveal a genuine trade-off between lap time and consistency:

| Setup | yaw_damping | Mean Lap | Consistency (σ) | Best For |
|-------|-------------|----------|-----------------|----------|
| Aggressive | 0-5 | 41.11-41.13s | ±0.017-0.020s | Qualifying |
| **Balanced** | **10-20** | **41.15-41.16s** | **±0.014s** | **Endurance** |
| Conservative | 50-60 | 41.20-41.26s | ±0.003-0.020s | Reliability focus |

---

## Candidate Calibration Region

| Parameter | Value | Explored Range | Notes |
|-----------|-------|----------------|-------|
| yaw_gain | ~510 | 500–2000 | Consistent across all setups |
| yaw_damping | 10-20 | 0–60 | **The key trade-off parameter** |
| tc_slip_target | ~0.08 | 0.05–0.20 | Limited influence |
| tc_aggressiveness | ~0.17 | 0.1–1.0 | Limited influence |
| regen_ratio | ~0.97 | 0–1 | Minor energy effect |
| brake_bias | ~0.58 | 0.55–0.70 | Limited influence |

---

## Parameter Sensitivity

### Yaw Control

Yaw-related parameters dominated performance within the explored autocross simulation:

- yaw_gain explained 96.1% of yaw-error variance
- yaw_damping explained 92.6% of lap-time variance

The optimal yaw_damping value reached the upper search boundary in earlier surrogate analysis, but the full physics dataset reveals the boundary recommendation was an extrapolation artifact. The real optimum is at moderate damping levels.

### Longitudinal Control

Traction control, braking, and energy-management parameters showed limited influence in the current lateral-dynamics-focused simulation. Their importance may increase for endurance events, acceleration zones, heavy braking sections, or different track layouts.

---

## Methodology

Optimization pipeline:

- 50,000 Latin Hypercube physics simulations (Stage 1)
- 1,230,000+ robustness simulations across 7 uncertainty scenarios (Stage 2)
- 1.1 billion surrogate model evaluations
- Sobol sensitivity analysis with 5,000 bootstrap resamples
- Gaussian Process with Automatic Relevance Detection
- Definitive neural network surrogate trained on full Stage 2 dataset (175K+ samples)

Validation against public FSG 2025 performance data:

- Skidpad: 5.20s simulated vs 5.14s public result (+1.2%)
- Acceleration: 3.76s simulated vs 3.57s public result (+5.3%)

---

## Robustness Objective

The selected calibration was not optimized for minimum lap time under ideal conditions only. The objective minimized:

```
J = mean(lap time) + λ × variance(lap time)
```

across simulated variations in tire grip, driver input consistency, and track conditions.

---

## Computational Resources

Experiments were performed on a shared academic HPC cluster using 792 CPU cores with Slurm-based parallel execution and a containerized simulation environment. The framework remains compatible with local execution.

---

## Scope and Assumptions

The current study uses publicly available KIT25e specifications, MF5.0 Hoosier R20 tire data, estimated aerodynamic parameters, and a representative autocross simulation. No KA-RaceIng telemetry or internal vehicle data were used.

With team-specific data, this framework can be extended for vehicle-specific calibration and validation.

## Endurance Validation

An 11-lap, 24.9 km endurance simulation with tire degradation (grip 1.0→0.82) and battery state-of-charge tracking was run to evaluate whether TC parameters become influential over longer distances. The autocross calibration (tc_slip_target=0.12) and an endurance-tuned calibration (tc_slip_target=0.16) produced identical results — TC never engaged in either case. Peak longitudinal slip reached 0.076 on degraded tires, below the 0.12 threshold.

This confirms that the KIT25e's 4WD system with Hoosier R20 slicks provides sufficient mechanical grip that traction control is unnecessary for autocross and endurance events on the simulated track. Tuning effort should focus exclusively on the yaw controller.
