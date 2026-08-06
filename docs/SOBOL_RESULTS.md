# Sobol Sensitivity Analysis — VDC Parameter Influence in KIT25e Reference Model

## Method

First-order Sobol sensitivity analysis was performed on 50,000 Latin Hypercube simulation samples with 5,000 bootstrap resamples. The analysis was performed on shared HPC infrastructure to accelerate large-scale simulation and statistical evaluation.

The Sobol indices quantify variance contribution within the defined simulation parameter ranges. They do not imply that the parameter explains real-world vehicle performance outside the modeled conditions. First-order indices are reported; interaction effects were not included in this analysis.

## Results

### Lap Time Variance

| Parameter | S1 | Contribution within evaluated parameter space |
|-----------|-----|----------------------------------------------|
| yaw_damping | 0.926 | Dominant influence |
| yaw_gain | 0.024 | Minor influence |
| tc_slip_target | 0.0001 | Limited influence in current simulation |
| tc_aggressiveness | 0.0001 | Limited influence in current simulation |
| regen_ratio | 0.0003 | Limited influence in current simulation |
| brake_bias_front | 0.0001 | Limited influence in current simulation |

### Yaw Error Variance

| Parameter | S1 | Contribution within evaluated parameter space |
|-----------|-----|----------------------------------------------|
| yaw_gain | 0.961 | Dominant influence |
| yaw_damping | 0.011 | Limited influence |

### Energy Variance

| Parameter | S1 | Contribution within evaluated parameter space |
|-----------|-----|----------------------------------------------|
| yaw_damping | 0.837 | Dominant influence |
| yaw_gain | 0.033 | Minor influence |
| regen_ratio | 0.031 | Minor influence |

## Conclusion

The simulated VDC parameter space shows strong statistical decoupling under the evaluated autocross scenario:

- yaw_damping primarily affects simulated lap time and energy consumption
- yaw_gain primarily affects yaw tracking behavior
- Longitudinal VDC parameters showed lower sensitivity in this scenario

These results indicate where future calibration effort may be concentrated, rather than representing direct deployment recommendations without vehicle validation. The strong decoupling between yaw_gain (tracking) and yaw_damping (lap time) suggests these parameters can be tuned largely independently within the evaluated ranges.

## Total-Effect Indices (200K samples)

Total-effect Sobol indices (ST) capture both first-order effects and parameter interactions. Computed on the full Stage 2 dataset.

| Parameter | ST | First-Order S1 | Interaction (ST-S1) |
|-----------|-----|----------------|---------------------|
| yaw_damping | 0.933 | 0.926 | +0.007 |
| yaw_gain | 0.027 | 0.024 | +0.003 |
| tc_slip_target | ~0.000 | 0.0001 | ~0.000 |
| tc_aggressiveness | ~0.000 | 0.0001 | ~0.000 |
| regen_ratio | ~0.000 | 0.0003 | ~0.000 |
| brake_bias_front | ~0.000 | 0.0001 | ~0.000 |

The near-zero difference between ST and S1 confirms that VDC parameter interactions are negligible. Parameters can be tuned independently.
