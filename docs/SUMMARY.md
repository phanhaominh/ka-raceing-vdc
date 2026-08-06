# What We Learned — KIT25e Reference Model VDC Optimization

## Core Finding

After 1.45 million physics simulations, independent sensitivity analyses, and surrogate-model evaluations, the KIT25e reference model shows a strong dominance of yaw-control parameters in autocross performance.

The main findings are:

- yaw_damping is the dominant parameter affecting simulated lap time and energy consumption
- yaw_gain primarily affects yaw tracking behavior
- traction control, brake bias, and regen parameters showed lower sensitivity within the evaluated autocross scenario

## The Speed-Consistency Trade-Off

The simulations reveal a genuine trade-off between fast laps and consistent laps in the reference model:

| Setup | yaw_damping | Lap Time | Consistency |
|-------|-------------|----------|-------------|
| Aggressive | 0-5 | 41.11s | ±0.020s |
| Balanced | 10-20 | 41.15s | ±0.014s |
| Conservative | 50-60 | 41.20s | ±0.003s |

For applications prioritizing repeatability, yaw_damping = 10-20 provides a favorable speed-consistency trade-off. The fastest simulated configurations occurred at higher damping values, which reached the upper search boundary and require further investigation beyond the explored range.

## How We Know

Three independent statistical methods point to consistent conclusions:

1. **Sobol sensitivity analysis** (5,000 bootstrap resamples on 50,000 samples): yaw_damping explains 92.6% of lap-time variance within the evaluated parameter space
2. **Gaussian Process with Automatic Relevance Detection**: yaw_damping lengthscale = 0.10 (all other parameters > 2.6)
3. **Pearson correlation**: consistent ranking across all three performance metrics

The initial surrogate model prediction of "max damping = best lap time" was an extrapolation artifact at the search boundary. The full physics dataset from Stage 2 provided the correction.

## Implications for Development

- Yaw-control parameters showed the highest sensitivity in the current autocross simulation and may benefit from prioritization during calibration
- Longitudinal VDC parameters (traction control, brake bias) showed lower sensitivity in the evaluated scenario; their importance may increase for endurance events, acceleration zones, or different track layouts
- The definitive surrogate model can predict lap time and consistency for any candidate calibration in milliseconds

## Methodology Scale

- 50,000 Latin Hypercube samples (Stage 1 sensitivity)
- 1,400,000 robustness laps across 7 grip/driver scenarios (Stage 2)
- 1,100,000,000 virtual surrogate evaluations
- 792 CPU cores, ~6,300 core-hours
- Executed on shared academic HPC infrastructure

## Next Steps

The framework is designed to incorporate team-specific telemetry, tire measurements, and track data for vehicle-specific calibration and validation.
