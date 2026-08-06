# VDC Surrogate Models

## Purpose

The surrogate models accelerate exploration of the VDC parameter space after training on physics-based simulations. They are used for rapid candidate screening, while final calibration candidates are evaluated using the physics simulation framework.

The surrogate model does not replace physics simulation. It enables rapid search, with final candidates validated through the physics model.

---

## Neural Network (PyTorch)

Architecture:

```
6 → 128 → 128 → 128 → 3
```

Inputs: yaw_gain, yaw_damping, tc_slip_target, tc_aggressiveness, regen_ratio, brake_bias

Outputs: lap time, yaw tracking error, energy consumption

Training: 50,000 physics simulations, 4,000 epochs, held-out validation dataset

Performance:
- Lap-time RMSE: ~0.01s on held-out physics simulations
- Yaw-error RMSE: ~0.001

Inference speed: approximately 1M surrogate predictions/second on V100 GPU. These are model predictions, not physics simulations.

Model: `models/vdc_surrogate_v2.pt`

---

## Gaussian Process (GPyTorch)

Kernel: RBF with Automatic Relevance Detection

Training: 10,000 physics samples, exact inference

Performance:
- Lap-time error: ~0.002s on held-out physics simulations
- Native uncertainty estimates through predictive confidence intervals

Sensitivity interpretation (ARD lengthscales):
- yaw_damping: 0.10 (shortest lengthscale, highest sensitivity)
- Other parameters: >2.6 (longer lengthscales, lower sensitivity within the evaluated parameter space)

Model: `models/vdc_gp_surrogate.pt`

---

## Ensemble Validation

Five independently trained neural networks were evaluated to assess prediction consistency.

At the identified optimum:
- Prediction spread: ±0.0002s

The ensemble spread measures disagreement between surrogate models and is used as an indicator of prediction confidence. Agreement does not guarantee correctness; it indicates consistency among the trained surrogates.

Maximum uncertainty occurred near yaw_gain ≈ 744, yaw_damping ≈ 3 — a region corresponding to potentially unstable controller behavior, flagged for additional physics validation.

---

## Definitive Surrogate (Stage 2)

Trained on the full Stage 2 robustness dataset (175K+ samples), this model predicts both mean lap time and lap-time standard deviation from VDC parameters.

Architecture: 6 → 256 → 256 → 128 → 2 (mean_lap, std_lap)

Performance: RMSE ~0.08s on held-out Stage 2 data

Enables rapid evaluation of the speed-consistency trade-off for any candidate calibration.

Model: `models/vdc_surrogate_definitive.pt`
