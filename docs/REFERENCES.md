
## Full Literature Review

### Core Methodology
1. **Andersson et al. (2018)** - CasADi: a software framework for nonlinear optimization and optimal control. *Mathematical Programming Computation*. 4,091 citations. The optimization framework we use for minimum-lap-time problems.

2. **Yin et al. (IEEE 2025)** - Optimization of FSAE Vehicle Dimensional Parameters Based on 3-DOF Double-Track Model and Optimal Control. Validates 3-DOF model for FSAE lap time prediction with 10% validation error.

3. **Doyle et al. (SAE 2019)** - Lap Time Simulation Tool for the Development of an Electric Formula Student Car. Queen's University Belfast FSAE team. Identifies resource allocation trade-offs through simulation.

### Optimal Control for Racing
4. **Perantoni & Limebeer (2014)** - Optimal control for a Formula One car with variable parameters. 150 citations. The foundational paper for minimum-lap-time optimal control.

5. **West & Limebeer (2020)** - Optimal tyre management for a high-performance race car. Extends optimal control to tire wear and temperature management.

6. **Veneri & Massaro (2020)** - The effect of Ackermann steering on the performance of race cars. 34 citations. Validates steering geometry optimization.

### VDC and State Estimation for FSAE
7. **PoliTo MSc Thesis (2024)** - Model-based vehicle dynamics control system and states estimation for 4WD Formula SAE electric vehicle. Real track testing with dSpace + Kistler sensors.

### Related Lap Time Simulation
8. **Veneri & Massaro (2019)** - A free-trajectory quasi-steady-state optimal-control method for minimum lap-time of race vehicles. 48 citations.

9. **Lovato & Massaro (2021)** - Three-dimensional fixed-trajectory approaches to the minimum-lap time of road vehicles. 20 citations.

10. **Dal Bianco et al. (2018)** - Lap time simulation and design optimisation of a brushed DC electric motorcycle for the Isle of Man TT Zero Challenge.

11. **Heilmeier et al. (2019)** - A Quasi-Steady-State Lap Time Simulation for Electrified Race Cars.

12. **Liu et al. (2022)** - Energy-optimal overtaking manoeuvres of Formula-E cars.

13. **Kobayashi et al. (2022)** - Hybrid Systems for Small Race Car to Improve Dynamic Performance Using Lap Time Simulation.

### Optimization Methods
14. **de Buck & Martins (2022)** - Minimum lap time trajectory optimisation of performance vehicles with four-wheel drive and active aerodynamic control. 17 citations.

15. **Bartali et al. (2024)** - Schwarz decomposition for parallel minimum lap-time problems.

---

## How Our Work Extends These

All of the above papers use one of two approaches:
- **Optimal control** (CasADi + IPOPT) to find the single fastest lap
- **Parameter sweeps** (dozens to hundreds of simulations) to explore design trade-offs

Our work combines both at unprecedented scale:
- 1.45 million physics simulations (vs dozens to hundreds in prior work)
- Full VDC controller optimization (vs single design point optimization)
- Robustness across 7 uncertainty scenarios (vs nominal conditions only)
- Statistical sensitivity analysis (Sobol + GP ARD) (vs single-point comparisons)
- Open-source, reproducible framework (vs closed-source academic code)
