import numpy as np, pandas as pd, time
from scipy.linalg import solve_triangular

print("=== FULL GP 50K — FIXED SOLVE ===")

df = pd.read_csv("results/robustness/all_results.csv")
samples = np.load("results/robustness/robust_samples.npy")
params = ["yaw_gain","yaw_damping","tc_slip_target","tc_aggressiveness","regen_ratio","brake_bias_front"]
for i, name in enumerate(params):
    df[f"param_{name}"] = df['sample_id'].apply(lambda idx: samples[int(idx), i])

n_gp = 50000
idx = np.random.choice(len(df), n_gp, replace=False)
X = df.iloc[idx][[f"param_{n}" for n in params]].values.astype(np.float64)
y = df.iloc[idx]["mean_lap"].values.astype(np.float64)

X_mean, X_std = X.mean(0), X.std(0) + 1e-10
y_mean, y_std = y.mean(), y.std() + 1e-10
X_n = (X - X_mean) / X_std
y_n = (y - y_mean) / y_std

print("Building kernel...")
t0 = time.time()
lengthscales = np.ones(6) * 2.0
sqdist = np.zeros((n_gp, n_gp), dtype=np.float64)
for d in range(6):
    col = X_n[:, d].reshape(-1, 1)
    sqdist += (col - col.T)**2 / lengthscales[d]**2
K = np.exp(-0.5 * sqdist) + 1e-6 * np.eye(n_gp)
print(f"  Kernel: {time.time()-t0:.0f}s, {K.nbytes/1e9:.1f}GB")

print("Cholesky...")
t0 = time.time()
L = np.linalg.cholesky(K)
print(f"  Cholesky: {time.time()-t0:.0f}s")

print("Solving (triangular)...")
t0 = time.time()
z = solve_triangular(L, y_n, lower=True)
alpha = solve_triangular(L.T, z, lower=False)
print(f"  Solve: {time.time()-t0:.1f}s")

np.savez_compressed("models/full_gp_50k.npz",
    X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std,
    alpha=alpha, lengthscales=lengthscales, params=params)
print("Saved: models/full_gp_50k.npz")

x_test = np.array([1500, 15, 0.08, 0.17, 0.97, 0.58])
x_n = (x_test - X_mean) / X_std
k_star = np.zeros(n_gp)
for d in range(6):
    k_star += (x_n[d] - X_n[:, d])**2 / lengthscales[d]**2
k_star = np.exp(-0.5 * k_star)
pred_mean = np.dot(k_star, alpha) * y_std + y_mean
v = solve_triangular(L, k_star, lower=True)
pred_std = np.sqrt(max(1.0 - np.dot(v, v), 0)) * y_std
print(f"\nyaw_g=1500 yaw_d=15: lap = {pred_mean:.3f}s ± {pred_std*2:.4f}s (95% CI)")
