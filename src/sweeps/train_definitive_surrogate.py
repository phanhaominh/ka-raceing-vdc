"""Train definitive VDC surrogate on full 200K Stage 2 dataset."""
import torch, numpy as np, pandas as pd

print("=== DEFINITIVE VDC SURROGATE ===")
print("Loading full Stage 2 dataset...")

df = pd.read_csv("results/robustness/all_results.csv")
samples = np.load("results/robustness/robust_samples.npy")

param_names = ["yaw_gain","yaw_damping","tc_slip_target","tc_aggressiveness","regen_ratio","brake_bias_front"]
for i, name in enumerate(param_names):
    df[f"param_{name}"] = df['sample_id'].apply(lambda idx: samples[int(idx), i])

X = torch.tensor(df[[f"param_{n}" for n in param_names]].values, dtype=torch.float32)
y_mean = torch.tensor(df['mean_lap'].values, dtype=torch.float32).reshape(-1, 1)
y_std  = torch.tensor(df['std_lap'].values, dtype=torch.float32).reshape(-1, 1)
y_J    = torch.tensor(df['J'].values, dtype=torch.float32).reshape(-1, 1)

# Normalize
X_mean, X_std = X.mean(0), X.std(0)
y_mean_m, y_mean_s = y_mean.mean(), y_mean.std()
y_std_m, y_std_s = y_std.mean(), y_std.std()
X_n = (X - X_mean) / X_std
y_mean_n = (y_mean - y_mean_m) / y_mean_s
y_std_n = (y_std - y_std_m) / y_std_s

# Train/test split
n = len(X_n)
idx = torch.randperm(n)
n_train = int(n * 0.8)
X_train, X_test = X_n[idx[:n_train]].cuda(), X_n[idx[n_train:]].cuda()
y_train_mean, y_test_mean = y_mean_n[idx[:n_train]].cuda(), y_mean_n[idx[n_train:]].cuda()
y_train_std, y_test_std = y_std_n[idx[:n_train]].cuda(), y_std_n[idx[n_train:]].cuda()

# Model: predicts both mean and std simultaneously
model = torch.nn.Sequential(
    torch.nn.Linear(6, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, 2)  # [mean_lap, std_lap]
).cuda()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Train
for epoch in range(3000):
    opt.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred[:, 0], y_train_mean.squeeze()) + loss_fn(pred[:, 1], y_train_std.squeeze())
    loss.backward()
    opt.step()
    if epoch % 500 == 0:
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred[:, 0], y_test_mean.squeeze()) + loss_fn(test_pred[:, 1], y_test_std.squeeze())
        print(f"  Epoch {epoch}: train_loss={loss.item():.4f}  test_loss={test_loss.item():.4f}")

# Save
torch.save({
    'model': model.state_dict(),
    'X_mean': X_mean, 'X_std': X_std,
    'y_mean_m': y_mean_m, 'y_mean_s': y_mean_s,
    'y_std_m': y_std_m, 'y_std_s': y_std_s,
    'param_names': param_names,
}, "models/vdc_surrogate_definitive.pt")

# Quick eval
model.eval()
with torch.no_grad():
    # Test prediction
    x_test_n = (torch.tensor([[1500, 10, 0.12, 0.5, 0.8, 0.6]]) - X_mean) / X_std
    p = model(x_test_n.cuda())
    pred_mean = p[0,0].item() * y_mean_s.item() + y_mean_m.item()
    pred_std = p[0,1].item() * y_std_s.item() + y_std_m.item()
    print(f"\nTest: yaw_g=1500 yaw_d=10 → predicted mean_lap={pred_mean:.3f}s, std={pred_std:.4f}s")
    print(f"Model saved: models/vdc_surrogate_definitive.pt")
    print(f"Trained on {len(X_train)} samples, validated on {len(X_test)} samples")
