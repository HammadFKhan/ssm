#%%
import h5py
import numpy as np

mat_file_path = r"D:\BayesianWaveModel\M2Waves\NoTagSOM_Day4GridWaves.mat"

mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
output_dir = os.path.join(r"D:\BayesianWaveModel\model_output", mat_basename)
os.makedirs(output_dir, exist_ok=True)

with h5py.File(mat_file_path, 'r') as f:
    print(list(f.keys()))
    intan_behaviour = f['IntanBehaviour']

    # ---- lever traces (hit trials) ----
    hit = intan_behaviour['cueHitTrace']
    hit_trace = hit['trace']          # object references to each trial
    hit_refs = hit_trace[:].flatten()
    lever_traces = [f[ref][:] for ref in hit_refs]
    lever_traces = np.squeeze(np.stack(lever_traces, axis=0))  # (n_trials, T)

    print("lever_traces shape:", lever_traces.shape)

    # ---- wave struct for hit trials: dx, dy ----
    # adjust the key name here to match exactly what is in the file,
    # e.g. 'WavesHit', 'wavesHit', etc.
    waves_hit = f['WavesHit']

    # Each of these is an array of object references, one per trial
    dx_refs = waves_hit['dx'][:]      # shape (n_trials, 1) of refs
    dy_refs = waves_hit['dy'][:]

    dx_list = [f[ref][:] for ref in dx_refs.flatten()]
    dy_list = [f[ref][:] for ref in dy_refs.flatten()]

    # Assuming each dx/dy is (8, 8, 3001), stack into (n_trials, 8, 8, 3001)
    dx_array = np.stack(dx_list, axis=0)
    dy_array = np.stack(dy_list, axis=0)

    print("dx_array shape:", dx_array.shape)
    print("dy_array shape:", dy_array.shape)
# after constructing dx_array, dy_array with shape (n_trials, 3001, 8, 8)
dx_array = np.transpose(dx_array, (0, 1, 3, 2))  # (n_trials, 3001, 8, 8) with axes swapped
dy_array = np.transpose(dy_array, (0, 1, 3, 2))

# %%
import matplotlib.pyplot as plt
import numpy as np

# choose a trial and time index to inspect
trial_idx = 0          # 0..116
t_idx = 1500           # 0..3000, e.g. around movement

dx_trial = dx_array[trial_idx]   # (3001, 8, 8)
dy_trial = dy_array[trial_idx]

dx_frame = np.mean(dx_trial,0)       # (8, 8)
dy_frame = np.mean(dy_trial,0)       # (8, 8)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

im0 = axes[0].imshow(dx_frame, cmap='viridis', origin='upper')
axes[0].set_title(f'dx trial {trial_idx}, t={t_idx}')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(dy_frame, cmap='viridis', origin='upper')
axes[1].set_title(f'dy trial {trial_idx}, t={t_idx}')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()

# %% 
# lever_traces: (n_trials, T)
# dx_array, dy_array: (n_trials, T, 8, 8)

T = lever_traces.shape[1]

# Example: mean dx, dy over all 8x8 channels
dx_mean = dx_array.mean(axis=(2, 3))   # (n_trials, T)
dy_mean = dy_array.mean(axis=(2, 3))   # (n_trials, T)

# Stack into observation features y_t = [dx_mean, dy_mean]
Y = np.stack([dx_mean, dy_mean], axis=-1)   # (n_trials, T, K=2)

# Center features over time per trial (optional)
Y = Y - Y.mean(axis=1, keepdims=True)

# State we want to decode: lever position; approximate velocity by finite diff
pos = lever_traces
vel = np.diff(pos, axis=1, prepend=pos[:, :1])   # simple derivative
X_true = np.stack([pos, vel], axis=-1)           # (n_trials, T, 2)
print("Y shape:", Y.shape)
print("X_true shape:", X_true.shape)

# %% We build a cnn model to decode lever position from wave data

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# lever_traces: (n_trials, T)
# dx_array, dy_array: (n_trials, T, 8, 8)  # after transpose fix

n_trials, T = lever_traces.shape

# Build input X: (n_trials, T, 8, 8, 2)
X = np.stack([dx_array, dy_array], axis=-1)    # (n_trials, T, 8, 8, 2)

# Flatten spatial dims -> (n_trials, T, 128)
X = X.reshape(n_trials, T, -1).astype(np.float32)

# Target lever -> (n_trials, T, 1)
Y = lever_traces[..., None].astype(np.float32)

# Train/test split by trial (same as before)
idx = np.arange(n_trials)
np.random.shuffle(idx)
n_train = int(0.8 * n_trials)
train_idx = idx[:n_train]
test_idx  = idx[n_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]

# ---- standardize X over all training trials and time ----
X_mean = X_train.mean(axis=(0, 1), keepdims=True)         # (1,1,128)
X_std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

X_train_norm = (X_train - X_mean) / X_std
X_test_norm  = (X_test  - X_mean) / X_std

# ---- standardize Y (lever) ----
Y_mean = Y_train.mean()
Y_std  = Y_train.std() + 1e-6

Y_train_norm = (Y_train - Y_mean) / Y_std
Y_test_norm  = (Y_test  - Y_mean) / Y_std

class PhaseLeverDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = PhaseLeverDataset(X_train_norm, Y_train_norm)
test_ds  = PhaseLeverDataset(X_test_norm,  Y_test_norm)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)
# %% Define a small causal TCN
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]
    def forward(self, x):
        # x: (B, C_in, T)
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)

class TCNDecoder(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, n_layers=3, kernel_size=5):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i in range(n_layers):
            dilation = 2 ** i
            conv = CausalConv1d(ch_in, hidden_channels, kernel_size,
                                dilation=dilation)
            layers += [conv, nn.ReLU(), nn.Dropout(0.2)]
            ch_in = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.readout = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, T, 128) -> (B, 128, T)
        x = x.transpose(1, 2)
        h = self.tcn(x)                # (B, hidden, T)
        y = self.readout(h)            # (B, 1, T)
        y = y.transpose(1, 2)          # (B, T, 1)
        return y
# %% Train the TCN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Example: smaller hidden size, more layers, larger kernel
hidden_channels = 32
n_layers = 4
kernel_size = 9
model = TCNDecoder(
    in_channels=X.shape[-1],   # 128
    hidden_channels=hidden_channels,        # try 32, 64, 128
    n_layers=n_layers,                # try 2–4
    kernel_size=kernel_size              # try 3, 5, 7
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-3)
criterion = nn.MSELoss()
smooth_lambda = 1e-1  # tune: e.g. 1e-3, 1e-2, 1e-1

n_epochs = 90

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for Xb, Yb in train_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        optimizer.zero_grad()
        Yhat = model(Xb)
        mse_loss = criterion(Yhat, Yb)
        diff = Yhat[:, 1:, 0] - Yhat[:, :-1, 0]
        smooth_loss = (diff**2).mean()
        loss = mse_loss + smooth_lambda * smooth_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(train_ds)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Yhat = model(Xb)
            loss = criterion(Yhat, Yb)
            test_loss += loss.item() * Xb.size(0)
    test_loss /= len(test_ds)

    print(f"Epoch {epoch+1:03d}  train MSE={train_loss:.4f}  test MSE={test_loss:.4f}")
# %% Plot some decoded lever traces from TCN
# ---------- 1. Rebuild feature tensors ----------
# full spatial features: (n_trials, T, 128)
X_full = np.stack([dx_array, dy_array], axis=-1)          # (n_trials, T, 8, 8, 2)
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], -1).astype(np.float32)

# mean features: (n_trials, T, 2)
dx_mean = dx_array.mean(axis=(2, 3))
dy_mean = dy_array.mean(axis=(2, 3))
X_mean = np.stack([dx_mean, dy_mean], axis=-1).astype(np.float32)

Y = lever_traces[..., None].astype(np.float32)            # (n_trials, T, 1)

# ---------- 2. Train/test split ----------
idx = np.arange(X_full.shape[0])
# use same test_idx as before; assume it already exists
train_idx = np.setdiff1d(idx, test_idx)

X_full_tr, X_full_te = X_full[train_idx], X_full[test_idx]
X_mean_tr, X_mean_te = X_mean[train_idx], X_mean[test_idx]
Y_tr,       Y_te      = Y[train_idx],     Y[test_idx]

# ---------- 3. Normalization (separate for full vs mean) ----------
# full
X_full_mean = X_full_tr.mean(axis=(0,1), keepdims=True)
X_full_std  = X_full_tr.std(axis=(0,1), keepdims=True) + 1e-6

Y_full_mean = Y_tr.mean()
Y_full_std  = Y_tr.std() + 1e-6

# mean
X_mean_mu = X_mean_tr.mean(axis=(0,1), keepdims=True)
X_mean_sd = X_mean_tr.std(axis=(0,1), keepdims=True) + 1e-6

Y_mean_mu = Y_tr.mean()
Y_mean_sd = Y_tr.std() + 1e-6

model.eval()
all_corrs = []
all_rmses = []

with torch.no_grad():
    for tr in test_idx:
        # normalize inputs using TRAIN-only stats
        x = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
        x = torch.from_numpy(x).to(device)

        y_true = Y[tr, :, 0]  # original lever

        y_pred_norm = model(x).cpu().numpy()[0, :, 0]
        # rescale outputs using TRAIN-only stats
        y_pred = y_pred_norm * Y_full_std + Y_full_mean

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        rmse = np.sqrt(((y_true - y_pred)**2).mean())
        all_corrs.append(corr)
        all_rmses.append(rmse)

print("TCN mean test correlation:", np.nanmean(all_corrs))
print("TCN mean test RMSE:", np.nanmean(all_rmses))

#%%  Plot some decoded lever traces from TCN
n_show = 4
trials_to_plot = test_idx[:n_show]
sigma = 1  # in samples; tune
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
from scipy.ndimage import gaussian_filter1d
for ax, tr in zip(axes.ravel(), trials_to_plot):
    # TCN prediction (remember to normalize X and rescale Y)
    x = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        y_pred_norm = model(x).cpu().numpy()[0, :, 0]
    y_pred = y_pred_norm * Y_full_std + Y_full_mean
    y_pred = gaussian_filter1d((y_pred*10), sigma=sigma)/10
    y_true = Y[tr, :, 0]

    ax.plot(y_true, label='True', color='C0')
    ax.plot(y_pred, label='TCN decoded', color='C1', linestyle='--')
    ax.set_title(f'Trial {tr}')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Lever')

axes[0,0].legend()
plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt


# --- TCN decoded lever for each test trial ---
tcn_decoded = []
model.eval()
with torch.no_grad():
    for tr in test_idx:
        x = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
        x = torch.from_numpy(x).to(device)
        y_pred_norm = model(x).cpu().numpy()[0, :, 0]
        y_pred = y_pred_norm * Y_full_std + Y_full_mean  # back to original scale
        y_pred = gaussian_filter1d((y_pred*100), sigma=10)/100
        tcn_decoded.append(y_pred)
tcn_decoded = np.stack(tcn_decoded, axis=0)   # (n_test, T)

# --- True lever traces for test trials ---
true_lever = pos[test_idx]                     # (n_test, T)

# --- Mean across trials ---
mean_true = true_lever.mean(axis=0)
mean_tcn  = tcn_decoded.mean(axis=0)

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(mean_true, label='True', color='k', linewidth=2)
plt.plot(mean_tcn,  label='Phase spatial structure', color='C2', linestyle=':')
plt.xlabel('Time (samples)')
plt.ylabel('Lever (mean across test trials)')
plt.title('Mean lever trajectory prediction based on wave regressors')
plt.legend()
plt.tight_layout()
plt.show()

# %% Train a TCN decoder using only mean phase features (no spatial structure)
# From your existing arrays
dx_mean = dx_array.mean(axis=(2, 3))   # (n_trials, T)
dy_mean = dy_array.mean(axis=(2, 3))   # (n_trials, T)

X_mean = np.stack([dx_mean, dy_mean], axis=-1).astype(np.float32)   # (n_trials, T, 2)
Y = lever_traces[..., None].astype(np.float32)                      # (n_trials, T, 1)

idx = np.arange(X_mean.shape[0])
np.random.shuffle(idx)
n_train = int(0.8 * len(idx))
train_idx = idx[:n_train]
test_idx  = idx[n_train:]

Xtr, Xte = X_mean[train_idx], X_mean[test_idx]
Ytr, Yte = Y[train_idx],      Y[test_idx]

# Normalize over training trials and time
X_mean_mu = Xtr.mean(axis=(0,1), keepdims=True)
X_mean_sd = Xtr.std(axis=(0,1), keepdims=True) + 1e-6
Xtr_n = (Xtr - X_mean_mu)/X_mean_sd
Xte_n = (Xte - X_mean_mu)/X_mean_sd

Y_mu = Ytr.mean()
Y_sd = Ytr.std() + 1e-6
Ytr_n = (Ytr - Y_mu)/Y_sd
Yte_n = (Yte - Y_mu)/Y_sd
# X_mean_n: (n_trials, T, 2), Y_n: (n_trials, T, 1) after normalization
train_ds_mean = PhaseLeverDataset(Xtr_n, Ytr_n)
test_ds_mean  = PhaseLeverDataset(Xte_n, Yte_n)

train_loader_mean = DataLoader(train_ds_mean, batch_size=8, shuffle=True)
test_loader_mean  = DataLoader(test_ds_mean,  batch_size=8, shuffle=False)

hidden_channels = 64
n_layers = 3
kernel_size = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean_tcn = TCNDecoder(
    in_channels=2,
    hidden_channels=hidden_channels,
    n_layers=n_layers,
    kernel_size=kernel_size
).to(device)

optimizer = torch.optim.Adam(mean_tcn.parameters(), lr=1e-3)
criterion = nn.MSELoss()
smooth_lambda = 1e-1
n_epochs = 70

for epoch in range(n_epochs):
    mean_tcn.train()
    train_loss = 0.0
    for Xb, Yb in train_loader_mean:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        optimizer.zero_grad()
        Yhat = mean_tcn(Xb)
        mse_loss = criterion(Yhat, Yb)
        diff = Yhat[:, 1:, 0] - Yhat[:, :-1, 0]
        smooth_loss = (diff**2).mean()
        loss = mse_loss + smooth_lambda * smooth_loss
        loss.backward()
        optimizer.step()
        train_loss += mse_loss.item() * Xb.size(0)
    train_loss /= len(train_ds_mean)

    mean_tcn.eval()
    test_loss = 0.0
    with torch.no_grad():
        for Xb, Yb in test_loader_mean:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Yhat = mean_tcn(Xb)
            loss = criterion(Yhat, Yb)
            test_loss += loss.item() * Xb.size(0)
    test_loss /= len(test_ds_mean)

    print(f"Epoch {epoch+1:03d} train MSE={train_loss:.4f} test MSE={test_loss:.4f}")
# %%
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_full = model    # trained full-feature TCN
model_mean = mean_tcn    # trained mean-feature TCN

# ---------- 1. Rebuild feature tensors ----------
# full spatial features: (n_trials, T, 128)
X_full = np.stack([dx_array, dy_array], axis=-1)          # (n_trials, T, 8, 8, 2)
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], -1).astype(np.float32)

# mean features: (n_trials, T, 2)
dx_mean = dx_array.mean(axis=(2, 3))
dy_mean = dy_array.mean(axis=(2, 3))
X_mean = np.stack([dx_mean, dy_mean], axis=-1).astype(np.float32)

Y = lever_traces[..., None].astype(np.float32)            # (n_trials, T, 1)

# ---------- 2. Train/test split ----------
idx = np.arange(X_full.shape[0])
# use same test_idx as before; assume it already exists
train_idx = np.setdiff1d(idx, test_idx)

X_full_tr, X_full_te = X_full[train_idx], X_full[test_idx]
X_mean_tr, X_mean_te = X_mean[train_idx], X_mean[test_idx]
Y_tr,       Y_te      = Y[train_idx],     Y[test_idx]

# ---------- 3. Normalization (separate for full vs mean) ----------
# full
X_full_mean = X_full_tr.mean(axis=(0,1), keepdims=True)
X_full_std  = X_full_tr.std(axis=(0,1), keepdims=True) + 1e-6

Y_full_mean = Y_tr.mean()
Y_full_std  = Y_tr.std() + 1e-6

# mean
X_mean_mu = X_mean_tr.mean(axis=(0,1), keepdims=True)
X_mean_sd = X_mean_tr.std(axis=(0,1), keepdims=True) + 1e-6

Y_mean_mu = Y_tr.mean()
Y_mean_sd = Y_tr.std() + 1e-6

# ---------- 4. Evaluate both models on test trials ----------
full_corrs, full_rmses = [], []
mean_corrs, mean_rmses = [], []

model_full.eval()
model_mean.eval()

with torch.no_grad():
    for tr_local, tr in enumerate(test_idx):
        # FULL TCN
        x_f = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
        x_f = torch.from_numpy(x_f).to(device)
        y_pred_f_norm = model_full(x_f).cpu().numpy()[0, :, 0]
        y_pred_f = y_pred_f_norm * Y_full_std + Y_full_mean

        # MEAN TCN
        x_m = ((X_mean[tr:tr+1] - X_mean_mu) / X_mean_sd).astype(np.float32)
        x_m = torch.from_numpy(x_m).to(device)
        y_pred_m_norm = model_mean(x_m).cpu().numpy()[0, :, 0]
        y_pred_m = y_pred_m_norm * Y_mean_sd + Y_mean_mu

        # TRUE lever
        y_true = lever_traces[tr]

        # metrics
        full_corrs.append(np.corrcoef(y_true, y_pred_f)[0, 1])
        mean_corrs.append(np.corrcoef(y_true, y_pred_m)[0, 1])

        full_rmses.append(np.sqrt(mean_squared_error(y_true, y_pred_f)))
        mean_rmses.append(np.sqrt(mean_squared_error(y_true, y_pred_m)))

print("Full TCN  mean corr:", np.nanmean(full_corrs), "mean RMSE:", np.nanmean(full_rmses))
print("Mean TCN  mean corr:", np.nanmean(mean_corrs), "mean RMSE:", np.nanmean(mean_rmses))
#%%
import numpy as np
import matplotlib.pyplot as plt

full_corrs = np.array(full_corrs)
mean_corrs = np.array(mean_corrs)

means = [np.nanmean(full_corrs), np.nanmean(mean_corrs)]
sems  = [np.nanstd(full_corrs, ddof=1)/np.sqrt(len(full_corrs)),
         np.nanstd(mean_corrs, ddof=1)/np.sqrt(len(mean_corrs))]

labels = ['Full TCN', 'Mean TCN']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.figure(figsize=(4, 4))
bars = plt.bar(labels, means, yerr=sems, capsize=5, color=['C2', 'C1'], alpha=0.8)
plt.ylabel('Test trial correlation')
plt.title('Decoder performance (r)')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"decoder_correlations.pdf"),
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace

# %%
full_preds = []
mean_preds = []
true_traces = []

with torch.no_grad():
    for tr in test_idx:
        # FULL TCN prediction
        x_f = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
        x_f = torch.from_numpy(x_f).to(device)
        y_f_norm = model_full(x_f).cpu().numpy()[0, :, 0]
        y_f = y_f_norm * Y_full_std + Y_full_mean

        # MEAN TCN prediction
        x_m = ((X_mean[tr:tr+1] - X_mean_mu) / X_mean_sd).astype(np.float32)
        x_m = torch.from_numpy(x_m).to(device)
        y_m_norm = model_mean(x_m).cpu().numpy()[0, :, 0]
        y_m = y_m_norm * Y_mean_sd + Y_mean_mu

        full_preds.append(y_f)
        mean_preds.append(y_m)
        true_traces.append(lever_traces[tr])

full_preds  = np.stack(full_preds,  axis=0)   # (n_test, T)
mean_preds  = np.stack(mean_preds,  axis=0)
true_traces = np.stack(true_traces, axis=0)

mean_true = true_traces.mean(axis=0)
mean_full = full_preds.mean(axis=0)
mean_mean = mean_preds.mean(axis=0)

plt.figure(figsize=(8, 4))
plt.plot(mean_true, label='True', color='k', linewidth=2)
plt.plot(mean_full, label='Full TCN (8×8×2)', color='C2', linestyle='--')
plt.plot(mean_mean, label='Mean TCN (avg phase)', color='C1', linestyle=':')
plt.xlabel('Time (samples)')
plt.ylabel('Lever (mean across test trials)')
plt.title('Mean lever trajectory: True vs Full TCN vs Mean TCN')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"lever_decoder.pdf"),
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace

# %% Saliency map for full TCN model
import torch
import numpy as np
import matplotlib.pyplot as plt

model_full.eval()

all_dx = []
all_dy = []

for trial in test_idx:
    # normalized input for this trial: (1, T, 128)
    x_f = ((X_full[trial:trial+1] - X_full_mean) / X_full_std).astype(np.float32)
    x = torch.from_numpy(x_f).to(device)
    x.requires_grad_(True)

    # forward pass (no no_grad)
    y_pred = model_full(x)              # (1, T, 1)

    # scalar target: sum over time
    y_scalar = y_pred[0, :, 0].sum()

    model_full.zero_grad()
    y_scalar.backward()

    # saliency wrt input: (T, 128)
    sal = x.grad.detach().cpu().numpy()[0]

    # aggregate over time
    sal_t = np.mean(np.abs(sal), axis=0)       # (128,)

    # reshape to (8, 8, 2)
    sal_map = sal_t.reshape(dx_array.shape[-2], dx_array.shape[-1], 2)
    all_dx.append(sal_map[:, :, 0])
    all_dy.append(sal_map[:, :, 1])

# trial-averaged saliency maps
mean_dx = np.mean(np.stack(all_dx, axis=0), axis=0)   # (8, 8)
mean_dy = np.mean(np.stack(all_dy, axis=0), axis=0)   # (8, 8)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

im0 = axes[0].imshow(mean_dx.T, cmap='viridis', origin='upper')
axes[0].set_title('Mean saliency: dx phase')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(mean_dy.T, cmap='viridis', origin='upper')
axes[1].set_title('Mean saliency: dy phase')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"saliency_maps.pdf"),
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace

# %% Latent TCN trajectories for a test trial
model_full.eval()
trial = test_idx[0]  # choose a test trial

# normalized full input for this trial: (1, T, 128)
x_f = ((X_full[trial:trial+1] - X_full_mean) / X_full_std).astype(np.float32)
x = torch.from_numpy(x_f).to(device)

with torch.no_grad():
    # pass through TCN block only
    x_ch = x.transpose(1, 2)          # (1, C_in=128, T)
    h = model_full.tcn(x_ch)          # (1, hidden_channels, T)

h = h.cpu().numpy()[0]                # (hidden_channels, T)
n_hidden, T = h.shape

n_show = min(4, n_hidden)  # plot first 4 channels
time = np.arange(T)

fig, axes = plt.subplots(n_show, 1, figsize=(8, 6), sharex=True)

for i in range(n_show):
    axes[i].plot(time, h[i], linewidth=1)
    axes[i].set_ylabel(f'h{i}')
    # optionally overlay the true lever (scaled) for context
    # axes[i].twinx().plot(time, lever_traces[trial], color='k', alpha=0.3)

axes[-1].set_xlabel('Time (samples)')
fig.suptitle(f'Latent TCN trajectories (trial {trial})')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"Latent_TCN.pdf"),
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace
# %% Readout weights from latent channels to lever

readout = model_full.readout  # Conv1d
W = readout.weight.detach().cpu().numpy()   # shape (1, hidden_channels, 1)
b = readout.bias.detach().cpu().numpy()     # shape (1,)

# collapse extra dims -> (hidden_channels,)
w_vec = W[0, :, 0]

plt.figure(figsize=(6, 3))
plt.bar(np.arange(len(w_vec)), abs(w_vec))
plt.xlabel('Hidden channel index k')
plt.ylabel('Readout weight w_k')
plt.title('Readout weights from h_k to lever')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"Readout_weights.pdf"),
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace
# %%
model_full.eval()
trial = test_idx[0]  # choose a test trial

# normalized input for this trial
x_f = ((X_full[trial:trial+1] - X_full_mean) / X_full_std).astype(np.float32)
x = torch.from_numpy(x_f).to(device)

with torch.no_grad():
    x_ch = x.transpose(1, 2)          # (1, C_in, T)
    h = model_full.tcn(x_ch)          # (1, hidden, T)
h = h.cpu().numpy()[0]                # (hidden_channels, T)

lever_true = lever_traces[trial]      # (T,)

# compute correlation per channel
corrs = []
for k in range(h.shape[0]):
    hk = h[k]
    if np.std(hk) < 1e-8:
        corrs.append(0.0)        # constant channel -> define r=0
    else:
        c = np.corrcoef(hk, lever_true)[0, 1]
        if np.isnan(c):
            c = 0.0
        corrs.append(c)
corrs = np.array(corrs)

top3 = np.argsort(np.abs(corrs))[-3:][::-1]


print("Top 3 channels and correlations:", list(zip(top3, corrs[top3])))

# plot them
time = np.arange(h.shape[1])
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

for i, k in enumerate(top3):
    axes[i].plot(time, h[k], label=f'h{k} (r={corrs[k]:.2f})')
    axes[i].set_ylabel(f'h{k}')
    ax2 = axes[i].twinx()
    ax2.plot(time, lever_true, color='k', alpha=0.3, label='Lever')
    if i == 0:
        axes[i].legend(loc='upper left')
        ax2.legend(loc='upper right')

axes[-1].set_xlabel('Time (samples)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"top_latents.pdf"),
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace

# %%
from sklearn.linear_model import LinearRegression
model_full.eval()

hk_means = []
lever_means = []

with torch.no_grad():
    for tr in test_idx:
        # latent trajectories for this trial
        x_f = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
        x = torch.from_numpy(x_f).to(device)
        h = model_full.tcn(x.transpose(1, 2)).cpu().numpy()[0]   # (hidden, T)
        lever_true = lever_traces[tr]                            # (T,)

        # pick one channel, e.g. k
        k = top3[0]
        hk_means.append(h[k].mean())
        lever_means.append(lever_true.mean())

hk_means = np.array(hk_means)
lever_means = np.array(lever_means)

# correlation
r = np.corrcoef(hk_means, lever_means)[0, 1]

# regression
X_reg = hk_means.reshape(-1, 1)
reg = LinearRegression().fit(X_reg, lever_means)
y_line = reg.predict(X_reg)
idx = np.argsort(hk_means)

plt.figure(figsize=(5, 4))
plt.scatter(hk_means, lever_means, s=20, alpha=0.7, label='Trials')
plt.plot(hk_means[idx], y_line[idx], color='red', label='Linear fit')
plt.xlabel(f'Mean latent h{k} (per trial)')
plt.ylabel('Mean lever (per trial)')
plt.title(f'Mean h{k} vs mean lever (r={r:.2f})')
plt.legend()
plt.tight_layout()
plt.show()
# %% Preparatory wave (t >= 1500) vs lever response
from scipy.stats import pearsonr
model_full.eval()

hk_means = []
lever_means = []

t0 = 1500  # start index for preparatory/hold period

with torch.no_grad():
    for tr in test_idx:
        # latent trajectories for this trial
        x_f = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
        x = torch.from_numpy(x_f).to(device)
        h = model_full.tcn(x.transpose(1, 2)).cpu().numpy()[0]   # (hidden, T)
        lever_true = lever_traces[tr]                            # (T,)

        k = top3[0]  # chosen latent channel index
        hk_seg = h[k, t0:]           # only t >= 1500
        lev_seg = lever_true[t0:]

        hk_means.append(hk_seg.mean())
        lever_means.append(lev_seg.mean())

hk_means = np.array(hk_means)
lever_means = np.array(lever_means)

# hk_means, lever_means already computed (e.g., t >= 1500)
r, p = pearsonr(hk_means, lever_means)

X_reg = hk_means.reshape(-1, 1)
reg = LinearRegression().fit(X_reg, lever_means)
y_line = reg.predict(X_reg)
idx = np.argsort(hk_means)

plt.figure(figsize=(5, 4))
plt.scatter(hk_means, lever_means, s=20, alpha=0.7,
            label=f'Trials (r={r:.2f}, p={p:.3g})')
plt.plot(hk_means[idx], y_line[idx], color='red')
plt.xlabel(f'Mean latent h{k}')
plt.ylabel(f'Mean lever')
plt.legend(loc='best')
plt.tight_layout()
plt.title(f'Prepatory wave vs lever response')
plt.savefig(os.path.join(output_dir, f"wave_pg_regressor.pdf"),
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace
# %% Latent saliancy maps for top latent channel

import torch
import numpy as np


model_full.eval()
k = top3[0]          # index of lever-informative latent channel
tcn = model_full.tcn

all_maps_dx = []
all_maps_dy = []

for tr in test_idx:
    # normalized input for this trial
    x_f = ((X_full[tr:tr+1] - X_full_mean) / X_full_std).astype(np.float32)
    x = torch.from_numpy(x_f).to(device)
    x.requires_grad_(True)

    # forward through TCN (no no_grad)
    x_ch = x.transpose(1, 2)           # (1, C_in, T)
    h = tcn(x_ch)                      # (1, hidden, T)

    # scalar objective: sum of latent channel k over time
    y_latent = h[0, k].sum()

    model_full.zero_grad()
    y_latent.backward()

    sal = x.grad.detach().cpu().numpy()[0]        # (T, 128)
    sal_t = np.mean(np.abs(sal), axis=0)          # (128,)
    sal_map = sal_t.reshape(dx_array.shape[-2], dx_array.shape[-1], 2)              # (8, 8, 2)

    all_maps_dx.append(sal_map[:, :, 0])
    all_maps_dy.append(sal_map[:, :, 1])

# mean across trials
mean_dx = np.mean(np.stack(all_maps_dx, axis=0), axis=0)   # (8, 8)
mean_dy = np.mean(np.stack(all_maps_dy, axis=0), axis=0)   # (8, 8)

# plot
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

im0 = axes[0].imshow(mean_dx.T, cmap='viridis', origin='upper')
axes[0].set_title(f'Mean saliency (dx), latent h{k}')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(mean_dy.T, cmap='viridis', origin='upper')
axes[1].set_title(f'Mean saliency (dy), latent h{k}')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
# %% Output variables into a file for external analysis
import numpy as np
import os
import scipy.io as sio

# Create output directory
mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
output_dir = os.path.join(r"D:\BayesianWaveModel\model_output", mat_basename)
os.makedirs(output_dir, exist_ok=True)
# Save as .npz (NumPy compressed)
np.savez(
    os.path.join(output_dir, 'tcn_decoder_results.npz'),
    test_idx=test_idx,
    train_idx=train_idx,
    lever_true=true_traces,
    lever_tcn_full=full_preds,
    lever_tcn_mean=mean_preds,
    dx_array=dx_array,
    dy_array=dy_array,
    lever_traces=lever_traces,
    X_full_mean=X_full_mean,
    X_full_std=X_full_std,
    Y_full_mean=Y_full_mean,
    Y_full_std=Y_full_std,
    X_mean_mu=X_mean_mu,
    X_mean_sd=X_mean_sd,
    Y_mean_mu=Y_mean_mu,
    Y_mean_sd=Y_mean_sd
)

# Save as .mat file (MATLAB format)
mat_data = {
    'test_idx': test_idx,
    'train_idx': train_idx,
    'lever_true': true_traces,
    'lever_tcn_full': full_preds,
    'lever_tcn_mean': mean_preds,
    'dx_array': dx_array,
    'dy_array': dy_array,
    'lever_traces': lever_traces,
    'X_full_mean': X_full_mean,
    'X_full_std': X_full_std,
    'Y_full_mean': Y_full_mean,
    'Y_full_std': Y_full_std,
    'X_mean_mu': X_mean_mu,
    'X_mean_sd': X_mean_sd,
    'Y_mean_mu': Y_mean_mu,
    'Y_mean_sd': Y_mean_sd
}

sio.savemat(os.path.join(output_dir, 'tcn_decoder_results.mat'), mat_data)

# Save PyTorch models
torch.save(model_full.state_dict(), os.path.join(output_dir, 'model_full_tcn.pt'))
torch.save(mean_tcn.state_dict(), os.path.join(output_dir, 'model_mean_tcn.pt'))

print(f"Results saved to {output_dir}")

# %%
