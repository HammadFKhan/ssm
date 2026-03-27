#%%
import h5py
import numpy as np

mat_file_path = r"D:\BayesianWaveModel\M1Waves\Day4WavesIntan.mat"

with h5py.File(mat_file_path, 'r') as f:
    print(list(f.keys()))
    intan_behaviour = f['IntanBehaviour']

    # ---- lever traces (hit trials) ----
    hit = intan_behaviour['hitTrace']
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
# %% Delta-ahead TCN decoding: delays 0–800 ms in 50 ms steps
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

fs = 1000.0  # Hz; change if your sampling rate differs
max_delay_ms = 800
step_ms = 50
delays_ms = np.arange(0, max_delay_ms + step_ms, step_ms)
delays_samples = (delays_ms * fs / 1000.0).astype(int)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhaseLeverDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

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

def train_tcn_for_delay(X_full, lever_traces, delay_samp, train_idx,
                        test_idx,hidden_channels=32,
                        n_layers=4, kernel_size=7, n_epochs=40,
                        batch_size=8, lr=1e-3, smooth_lambda=1e-1):
    """
    Train a TCN to predict lever(t + delay) from X_full(t).
    Returns trained model, normalization stats, and test performance.
    """
    n_trials, T, C = X_full.shape
    assert lever_traces.shape[1] == T
    assert delay_samp < T

    max_T = T - delay_samp
    X_shift = X_full[:, :max_T, :]
    Y_shift = lever_traces[:, delay_samp:T]
    Y_shift = Y_shift[..., None].astype(np.float32)

    # Normalize X over all data before split
    X_mean = X_shift.mean(axis=(0, 1), keepdims=True)
    X_std = X_shift.std(axis=(0, 1), keepdims=True) + 1e-6
    X_shift_n = (X_shift - X_mean) / X_std
    
    # fixed split on normalized data
    X_train_n = X_shift_n[train_idx]
    X_test_n = X_shift_n[test_idx]
    X_train = X_shift[train_idx]
    X_test = X_shift[test_idx]

    # Normalize Y over all data before split
    Y_mean = Y_shift.mean(axis=(0, 1), keepdims=True)
    Y_std  = Y_shift.std(axis=(0, 1), keepdims=True) + 1e-6
    Y_shift_n = (Y_shift - Y_mean) / Y_std
    
    # fixed split on normalized data
    Y_train_n = Y_shift_n[train_idx]
    Y_test_n  = Y_shift_n[test_idx]
    Y_train = Y_shift[train_idx]
    Y_test = Y_shift[test_idx]

    train_ds = PhaseLeverDataset(X_train_n, Y_train_n)
    test_ds  = PhaseLeverDataset(X_test_n,  Y_test_n)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


    model = TCNDecoder(
        in_channels=C,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        kernel_size=kernel_size
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in train_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            opt.zero_grad()
            Yhat = model(Xb)
            mse_loss = crit(Yhat, Yb)
            diff = Yhat[:, 1:, 0] - Yhat[:, :-1, 0]
            smooth_loss = (diff**2).mean()
            loss = mse_loss + smooth_lambda * smooth_loss
            loss.backward()
            opt.step()

        # Evaluate on test set
    model.eval()
    all_corrs, all_rmses = [], []
    per_trial_preds = []   # collect y_pred per test trial
    per_trial_true  = []   # collect y_true per test trial

    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Yhat_n = model(Xb).cpu().numpy()       # (B, T', 1)
            Ytrue_n = Yb.cpu().numpy()
            # de-normalize
            Yhat = Yhat_n * Y_std + Y_mean
            Ytrue = Ytrue_n * Y_std + Y_mean

            for b in range(Yhat.shape[0]):
                y_pred = Yhat[b, :, 0]
                y_true = Ytrue[b, :, 0]

                per_trial_preds.append(y_pred)
                per_trial_true.append(y_true)

                if np.std(y_true) < 1e-8:
                    continue
                c = np.corrcoef(y_true, y_pred)[0, 1]
                if np.isnan(c):
                    continue
                all_corrs.append(c)
                all_rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))

    mean_corr = float(np.nanmean(all_corrs)) if len(all_corrs) > 0 else np.nan
    mean_rmse = float(np.nanmean(all_rmses)) if len(all_rmses) > 0 else np.nan

    per_trial_preds = np.stack(per_trial_preds, axis=0)  # (n_test, T_eff)
    per_trial_true  = np.stack(per_trial_true,  axis=0)

    stats = dict(
        X_mean=X_mean, X_std=X_std,
        Y_mean=Y_mean, Y_std=Y_std,
        test_idx=test_idx,
        mean_corr=mean_corr,
        mean_rmse=mean_rmse,
        per_trial_preds=per_trial_preds,
        per_trial_true=per_trial_true
    )
    return model, stats


# full spatial features: (n_trials, T, 128)
# ---------- 1. Rebuild feature tensors ----------
X_full = np.stack([dx_array, dy_array], axis=-1)          # (n_trials, T, 8, 8, 2)
X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], -1).astype(np.float32)
lever_traces = lever_traces.astype(np.float32)           # (n_trials, T)

n_trials = X_full.shape[0]
# fixed train/test split reused for all delays
idx = np.arange(n_trials)
n_train = int(0.8 * n_trials)
train_idx = idx[:n_train]
test_idx = idx[n_train // 10:]


# mean features: (n_trials, T, 2)
dx_mean = dx_array.mean(axis=(2, 3))
dy_mean = dy_array.mean(axis=(2, 3))
X_mean = np.stack([dx_mean, dy_mean], axis=-1).astype(np.float32)

Y = lever_traces[..., None].astype(np.float32)            # (n_trials, T, 1)

Yhat_delay = {}
Ytrue_delay = {}
delay_results = []

for delay_ms, delay_samp in zip(delays_ms, delays_samples):
    print(f"\nTraining delta-ahead TCN for delay = {delay_ms} ms ({delay_samp} samples)")
    model_d, stats_d = train_tcn_for_delay(
        X_full=X_full,
        lever_traces=lever_traces,
        delay_samp=delay_samp,
        train_idx=train_idx,
        test_idx=test_idx,
        hidden_channels=32,
        n_layers=4,
        kernel_size=7,
        n_epochs=40
    )
    delay_results.append({
        "delay_ms": int(delay_ms),
        "delay_samples": int(delay_samp),
        "mean_corr": stats_d["mean_corr"],
        "mean_rmse": stats_d["mean_rmse"]
    })
    # store per-trial predictions/true traces for permutation
    Yhat_delay[int(delay_ms)]  = stats_d["per_trial_preds"]
    Ytrue_delay[int(delay_ms)] = stats_d["per_trial_true"]

    print(f"  -> mean test corr = {stats_d['mean_corr']:.3f}, "
          f"RMSE = {stats_d['mean_rmse']:.4f}")

# delay_results is a list of dicts you can later save or plot:
# correlation / RMSE as a function of prediction horizon (0–800 ms).


# %%
import numpy as np
import matplotlib.pyplot as plt

# Extract arrays from results
delays_ms_arr = np.array([r["delay_ms"] for r in delay_results])
mean_corrs    = np.array([r["mean_corr"] for r in delay_results])
mean_mses     = np.array([r["mean_rmse"] for r in delay_results])

# ----- Fraction-of-zero-lag cutoff -----
alpha = 0.5  # fraction of zero-lag correlation to define cutoff

# find zero-lag index and zero-lag correlation
zero_idx = np.where(delays_ms_arr == 0)[0]
if len(zero_idx) == 0:
    raise ValueError("No 0 ms delay found in delay_results.")
zero_idx = zero_idx[0]
r0 = mean_corrs[zero_idx]
thr = (mean_corrs[zero_idx]-mean_corrs[-1:])*alpha+mean_corrs[-1]  # threshold for cutoff

# largest delay with correlation above threshold
valid_idx = np.where(mean_corrs >= thr)[0]
cutoff_delay = None
if len(valid_idx) > 0:
    cutoff_idx = valid_idx[-1]
    cutoff_delay = delays_ms_arr[cutoff_idx]

print(f"Fraction-of-zero-lag cutoff (alpha={alpha:.2f}): "
      f"{cutoff_delay} ms")

# ----- Plot with cutoff indicated -----
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax1 = plt.subplots(figsize=(4, 4))

color1 = 'tab:blue'
ax1.plot(delays_ms_arr, mean_corrs, '-o', color=color1, label='Mean corr')
ax1.axhline(thr, color=color1, linestyle='--', alpha=0.4,
            label=f'{alpha*100:.0f}% of r(0)')

if cutoff_delay is not None:
    ax1.axvline(cutoff_delay, color='gray', linestyle='--', alpha=0.7)
    ax1.text(cutoff_delay, ax1.get_ylim()[1]*0.9,
             f'{cutoff_delay} ms', rotation=90,
             va='top', ha='right', fontsize=8)

ax1.set_xlabel('Prediction delay (ms)')
ax1.set_ylabel('Mean test correlation', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title('TCN decoding vs prediction horizon')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.plot(delays_ms_arr, mean_mses, '-s', color=color2, label='Mean MSE')
ax2.set_ylabel('Mean test MSE', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.savefig('waveDecoderDelays.pdf',
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace
plt.show()


# %%
import numpy as np
from tqdm import tqdm

# parameters for outer loop
n_splits = 10          # number of random train/test splits
train_frac = 0.8       # fraction of trials used for training

all_delay_results = []  # list of length n_splits, each entry is delay_results for that split

n_trials = X_full.shape[0]

for split_id in tqdm(range(n_splits), desc="Outer splits"):
    # ----- new random train/test split -----
    idx = np.arange(n_trials)
    np.random.shuffle(idx)
    n_train = int(train_frac * n_trials)
    train_idx = idx[:n_train]
    test_idx  = idx[int(n_train * 0.9):]

    delay_results = []

    for delay_ms, delay_samp in tqdm(
        list(zip(delays_ms, delays_samples)),
        desc=f"Delays (split {split_id+1}/{n_splits})",
        leave=False
    ):
        model_d, stats_d = train_tcn_for_delay(
            X_full=X_full,
            lever_traces=lever_traces,
            delay_samp=delay_samp,
            train_idx=train_idx,
            test_idx=test_idx,
            hidden_channels=32,
            n_layers=4,
            kernel_size=7,
            n_epochs=40
        )

        delay_results.append({
            "delay_ms": int(delay_ms),
            "delay_samples": int(delay_samp),
            "mean_corr": stats_d["mean_corr"],
            "mean_rmse": stats_d["mean_rmse"]
        })

    all_delay_results.append(delay_results)

#%% ----- aggregate across splits to get mean ± SEM -----
delays_ms_arr = np.array([r["delay_ms"] for r in all_delay_results[0]])
n_delays = len(delays_ms_arr)
n_splits = len(all_delay_results)

corr_mat = np.zeros((n_splits, n_delays))
mse_mat  = np.zeros((n_splits, n_delays))

for s, delay_results in enumerate(all_delay_results):
    corr_mat[s, :] = [r["mean_corr"] for r in delay_results]
    mse_mat[s, :]  = [r["mean_rmse"] for r in delay_results]

corr_mean = corr_mat.mean(axis=0)
corr_sem  = corr_mat.std(axis=0, ddof=1) 

mse_mean  = mse_mat.mean(axis=0)
mse_sem   = mse_mat.std(axis=0, ddof=1) 

# ----- Fraction-of-zero-lag cutoff -----
alpha = 0.5  # fraction of zero-lag correlation to define cutoff

# find zero-lag index and zero-lag correlation
zero_idx = np.where(delays_ms_arr == 0)[0]
if len(zero_idx) == 0:
    raise ValueError("No 0 ms delay found in delay_results.")
zero_idx = zero_idx[0]
r0 = corr_mean[zero_idx]
thr = (corr_mean[zero_idx]-corr_mean[-1:])*alpha+corr_mean[-1]  # threshold for cutoff

# largest delay with correlation above threshold
valid_idx = np.where(corr_mean >= thr)[0]
cutoff_delay = None
if len(valid_idx) > 0:
    cutoff_idx = valid_idx[-1]
    cutoff_delay = delays_ms_arr[cutoff_idx]

print(f"Fraction-of-zero-lag cutoff (alpha={alpha:.2f}): "
      f"{cutoff_delay} ms")

# ----- plot with SEM -----
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
fig, ax1 = plt.subplots(figsize=(5, 4))

color1 = 'tab:blue'
ax1.errorbar(delays_ms_arr, corr_mean, yerr=corr_sem,
             fmt='-o', color=color1, capsize=4, label='Mean corr ± SEM')
ax1.set_xlabel('Prediction delay (ms)')
ax1.set_ylabel('Mean test correlation', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title('TCN decoding vs prediction horizon (across splits)')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.errorbar(delays_ms_arr, mse_mean, yerr=mse_sem,
             fmt='-s', color=color2, capsize=4, label='Mean MSE ± SEM')
ax2.set_ylabel('Mean test MSE', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

if cutoff_delay is not None:
    ax1.axvline(cutoff_delay, color='gray', linestyle='--', alpha=0.7)
    ax1.text(cutoff_delay, ax1.get_ylim()[1]*0.9,
             f'{cutoff_delay} ms', rotation=90,
             va='top', ha='right', fontsize=8)
    
fig.tight_layout()
plt.savefig('waveDecoderDelays.pdf',
            format='pdf',
            bbox_inches='tight')   # trims extra whitespace
plt.show()

plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Extract arrays from results
delays_ms_arr = np.array([r["delay_ms"] for r in delay_results])
mean_corrs    = np.array([r["mean_corr"] for r in delay_results])
mean_mses     = np.array([r["mean_rmse"] for r in delay_results])

# ----- Fraction-of-zero-lag cutoff -----
alpha = 0.5  # fraction of zero-lag correlation to define cutoff

# find zero-lag index and zero-lag correlation
zero_idx = np.where(delays_ms_arr == 0)[0]
if len(zero_idx) == 0:
    raise ValueError("No 0 ms delay found in delay_results.")
zero_idx = zero_idx[0]
r0 = mean_corrs[zero_idx]
thr = (mean_corrs[zero_idx]-mean_corrs[-1:])*alpha+mean_corrs[-1]  # threshold for cutoff

# largest delay with correlation above threshold
valid_idx = np.where(mean_corrs >= thr)[0]
cutoff_delay = None
if len(valid_idx) > 0:
    cutoff_idx = valid_idx[-1]
    cutoff_delay = delays_ms_arr[cutoff_idx]

print(f"Fraction-of-zero-lag cutoff (alpha={alpha:.2f}): "
      f"{cutoff_delay} ms")

# ----- Plot with cutoff indicated -----
fig, ax1 = plt.subplots(figsize=(6, 4))

color1 = 'tab:blue'
ax1.plot(delays_ms_arr, mean_corrs, '-o', color=color1, label='Mean corr')
ax1.axhline(thr, color=color1, linestyle='--', alpha=0.4,
            label=f'{alpha*100:.0f}% of r(0)')

if cutoff_delay is not None:
    ax1.axvline(cutoff_delay, color='gray', linestyle='--', alpha=0.7)
    ax1.text(cutoff_delay, ax1.get_ylim()[1]*0.9,
             f'{cutoff_delay} ms', rotation=90,
             va='top', ha='right', fontsize=8)

ax1.set_xlabel('Prediction delay (ms)')
ax1.set_ylabel('Mean test correlation', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title('TCN decoding vs prediction horizon')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.plot(delays_ms_arr, mean_mses, '-s', color=color2, label='Mean MSE')
ax2.set_ylabel('Mean test MSE', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.show()
# %%
