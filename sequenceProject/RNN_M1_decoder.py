
#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os # For saving files

import h5py
import RNN_utilities as RNU


#%% 1. Wrangle data
mat_file_path = r"D:\SequenceProject\WarpedSpikes\M1\Day6_M1_warpedSpks.mat"

f =  h5py.File(mat_file_path, 'r')
# Access the 'warpedSpks' dataset (the MATLAB struct)
print(f.keys())

# Typically, the struct array contains references
# Get the first struct in the array (adjust the index for your case)
intan_behaviour = f['IntanBehaviour']
hit = intan_behaviour['hitTrace']
print(hit.keys())  # just to verify available fields
hit_trace = hit['trace']

# Flatten reference array for easy iteration
refs = hit_trace[:].flatten()

# Extract all traces as a list of numpy arrays
all_traces = [f[ref][:] for ref in refs]

# Optionally, stack into a 2D or 3D NumPy array if dimensions match
# For example, if each trace is (timepoints,), stack into (n_trials, timepoints)
all_traces_array = np.squeeze(np.stack(all_traces))

print(f"Extracted {len(all_traces)} traces")
print(f"Shape of single trace: {all_traces[0].shape}")
print(f"Shape of stacked array: {all_traces_array.shape}")
# Dereference to get the struct group
struct = f['warpedSpks' ]
print("Fields in the struct:")
print(list(struct.keys()))  # Should list all fields, including 'warpedSpikes'

# Access the 'warpedSpikes' field
warp_spk = struct['warpSpikes'][:]

structp = struct['pull3A']
pull1 = structp['pull1'][:]
pull2 = structp['pull2'][:]
pull3 = structp['pull3'][:]

# Load in warp data and wrangle it into what we need it to be
#warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\DLS\Day9_DLS_warpedSpks_rslds.npy")
#warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\M1\Day6_M1_warpedSpks_rslds.npy")

warp_spk = np.transpose(warp_spk, (3,2,1,0))
[n_trials,n_time,n_neurons,nPulls] = warp_spk.shape
warp_spk_ref = warp_spk[:,:,:,2]
spike_data = warp_spk_ref.reshape(n_time*n_trials,n_neurons)
#spike_data = np.load(r"D:\SequenceProject\WarpedSpikes\M1\Day6_rslds_test.npy")

print(spike_data.shape)
# Transpose data here
# original data should be neuronsxtime
# Check the shape of the loaded data (should now be time x neurons)
print(f"Generated data shape: {spike_data.shape}")

from motorCortexLever.spike_utilities import compute_binned_spike_data
# Parameters
sigma = 1  # Smoothing parameter (in bins)
bin_size_ms = 20  # Bin size in milliseconds
# Compute firing rates - make sure binned_spike_data is shape (neurons, time)
binned_spike_data = compute_binned_spike_data(spike_data, sigma, bin_size_ms)


num_timepoints_per_trial = n_time//bin_size_ms
num_total_timepoints, numPC_show = binned_spike_data.shape
num_trials = num_total_timepoints // num_timepoints_per_trial
# Prepare subplots: numPC_show rows, 2 columns
comp_combine = []
for comp in range(numPC_show):
    comp_all_time = binned_spike_data[:, comp]
    comp_by_trial = comp_all_time[:num_trials * num_timepoints_per_trial].reshape(num_trials, num_timepoints_per_trial)
    comp_combine.append(comp_by_trial) 

spks_decode = np.stack(comp_combine, axis=2) # targets shoudl always be trials x time x neurons
print("Result with stack (shape):", spks_decode.shape) 
inputs = np.concatenate((pull1,pull2,pull3),axis = 1)

inputs = inputs//bin_size_ms
# Assume: pulls is (trials x n_pulls), neural_data is (trials x T x latent_dim)
# Create event matrix (trials x T x 1)
event_matrix = np.zeros((n_trials, num_timepoints_per_trial, 1))
for trial in range(n_trials):
    for pull_time in inputs[trial]:
        event_matrix[trial, pull_time:pull_time+1, 0] = 1  # Or use real-valued feature
dt_traces = np.expand_dims(all_traces_array[:,1::bin_size_ms],axis=2)
event_matrix = np.squeeze(np.stack((event_matrix,dt_traces),axis = 2))
#%%

# 2. Re-initialize Model 
model = RNU.SimpleRNN(input_size=42, hidden_size=200, output_size=2)
# Define common parameters

# Training hyperparameters
epochs = 2000
initial_lr = 0.001
lr_reduction_factor = 0.95
lr_reduction_patience = 2000


all_training_results = {}

inputs = torch.tensor(spks_decode, dtype=torch.float32)  # inputs as float tensors
targets = torch.tensor(event_matrix, dtype=torch.float32)  # targets as 
print(f"\n--- Starting training for M1RNN---")
# 3. Train Model
trained_model, output_val, train_losses, val_losses, X_val, y_val = RNU.train_model(
    model, inputs, targets, epochs, initial_lr, lr_reduction_factor, lr_reduction_patience
)

# Store results
all_training_results = {
    'model': trained_model,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'output_val': output_val,
    'inputs': inputs,
    'targets': targets,
    'X_val': X_val,
    'y_val': y_val
}

# 4. Plot Results
#plot_results(inputs, targets, output_val, train_losses, val_losses, freq)

# 5. Save Model Data (Optional, if you want to save each model's output)
from scipy.io import savemat
output_n_current = output_val.detach().cpu().numpy()
targets_np_current = targets.detach().cpu().numpy()
input_np_current = inputs.detach().cpu().numpy()

RNN_output_data = {
    'RNN_output': output_n_current,
    'input': input_np_current,
    'target': targets_np_current
}
file_name_mat = f"RNN_model_decoder_M1_lever.mat"
savemat(file_name_mat, {'RNN_data': RNN_output_data})
print(f"Model data for M1 saved: {os.path.exists(file_name_mat)}")

print("\n--- All training runs completed ---")

# %%
RNU.plot_results(inputs, targets, output_val, train_losses, val_losses, ['0.5'])
# %%
"""
Plots training/validation loss and sample input/output/target comparisons.
"""
# Plotting Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Sample Input/Output/Target
num_trials_to_plot = min(3, inputs.shape[0]) # Plot up to 3 trials
time = range(inputs.shape[1])

fig, axs = plt.subplots(num_trials_to_plot, 3, figsize=(15, 5 * num_trials_to_plot))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

for trial_idx in range(num_trials_to_plot):
    # Model Input
    axs[trial_idx, 0].imagesc(time, inputs[trial_idx, :, 0].detach().cpu().numpy(), linestyle='--', color='blue')
    axs[trial_idx, 0].set_title('Model Input')
    axs[trial_idx, 0].set_ylabel(f'Trial {trial_idx+1}')

    # RNN Output
    axs[trial_idx, 1].plot(time, output_val[trial_idx, :, 0].detach().cpu().numpy(), linestyle='--', label='Dim 1', color='blue')
    axs[trial_idx, 1].plot(time, output_val[trial_idx, :, 1].detach().cpu().numpy(), linestyle='--', label='Dim 2', color='orange')
    axs[trial_idx, 1].set_title('RNN Output')
    if trial_idx == 0:
        axs[trial_idx, 1].legend()

    # Target Output
    axs[trial_idx, 2].plot(time, targets[trial_idx, :, 0].detach().cpu().numpy(), linestyle='--', label='Dim 1', color='blue')
    axs[trial_idx, 2].plot(time, targets[trial_idx, :, 1].detach().cpu().numpy(), linestyle='--', label='Dim 2', color='orange')
    axs[trial_idx, 2].plot(time, targets[trial_idx, :, 2].detach().cpu().numpy(), linestyle='--', label='Dim 3', color='green')
    axs[trial_idx, 2].set_title('Target Output')
    if trial_idx == 0:
        axs[trial_idx, 2].legend()

plt.xlabel('Timepoints')
fig.suptitle(f'Input/Output/Target Comparison (Input Freq: {current_frequency})', y=1.02, fontsize=16)
plt.tight_layout()
# Save figure
filename = f"{filename_prefix}_freq_{str(current_frequency).replace('.', '_')}_comparison.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename}")
plt.show()

# Plot 3D RNN Output Trajectories
output_n = model_output_val.detach().cpu().numpy()
fig_3d_output = plt.figure(figsize=(8, 8))
ax_3d_output = fig_3d_output.add_subplot(111, projection='3d')
for i in range(output_n.shape[0]):
    ax_3d_output.plot(output_n[i, :, 0], output_n[i, :, 1], output_n[i, :, 2], alpha=0.8)
ax_3d_output.set_xlabel('PC 1')
ax_3d_output.set_ylabel('PC 2')
ax_3d_output.set_zlabel('PC 3')
ax_3d_output.set_title(f'RNN Output 3D Trajectories (Input Freq: {current_frequency})')
plt.tight_layout()
# Save figure
filename_3d_output = f"{filename_prefix}_freq_{str(current_frequency).replace('.', '_')}_3d_output.pdf"
plt.savefig(filename_3d_output, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename_3d_output}")
plt.show()

# Plot 3D Target Trajectories (Unsmoothed)
targets_np_plot = targets.detach().cpu().numpy()
fig_3d_target = plt.figure(figsize=(8, 8))
ax_3d_target = fig_3d_target.add_subplot(111, projection='3d')
for i in range(targets_np_plot.shape[0]):
    ax_3d_target.plot(targets_np_plot[i, :, 0], targets_np_plot[i, :, 1], targets_np_plot[i, :, 2], alpha=0.8)
ax_3d_target.set_xlabel('PC 1')
ax_3d_target.set_ylabel('PC 2')
ax_3d_target.set_zlabel('PC 3')
ax_3d_target.set_title(f'Target 3D Trajectories (Input Freq: {current_frequency})')
plt.tight_layout()
# Save figure
filename_3d_target = f"{filename_prefix}_freq_{str(current_frequency).replace('.', '_')}_3d_target.pdf"
plt.savefig(filename_3d_target, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename_3d_target}")
plt.show()
