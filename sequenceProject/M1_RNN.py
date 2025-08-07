
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
#%%
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Choose number of components for latent space (2-3 is good for visualization)
n_components = 10

# Fit PCA to the spike count data
scaler = StandardScaler(with_std=False)
smoothed_spikes_standardized = scaler.fit_transform(binned_spike_data)

pca = PCA(n_components=n_components)
latent_dynamics = pca.fit_transform(smoothed_spikes_standardized)

num_timepoints_per_trial = n_time//bin_size_ms
num_total_timepoints, numPC_show = latent_dynamics.shape
num_trials = num_total_timepoints // num_timepoints_per_trial
# Prepare subplots: numPC_show rows, 2 columns
comp_combine = []
for comp in range(numPC_show):
    comp_all_time = latent_dynamics[:, comp]
    comp_by_trial = comp_all_time[:num_trials * num_timepoints_per_trial].reshape(num_trials, num_timepoints_per_trial)
    comp_combine.append(comp_by_trial) 

targets = np.stack(comp_combine, axis=2) # targets shoudl always be trials x time x neurons
print("Result with stack (shape):", targets.shape) 
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
model = RNU.SimpleRNN(input_size=1, hidden_size=200, output_size=10)
# Define common parameters

# Training hyperparameters
epochs = 2000
initial_lr = 0.001
lr_reduction_factor = 0.95
lr_reduction_patience = 2000


all_training_results = {}

inputs = torch.tensor(dt_traces, dtype=torch.float32)  # inputs as float tensors
targets = torch.tensor(targets, dtype=torch.float32)  # targets as 
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
file_name_mat = f"RNN_model_data_M1_lever.mat"
savemat(file_name_mat, {'RNN_data': RNN_output_data})
print(f"Model data for M1 saved: {os.path.exists(file_name_mat)}")

print("\n--- All training runs completed ---")

# %%
RNU.plot_results(inputs, targets, output_val, train_losses, val_losses, ['0.5'])
# %%
