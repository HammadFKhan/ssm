#%% 
# Imports and sets up the environment for a Switching Linear Dynamical System (SLDS) analysis.
import autograd.numpy as np
import autograd.numpy.random as npr
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from ssm.util import rle, find_permutation

from ssm import SLDS

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm  # Import tqdm for loading bar
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score

import os
npr.seed(0)
import h5py
# %% Load data
#loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\DLS\rslds_models\Day9_DLS_warpedSpks_rsldsModel.npz",allow_pickle=True)
#loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\DLS\rslds_models\Day8_DLS_warpedSpks_rsldsModel.npz",allow_pickle=True)
#loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\M1\rslds_models\Day6_M1_warpedSpks_rsldsModel.npz",allow_pickle=True)
filename = r"Y:\Hammad\Ephys\SeqProject\ForceField\rsldsSpks_sessions\rslds_models_new\Mouse4_Day16_DLS_Spikes_rsldsSpks_rsldsModel.npz"
loaded = np.load(filename,allow_pickle=True)

print(list(loaded.keys()))
rsldsData = loaded['rsldsData'].item()  # .item() gets the actual dictionary
groundTruthData = loaded['groundTruthData'].item()  
q_elbos = rsldsData['q_elbos']
slds = rsldsData['rslds_model']
rslds_states = rsldsData['discrete_states']
latent_dynamics = groundTruthData['latent_dynamics']
num_states = np.max(rslds_states)+1
inferred_latent_dynamics = rsldsData['latent_states']
mat_file_path = groundTruthData['originalFilepath']
q_lem = rsldsData['q_lem']
q_lem_y = rsldsData['inferred_spikes']


with h5py.File(mat_file_path, 'r') as f:
    # Access the 'warpedSpks' dataset (the MATLAB struct)
    warpedSpks = f['rsldsSpikes']
    print(f.keys())
    struct = f['rsldsSpikes' ]
    print("Fields in the struct:")
    print(list(struct.keys()))  # Should list all fields, including 'warpedSpikes'

    # Access the 'warpedSpikes' field
    perturbbehav = struct['hiteffortperturb']
    print(list(perturbbehav.keys()))
    #structp = struct['pull3A']
    pull1 = perturbbehav['pull1'][:]
    pull2 = perturbbehav['pull2'][:]
    pull3 = perturbbehav['pull3'][:]
    totalPulls = [pull1, pull2, pull3]
    isnoeffort = perturbbehav['isnoeffort'][:]
    leverTraces = perturbbehav['leverTraces'][:]
    print("\n--- Extracted Pull Data ---")  
    print(f"Shape of leverTraces: {leverTraces.shape}")
    # Typically, the struct array contains references
    # Get the first struct in the array (adjust the index for your case)
    
    # Access the 'warpedSpikes' field
    task_spikes_refs = struct['taskSpikes'][:]
    task_labels = struct['taskLabel'][:]
    
    # 3. Handle the dimensions of the cell array.
    # A 1xN MATLAB cell array will be a (1, N) NumPy array of references.
    # You need to flatten it to iterate over the individual cell references.
    task_spikes_refs_flat = task_spikes_refs.flatten()
    task_labels = task_labels.flatten()
    # 4. Loop through the references to dereference and access the data.
    # This will give you a list of the contents of each cell.
    rslds_spk = []
    for ref in task_spikes_refs_flat:
        # Dereference the object reference and read the data
        cell_content = f[ref][:]
        rslds_spk.append(cell_content)

    task_lb = []
    for ref in task_labels:
        # Dereference and extract data (likely uint16)
        char_arr = f[ref][:]
        # Convert to Python string
        label = ''.join([chr(c) for c in char_arr.flatten() if c != 0])
        task_lb.append(label)
    # 5. Process the extracted cell data.
    # For a list of 3D arrays, this list will contain each array.
    if rslds_spk:
        print("\n--- Extracted Data ---")
        print("Number of task variables:", len(rslds_spk))
        print("Task labels:", task_lb)
        print("Shape of the first cell's array:", rslds_spk[0].shape)
        
        # Dereference again to get the actual 3D array data
        #print(f"Shape of warpedSpikes array: {warpedSpikes_data.shape}")
        # Now, warpedSpikes_data is a NumPy array with your 3D data

    # Load in warp data and wrangle it into what we need it to be
    #warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\DLS\Day9_DLS_warpedSpks_rslds.npy")
    #warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\M1\Day6_M1_warpedSpks_rslds.npy")
spk = rslds_spk[3] # Take the effort and no effort trials that are combined from the model
spk_ref = np.transpose(spk, (2,1,0))

[n_trials,n_time,n_neurons] = spk_ref.shape
print("\n--- Extracted Data ---")
print("Number of trials:", n_trials)
print("Time for each trial:", n_time)
print("Number of neurons:", n_neurons)

fSave = r'Figures\\EffortPerturbation\\'
# concatenate fsave with the name of the matfile without the extension
fSave = os.path.join(fSave, os.path.splitext(os.path.basename(mat_file_path))[0])
# make folder directory if fsave location does not exist
os.makedirs(fSave, exist_ok=True)
# print the save location
print(f"Figures will be saved to: {fSave}")
assert leverTraces.shape[0] == n_trials, "Number of trials in the lever trace does not match number of trials in spk_ref"

#%%

# Example variables (from your code/data)
# spk_ref: shape (n_trials, n_time, n_neurons)
# isnoeffort: shape (n_trials,)

# Convert isnoeffort to a boolean mask if needed
isnoeffort = np.array(isnoeffort).flatten()
# Index for effort and no effort
effort_mask = isnoeffort == 0
noeffort_mask = isnoeffort == 1

# Separate data by trial type
spk_effort = spk_ref[effort_mask, :, :]     # (n_effort_trials, n_time, n_neurons)
spk_noeffort = spk_ref[noeffort_mask, :, :] # (n_noeffort_trials, n_time, n_neurons)

# OR, flatten if you want time x neurons 2D for each:
spk_effort_flatten = spk_effort.reshape(-1, spk_ref.shape[2])
spk_noeffort_flatten = spk_noeffort.reshape(-1, spk_ref.shape[2])
spk_flatten = spk_ref.reshape(-1, spk_ref.shape[2])
# seperate lever data by trial type
lever_effort = leverTraces[effort_mask, :]     # (n_effort_trials, n_time)
lever_noeffort = leverTraces[noeffort_mask, :] # (n_noeffort_trials, n_time)

print('Effort shape:', spk_effort.shape)
print('No Effort shape:', spk_noeffort.shape)
print('Lever Effort shape:', lever_effort.shape)
print('Lever No Effort shape:', lever_noeffort.shape)

assert spk_effort.shape[0] == lever_effort.shape[0], "Number of effort trials in spike data does not match lever data"
assert spk_noeffort.shape[0] == lever_noeffort.shape[0], "Number of no effort trials in spike data does not match lever data"
# We next have to format the rslds models correctly to conduct the analysis that we want

#%% Here we assign the effort and non effort data

# spike_data = spk_ref.reshape(n_time*n_trials,n_neurons)
spike_data = spk_flatten
#spike_data = np.load(r"D:\SequenceProject\WarpedSpikes\M1\Day6_rslds_test.npy")
all_traces_array = leverTraces
print(spike_data.shape)
# Transpose data here
# original data should be neuronsxtime
# Check the shape of the loaded data (should now be time x neurons)
print(f"Generated data shape: {spike_data.shape}")

from motorCortexLever.spike_utilities import compute_binned_spike_data
# Parameters
sigma = 5  # Smoothing parameter (in bins)
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

# Get PC weights (loadings) for each neuron
# In sklearn, components_ is of shape (n_components, n_features)
pc_weights = pca.components_  # Each row is a PC, each column is a neuron

# Create a sorting index based on PC weights
# Sort neurons primarily by their PC1 weights, then PC2, then PC3
# First, let's group by sign of PC1
pc1_positive = pc_weights[0] > 0
pc1_negative = ~pc1_positive

# Within each group, sort by magnitude of PC1 weight
sort_idx = np.zeros(pc_weights.shape[1], dtype=int)
pos_idx = np.where(pc1_positive)[0]
neg_idx = np.where(pc1_negative)[0]

# Sort positive PC1 neurons by decreasing weight
sort_idx[:len(pos_idx)] = pos_idx[np.argsort(-pc_weights[0, pos_idx])]
# Sort negative PC1 neurons by increasing weight (most negative first)
sort_idx[len(pos_idx):] = neg_idx[np.argsort(pc_weights[0, neg_idx])]

leverTrace = np.reshape(all_traces_array, all_traces_array.shape[0]*all_traces_array.shape[1])
leverTrace = leverTrace[1::bin_size_ms]
#leverTrace = gaussian_filter1d(leverTrace*100, 2)/100
# normalize leverTrace to be between 0 and 1
leverTrace = (leverTrace - np.min(leverTrace)) / (np.max(leverTrace) - np.min(leverTrace))
leverTrace = np.reshape(all_traces_array, all_traces_array.shape[0]*all_traces_array.shape[1])
# normalize leverTrace to be between 0 and 1
def bin_lever_trace(leverTrace, bin_size_ms=10):
    n_timebins = len(leverTrace)
    n_bins = n_timebins // bin_size_ms
    n_timebins_trimmed = n_bins * bin_size_ms
    leverTrace = leverTrace[:n_timebins_trimmed]
    # mean or some other stat per bin
    leverTrace_binned = leverTrace.reshape(n_bins, bin_size_ms).mean(axis=1)
    return leverTrace_binned
leverTrace = bin_lever_trace(leverTrace, bin_size_ms=bin_size_ms)
leverTrace = gaussian_filter1d(leverTrace*100, 5)/100
leverTrace = (leverTrace - np.min(leverTrace)) / (np.max(leverTrace) - np.min(leverTrace))
# convert totalPulls into np array and bin the pull times by bin_size_ms but ignore nans
totalPullTimes = np.array(totalPulls)
# squueze totalPullTimes to be 2D (pull_num, pull_time)
totalPullTimes = np.squeeze(totalPullTimes)

# print the shape of the lever trace and rslds states to make sure they match and binned spike states
print(f"Lever trace shape: {leverTrace.shape}")
print(f"RSLDS states shape: {rslds_states.shape}")
print(f"Binned spike data shape: {binned_spike_data.shape}")
print(f"Pull trial times shape: {totalPullTimes.shape}")
#%Reshape lever trace and rslds states to be (n_trials, n_time) and (n_trials, n_time) respectively
# Reshape lever trace to (n_trials, n_time)
n_trials = totalPullTimes.shape[1] # Assuming totalPullTimes is (pull_num, n_trials)
n_time = leverTrace.shape[0] // n_trials
n_neurons = binned_spike_data.shape[1]
n_neurons = binned_spike_data.shape[1]
n_bins_per_trial = n_time
binned_spike__trial_data = binned_spike_data[:n_bins_per_trial*n_trials, :]  # Trim to fit exact number of trials and bins
binned_spike__trial_data = binned_spike__trial_data.reshape(n_trials, n_bins_per_trial, n_neurons)


leverTrace = leverTrace[:n_trials*n_time]  # Trim to fit exact number of trials and time
leverTrace = leverTrace.reshape(n_trials, n_time)
# Reshape rslds states to (n_trials, n_time)
rslds_states = rslds_states[:n_trials*n_time]  # Trim to fit exact number of trials and time
rslds_states = rslds_states.reshape(n_trials, n_time)
print("\n--- Reshaped Data ---")
print(f"Lever trace shape (n_trials, n_time): {leverTrace.shape}")
print(f"RSLDS states shape (n_trials, n_time): {rslds_states.shape}")
print(f"Binned spike data shape (n_trials, n_bins_per_trial, n_neurons): {binned_spike__trial_data.shape}")
print(f"Pull trial times shape: {totalPullTimes.shape}")
# %% Do trial by trial rslds analysis
# plot example rslds states and lever trace for one trial
trial_num = 9
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(leverTrace[trial_num], label='Lever Trace')
plt.title(f'Trial {trial_num} Lever Trace')
plt.xlabel('Time (binned)')
plt.ylabel('Lever Position (normalized)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(rslds_states[trial_num], label='RSLDS States')
plt.title(f'Trial {trial_num} RSLDS States')
plt.xlabel('Time (binned)')
plt.ylabel('Discrete State')
plt.legend()
plt.tight_layout()
plt.show()
# %% Plot out average lever trace and rslds states across all trials
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(leverTrace.T, color='gray', alpha=0.5)
plt.plot(np.mean(leverTrace, axis=0), color='blue', label='Average Lever Trace (All Trials)')
plt.title('Lever Trace for All Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Lever Position (normalized)')
plt.legend()
plt.tight_layout()
plt.show()
plt.subplot(2, 1, 2)
for state in range(num_states):
    state_freq = np.mean(rslds_states == state, axis=0)
    plt.plot(state_freq, label=f'State {state}')
plt.title('RSLDS State Frequency for All Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
# %% Plot out lever trace and rslds state for non effort trials only
non_effort_trials = np.where(isnoeffort == 1)[0]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(leverTrace[non_effort_trials].T, color='gray', alpha=0.5)
plt.plot(np.mean(leverTrace[non_effort_trials], axis=0), color='blue', label='Average Lever Trace (No Effort)')
plt.title('Lever Trace for No Effort Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Lever Position (normalized)')
plt.legend()
plt.tight_layout()
plt.show()
plt.subplot(2, 1, 2)
for state in range(num_states):
    state_freq = np.mean(rslds_states[non_effort_trials] == state, axis=0)
    plt.plot(state_freq, label=f'State {state}')
plt.title('RSLDS State Frequency for No Effort Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
# %% Plot out lever trace and rslds state for effort trials only
effort_trials = np.where(isnoeffort == 0)[0]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(leverTrace[effort_trials].T, color='gray', alpha=0.5)
plt.plot(np.mean(leverTrace[effort_trials], axis=0), color='blue', label='Average Lever Trace (Effort)')
plt.title('Lever Trace for Effort Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Lever Position (normalized)')
plt.legend()
plt.tight_layout()
plt.show()
plt.subplot(2, 1, 2)
for state in range(num_states):
    state_freq = np.mean(rslds_states[effort_trials] == state, axis=0)
    plt.plot(state_freq, label=f'State {state}')
plt.title('RSLDS State Frequency for Effort Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# %% 
# state switching analysis - plot out the probability of switching states at each time point across all trials and compare between effort and no effort trials
state_switches = np.diff(rslds_states, axis=1) != 0  # Boolean array where True indicates a state switch
switch_prob = np.mean(state_switches, axis=0)  # Average across trials to get probability of switching at each time point
plt.figure(figsize=(12, 6))
plt.plot(switch_prob, label='State Switch Probability')
plt.title('Probability of Switching RSLDS States Across All Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Switch Probability')
plt.legend()
plt.tight_layout()
plt.show()
# Compare switch probability between effort and no effort trials
switch_prob_effort = np.mean(state_switches[effort_trials], axis=0)
switch_prob_noeffort = np.mean(state_switches[non_effort_trials], axis=0)
plt.figure(figsize=(12, 6))
plt.plot(switch_prob_effort, label='Effort Trials')
plt.plot(switch_prob_noeffort, label='No Effort Trials')
plt.title('Probability of Switching RSLDS States: Effort vs No Effort Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Switch Probability')
plt.legend()
plt.tight_layout()
plt.show()
# print mean switch probability for effort and no effort trials
mean_switch_prob_effort = np.mean(switch_prob_effort)
mean_switch_prob_noeffort = np.mean(switch_prob_noeffort)
print(f"Mean switch probability for effort trials: {mean_switch_prob_effort:.4f}")
print(f"Mean switch probability for no effort trials: {mean_switch_prob_noeffort:.4f}")
# %%
# make a histogram of the number of state switches per trial for effort and no effort trials
num_switches_per_trial = np.sum(state_switches, axis=1)  # Total number of switches per trial
bins = np.arange(0, np.max(num_switches_per_trial) + 1) - 0.5  # Bin edges for histogram
plt.figure(figsize=(12, 6))
plt.hist(num_switches_per_trial[effort_trials], bins=bins, alpha=0.5, label='Effort Trials')
plt.hist(num_switches_per_trial[non_effort_trials], bins=bins, alpha=0.5, label='No Effort Trials')
plt.title('Histogram of Number of RSLDS State Switches per Trial')
plt.xlabel('Number of State Switches')
plt.ylabel('Number of Trials')
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Plot out the binary response of iseffort 
plt.figure(figsize=(12, 6))
plt.plot(isnoeffort, label='Is No Effort (1) vs Effort (0)')
plt.title('Binary Response of No Effort vs Effort Trials')
plt.xlabel('Trial Number')
plt.ylabel('No Effort (1) vs Effort (0)')
plt.legend()
plt.tight_layout()
# %% Plot out the state duration for as a function of trials
state_durations = []
for trial in range(rslds_states.shape[0]):
    durations = rle(rslds_states[trial])
    state_durations.append(durations)

n_trials = rslds_states.shape[0]
n_states = int(rslds_states.max()) + 1  # or set manually

mean_dur = np.full((n_trials, n_states), np.nan)

for trial in range(n_trials):
    states, durations = state_durations[trial]
    for s in range(n_states):
        mask = (states == s)
        if np.any(mask):
            mean_dur[trial, s] = durations[mask].mean()/rslds_states.shape[1]  # Normalize by trial length to get percentage of trial spent in each state
            mean_dur = np.nan_to_num(mean_dur, nan=0.0)


trials = np.arange(n_trials)

plt.figure(figsize=(8, 5))
for s in range(n_states):
    plt.plot(trials, mean_dur[:, s], label=f"State {s}", alpha=0.7)

# overlab vertical lines where an effort trial occurs
for trial in range(n_trials):
    if isnoeffort[trial] == 0:  # Effort trial
        plt.axvline(x=trial, color='red', alpha=0.3, linestyle='--')
        
plt.xlabel("Trial")
plt.ylabel("Mean state duration (% of trial)")
plt.title("State persistence across trials")
plt.legend()
plt.tight_layout()
plt.show()