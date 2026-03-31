#%% 
# Imports and sets up the environment for a Switching Linear Dynamical System (SLDS) analysis.
import autograd.numpy as np
import autograd.numpy.random as npr
from numpy import cumsum
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
filename = r"Y:\Hammad\Ephys\SeqProject\ForceField\rsldsSpks_sessions\rslds_models_new\Mouse4_Day17_DLS_Spikes_rsldsSpks_rsldsModel.npz"
#filename = r"Y:\Hammad\Ephys\SeqProject\ForceField\rsldsSpks_sessions\rslds_models_new_inputdriven\Mouse4_Day17_DLS_Spikes_rsldsSpks_rsldsModel.npz"
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
#latent_dynamics_trial = latent_dynamics.reshape(n_trials, num_timepoints_per_trial, n_components)

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
se_mean_dur = np.full((n_trials, n_states), np.nan)

for trial in range(n_trials):
    states, durations = state_durations[trial]
    for s in range(n_states):
        mask = (states == s)
        if np.any(mask):
            mean_dur[trial, s] = durations[mask].mean()/rslds_states.shape[1]  # Normalize by trial length to get percentage of trial spent in each state
            # calculate standard error of the mean duration for each state across trials
            se_mean_dur[trial, s] = (durations[mask].std() / np.sqrt(np.sum(mask)))/rslds_states.shape[1]  # Standard error of the mean
            mean_dur = np.nan_to_num(mean_dur, nan=0.0)

# smooth mean_dur across trials for each state
for s in range(n_states):
    mean_dur[:, s] = gaussian_filter1d(mean_dur[:, s], sigma=2)
    se_mean_dur[:, s] = gaussian_filter1d(se_mean_dur[:, s], sigma=2)
trials = np.arange(n_trials)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# ---------- Top: state persistence ----------
ax1 = axes[0]

for s in range(n_states):
    ax1.plot(trials, mean_dur[:, s], label=f"State {s}", alpha=0.7)
    #ax1.fill_between(trials, mean_dur[:, s] - se_mean_dur[:, s], mean_dur[:, s] + se_mean_dur[:, s], alpha=0.2)
for trial in range(n_trials):
    if isnoeffort[trial] == 0:  # Effort trial
        ax1.axvline(x=trial, color='black', alpha=0.1, linestyle='--')
        if np.sum(~np.isnan(totalPullTimes[:, trial]))>1:
            ax1.axvline(x=trial, color='blue', alpha=0.5, linestyle='--')

ax1.set_ylabel("Mean state duration\n(% of trial)")
ax1.set_title("State persistence across trials")
ax1.legend()

# ---------- Bottom: cumulative effort trials ----------
ax2 = axes[1]

effort_indicator = (isnoeffort == 0).astype(int)
window_size = 10
c = np.cumsum(effort_indicator)  # length n_trials
moving_sum = np.empty_like(effort_indicator, dtype=float)
for i in range(len(effort_indicator)):
    start = max(0, i - window_size + 1)
    if start == 0:
        moving_sum[i] = c[i]
    else:
        moving_sum[i] = c[i] - c[start - 1]

# optional: convert to fraction in window
moving_frac = moving_sum / np.minimum(window_size, np.arange(len(effort_indicator)) + 1)
# filter moving_frac with gaussian filter for smoother plot
moving_frac = gaussian_filter1d(moving_frac, sigma=2)
ax2.plot(trials, moving_frac, color='red')
ax2.set_xlabel("Trial")
ax2.set_ylabel("Cumulative effort\ntrials")
ax2.set_title("Cumulative number of effort trials")

plt.tight_layout()
plt.show()
#%%
# Mark pull times as a function of effort and non effort trials
pull_non_effort = totalPullTimes[:, noeffort_mask]
pull_effort = totalPullTimes[:, effort_mask]
# now plot the pull times for effort and non effort trials
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
for trial in range(pull_non_effort.shape[1]):
    pull_times = pull_non_effort[:, trial]
    pull_times = pull_times[~np.isnan(pull_times)]
    plt.scatter(pull_times, np.full_like(pull_times, trial), color='blue', label='No Effort' if trial == 0 else "", alpha=0.7)
plt.title('Pull Times for No Effort Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Trial Number')
plt.legend()
plt.subplot(2, 1, 2)
for trial in range(pull_effort.shape[1]):
    pull_times = pull_effort[:, trial]
    pull_times = pull_times[~np.isnan(pull_times)]
    plt.scatter(pull_times, np.full_like(pull_times, trial), color='red', label='Effort' if trial == 0 else "", alpha=0.7)
plt.title('Pull Times for Effort Trials')
plt.xlabel('Time (binned)')
plt.ylabel('Trial Number')
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Plot out the fraction of effort trials that had a complete pull sequence (pull1, pull2, pull3) vs those that did not have a complete pull sequence for each trial
complete_pull_effort = np.sum(~np.isnan(pull_effort), axis=0)
plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
# show as pie chart
labels = ['Complete Pull Sequence', 'Incomplete Pull Sequence']
num_complete = np.sum(complete_pull_effort >1)
num_incomplete = np.sum(complete_pull_effort == 1)
plt.pie([num_complete, num_incomplete], labels=labels, autopct='%1.1f%%', colors=['red', 'gray'], startangle=90)
plt.title('Fraction of Effort Trials with Complete Pull Sequence')
plt.tight_layout()
plt.show()

# %% K-means clustering of initial condition on latent dynamics
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
# reshape latent_dynamics to be (n_trials, n_time, n_components)
num_timepoints_per_trial = binned_spike_data.shape[0] // n_trials
latent_dynamics_trial = latent_dynamics[:n_trials * num_timepoints_per_trial,:].reshape(n_trials, num_timepoints_per_trial,n_components)
print(f"Latent dynamics shape (n_trials, n_time, n_components): {latent_dynamics_trial.shape}")

initCondTimepoint = 45
latentComp = 3
# extract the initial condition for each trial at the specified timepoint
init_conditions = latent_dynamics_trial[:, initCondTimepoint, :3]
# perform k-means clustering on the initial conditions
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(init_conditions)
cluster_labels = kmeans.labels_

# Reduction of Variation analysis of kmeans
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(init_conditions, cluster_labels)
print(f"Silhouette Score for K-means clustering: {silhouette_avg:.4f}")

inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(init_conditions)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(5, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()
#%% plot the clusters in the space of the first 2 principal components
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for cluster in range(n_clusters):
    cluster_points = init_conditions[cluster_labels == cluster]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
               label=f'Cluster {cluster}', alpha=0.9, s=80)

ax.set_title(f'K-means Clustering of Initial Conditions at Timepoint {initCondTimepoint}')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.legend()
plt.tight_layout()
plt.show()
#%%
# sort latent dynamics by clusters and plot the average latent trajectory for each cluster
cluster_latents = []
cluster_leverTraces = []
for cluster in range(n_clusters):
    cluster_trials = np.where(cluster_labels == cluster)[0]
    latent_dynamics_cluster = latent_dynamics_trial[cluster_trials]
    leverTraces_cluster = leverTrace[cluster_trials]
    cluster_leverTraces.append(leverTraces_cluster)
    cluster_latents.append(latent_dynamics_cluster)

plt.figure(figsize=(12, 6))
for cluster in range(len(cluster_latents)):
    for trial in range(cluster_latents[cluster].shape[0]):
        ax = plt.subplot(1, len(cluster_latents), cluster+1)
        ax.plot(cluster_latents[cluster][trial,:, 0], color='gray', alpha=0.5)
        ax.plot(np.mean(cluster_latents[cluster][:,:, 0], axis=0), color='blue', label=f'Cluster {cluster}', alpha=0.9, linewidth=3)
# plot out individual trial latent trajectories for each cluster in light color and the average trajectory for each cluster in bold color
plt.figure(figsize=(12, 6))
for cluster in range(len(cluster_latents)):
    for trial in range(cluster_latents[cluster].shape[0]):
        ax = plt.subplot(1, len(cluster_latents), cluster+1, projection='3d')
        ax.plot(cluster_latents[cluster][trial,:, 0], cluster_latents[cluster][trial,:, 1], cluster_latents[cluster][trial,:, 2], color='gray', alpha=0.5)


# plot out neural trajectories
plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
for cluster in range(len(cluster_latents)):
    mean_trajectory = np.mean(cluster_latents[cluster], axis=0)
    ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], mean_trajectory[:, 2], label=f'Cluster {cluster}', alpha=0.9, linewidth=3)
ax.set_title(f'Average Latent Trajectory for Each Cluster')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.legend()
plt.tight_layout()
plt.show()
# plot the average lever trace for each cluster with overlayed individual trial lever traces
plt.figure(figsize=(12, 6))
for cluster in range(len(cluster_leverTraces)):
    mean_leverTrace = np.mean(cluster_leverTraces[cluster], axis=0)
    plt.plot(mean_leverTrace, label=f'Cluster {cluster}', linewidth=3)
plt.title('Average Lever Trace for Each Cluster')
plt.xlabel('Time (binned)')
plt.ylabel('Lever Position (normalized)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Show percentage of effort trials in each cluster
effort_fraction_per_cluster = []
errort_trials_per_cluster = []
for cluster in range(n_clusters):
    cluster_trials = np.where(cluster_labels == cluster)[0]
    effort_trials_in_cluster = np.sum(isnoeffort[cluster_trials] == 0)
    errort_trials_per_cluster.append(effort_trials_in_cluster)
    total_trials_in_cluster = len(cluster_trials)
    fraction_effort = (effort_trials_in_cluster / total_trials_in_cluster) if total_trials_in_cluster > 0 else 0
    effort_fraction_per_cluster.append(fraction_effort)
# Plot as pie chart
labels = [f'Cluster {i}' for i in range(n_clusters)]
plt.figure(figsize=(6, 6))
plt.pie(effort_fraction_per_cluster, labels=labels, autopct='%1.1f%%')
plt.title('Fraction of Non Effort Trials in Each Cluster')
plt.tight_layout()
plt.show()
print(errort_trials_per_cluster)

labels = [f"Cluster {i}" for i in range(n_clusters)]
effort = np.array(effort_fraction_per_cluster)
non_effort = 1 - effort

x = np.arange(len(labels))

plt.figure(figsize=(6, 4))

plt.bar(x, non_effort, label='Non-effort', color='tab:blue')
plt.bar(x, effort, bottom=non_effort, label='Effort', color='tab:red')

plt.xticks(x, labels)
plt.ylabel("Fraction of trials")
plt.ylim(0, 1)
plt.title("Effort vs non-effort fraction per cluster")
plt.legend()
plt.tight_layout()
plt.show()
# %% Plot our rslds states as a function of the kmeans clusters to see if there is a relationship between the initial condition cluster and the rslds
#  states
plt.figure(figsize=(12, 6))
for cluster in range(n_clusters):
    cluster_trials = np.where(cluster_labels == cluster)[0]
    for state in range(num_states):
        ax = plt.subplot(1, n_clusters, cluster+1)
        state_freq = np.mean(rslds_states[cluster_trials] == state, axis=0)
        # smooth state_freq
        state_freq = gaussian_filter1d(state_freq, sigma=5)
        ax.plot(state_freq, label=f'State {state}')
    ax.set_title(f'States as a Function of K-means Cluster {cluster}')
    ax.set_xlabel('Time (binned)')
plt.ylabel('State Frequency')
plt.legend()
plt.tight_layout()
plt.show()
# %%
