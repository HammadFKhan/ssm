"""
This script performs a post-hoc analysis of a trained rsLDS model.

It reloads behavioral and neural spike data, then extracts model parameters
and latent variables from the trained model. Finally, it calls a series of
plotting and analysis functions to visualize and interpret the model's output.
"""
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
npr.seed(0)
import h5py
# %% Load data
#loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\DLS\rslds_models\Day9_DLS_warpedSpks_rsldsModel.npz",allow_pickle=True)
#loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\DLS\rslds_models\Day8_DLS_warpedSpks_rsldsModel.npz",allow_pickle=True)
loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\M1\rslds_models\Day6_M1_warpedSpks_rsldsModel.npz",allow_pickle=True)

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
    warpedSpks = f['warpedSpks']
    print(f.keys())
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
        
    # Typically, the struct array contains references
    # Get the first struct in the array (adjust the index for your case)
    struct_ref = 'warpedSpks' # or [0] if 1D
    
    # Dereference to get the struct group
    struct = f[struct_ref]
    
    print("Fields in the struct:")
    print(list(struct.keys()))  # Should list all fields, including 'warpedSpikes'
    
    # Access the 'warpedSpikes' field
    warp_spk = struct['warpSpikes'][:]
    
    # Dereference again to get the actual 3D array data
    #print(f"Shape of warpedSpikes array: {warpedSpikes_data.shape}")
    # Now, warpedSpikes_data is a NumPy array with your 3D data

# Load in warp data and wrangle it into what we need it to be
#warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\DLS\Day9_DLS_warpedSpks_rslds.npy")
#warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\M1\Day6_M1_warpedSpks_rslds.npy")

fSave = 'Figures\Day6M1_'
warp_spk = np.transpose(warp_spk, (3,2,1,0))
[n_trials,n_time,n_neurons,nPulls] = warp_spk.shape

#%%
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
leverTrace = gaussian_filter1d(leverTrace*100, 5)/100

#%% Show Plots by PC loading weights

from hammad.Fig_SimSpike import plot_spikes_pca, plot_state_transitions

plot_spikes_pca(binned_spike_data,pca,latent_dynamics,filename=f"{fSave}latent_weights.pdf")

#%% Plot out the data
import getFigures as sqFig
sqFig.plot_rslds_states(rslds_states,num_states,filename = f"{fSave}rslds_Discretestate.pdf")

sqFig.plot_trajectory_states(rslds_states,latent_dynamics)

sqFig.plot_trail_pca(latent_dynamics,num_timepoints_per_trial = 250,filename=f"{fSave}trialPCA.pdf")

# %% Plot out inferred latent dynamics
inferred_latent_dynamics =  np.zeros_like(q_lem.mean_continuous_states[0], dtype='int32')
for n in range(q_lem.mean_continuous_states[0].shape[1]):
        inferred_latent_dynamics[:, n] = gaussian_filter1d(q_lem.mean_continuous_states[0][:, n]*100, 5)
inferred_latent_dynamics = inferred_latent_dynamics/100

sqFig.plot_inferred_spks(binned_spike_data,q_lem_y)
sqFig.plot_inferred_latent_dynamics(latent_dynamics,inferred_latent_dynamics)
# %%
from matplotlib.colors import ListedColormap
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# Create figure
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

colors = sns.color_palette("viridis", num_states)
state_cmap = ListedColormap(colors)
# Plot inferred states with better colormap and colorbar
im2 = axs[0].imshow(rslds_states[None, :]+1, aspect="auto", cmap=state_cmap, 
                        vmin=1, vmax=num_states, interpolation='none',
                        extent=[0, len(rslds_states), -0.5, 0.5])


axs[0].set_ylabel("RSLDS Inferred $z$", fontsize=12)
axs[0].yaxis.set_ticks([])  # Remove y-axis ticks
cbar2 = fig.colorbar(im2, ax=axs[0], orientation="horizontal", fraction=0.046, pad=0.04)
cbar2.set_label("State", fontsize=10)
cbar2.ax.tick_params(labelsize=10)

# Plot data on each subplot
mins = inferred_latent_dynamics.min()
maxs = inferred_latent_dynamics.max()

axs[1].plot(inferred_latent_dynamics[:,0], '-k', lw=1)
axs[1].plot(inferred_latent_dynamics[:,1], '-r', lw=1)
axs[1].set_ylim(mins,maxs)
# Plot data on each subplot
axs[2].plot(leverTrace, '-k', lw=1)
totalPullTimes = np.array([pull1,pull2,pull3])//bin_size_ms
totalPullTimes = totalPullTimes.squeeze()
ymin = np.max(leverTrace)
for n in range(totalPullTimes.shape[1]):
     totalPullTimes[:,n]= totalPullTimes[:,n]+(250*n)
     axs[2].vlines(totalPullTimes[:,n], ymin, ymin+0.04, colors='k', linestyles='-')
# Set x-axis limits for both subplots (optional)
# Add shared x-axis label
axs[2].set_xlabel("Time Bins", fontsize=12)
axs[2].set_xlim(0,3000)

filename = f"{fSave}rslds_states.pdf"
plt.tight_layout()
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename}")
plt.show()

# %%
sqFig.plot_inferred_population(binned_spike_data,q_lem_y,x_min=0,x_max = 3000,filename = f"{fSave}Inferred_Spike_state.pdf")

sqFig.plot_trial_inferred_spks(binned_spike_data,q_lem_y,inferred_latent_dynamics,
                             trial_time = 250,
                             filename = f"{fSave}TrialAveraged_Spike_state.pdf")
# %%
# Create publication-quality figure with better dimensions
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d

# Assuming latent_dynamics and q_lem_y are already defined as in your snippet
# Below is a mock example setup to illustrate:
from scipy import signal
# Choose number of components for latent space (2-3 is good for visualization)
n_components = 10

# Fit PCA to the spike count data
scaler = StandardScaler(with_std=False)
smoothed_spikes_standardized = scaler.fit_transform(q_lem_y)

pca = PCA(n_components=n_components)
latent_dynamics = pca.fit_transform(smoothed_spikes_standardized)

# Smooth latent dynamics
x = np.zeros_like(latent_dynamics, dtype='int32')
for n in range(latent_dynamics.shape[1]):
    t = gaussian_filter1d(latent_dynamics[:, n]*100, 20)
    sos = signal.butter(1, .05, 'hp', fs=1000/20, output='sos')
    x[:, n] = t
x = x / 100

# %%
#Create figure with 2 vertical subplots
sqFig.plot_traj_spk_video(q_lem_y,x,leverTrace,
                        max_limit = 5000,
                        start_interval = 20,
                        end_interval = 10,
                        speedup_duration = 1000,
                        fname = f"{fSave}traj_videoDLS3.mp4")


#%%
transition_matrix = slds.transitions.transition_matrix
sqFig.plot_transition_matrix(transition_matrix,filename = f"{fSave}Transition_state.pdf")

# %%

sqFig.plot_state_probability(rslds_states,filename = f"{fSave}_state_proportion.pdf")

#%% Find run boundaries and states
changes = np.diff(rslds_states) != 0
run_starts = np.insert(np.where(changes)[0] + 1, 0, 0)
run_ends = np.append(np.where(changes)[0], len(rslds_states) - 1)
run_states = rslds_states[run_starts]
run_lengths = run_ends - run_starts + 1
# Compute average duration for each state
unique_states = np.unique(rslds_states)
avg_duration = []
for state in unique_states:
    durations = run_lengths[run_states == state]
    avg_duration.append(np.mean(durations)*bin_size_ms)

# Plot
plt.figure()
plt.bar(unique_states, avg_duration, tick_label=unique_states)
plt.xlabel('State')
plt.ylabel('Average Duration (ms)')
plt.title('Average Duration of Each State')
plt.tight_layout()
filename = f"{fSave}_state_duration.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()
# %% State dependant dynamics of behavior

# Assuming:
# rslds_states: 1D array of states (length T)
# latent_dynamics: 2D array with shape (T, features), you align on latent_dynamics[:, 2]
# pre_window, post_window defined as in your code
from scipy.signal import correlate
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
unique_states = sorted(set(rslds_states))  # Get all unique states, sorted for consistent order
pre_window = 50
post_window = 200
window_len = pre_window + post_window + 1

aligned_responses_dict = {}  # key: state, value: list of aligned windows
latent_aligned_responses_dict = {}
lever_aligned_responses_dict = {}
changes = np.diff(rslds_states, prepend=rslds_states[0]-1) != 0
aligned_responses_dict = {}
changes = np.diff(rslds_states, prepend=rslds_states[0]-1) != 0

for state in unique_states:
    onsets = np.where((rslds_states == state) & changes)[0]
    latent_windows = []
    lever_windows = []
    for onset in onsets:
        start = onset - pre_window
        end = onset + post_window + 1  # slice end exclusive
        
        # Check bounds
        if (start >= 0) and (end <= len(latent_dynamics[:, 0])) and (end <= len(leverTrace)):
            latent_window = latent_dynamics[:, 0][start:end]
            lever_window = leverTrace[start:end]
            latent_windows.append(latent_window)
            lever_windows.append(lever_window)
    
    aligned_responses_dict[state] = {
        'latent': np.array(latent_windows),
        'lever': np.array(lever_windows)
    }
# Define colors to match your donut chart
 # Colors (customize as needed)
state_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# Example: suppose you have a list of aligned responses (one for each state)
# aligned_responses_dict[state] = np.array(num_trials x window_len)
# window_len, pre_window as defined earlier

time_axis = np.arange(-pre_window, post_window + 1) * bin_size_ms

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Subplot 1: Lever aligned responses with SEM
for state, data in aligned_responses_dict.items():
    lever_responses = data['lever']
    mean_resp = np.mean(lever_responses, axis=0)
    sem_resp = np.std(lever_responses, axis=0) / np.sqrt(lever_responses.shape[0])
    axs[0].plot(time_axis, mean_resp, label=f'State {state}', color=state_colors[state])
    axs[0].fill_between(time_axis, mean_resp - sem_resp, mean_resp + sem_resp,
                        color=state_colors[state], alpha=0.2)
axs[0].axvline(0, color='gray', linestyle='--', linewidth=1)
axs[0].set_xlabel('(s)')
axs[0].set_ylabel('Lever Aligned Response')
axs[0].set_title('Lever State-Aligned Responses')
axs[0].legend()

# Subplot 2: Latent aligned responses with SEM
for state, data in aligned_responses_dict.items():
    latent_responses = data['latent']
    mean_resp = np.mean(latent_responses, axis=0)
    sem_resp = np.std(latent_responses, axis=0) / np.sqrt(latent_responses.shape[0])
    axs[1].plot(time_axis, mean_resp, label=f'State {state}', color=state_colors[state])
    axs[1].fill_between(time_axis, mean_resp - sem_resp, mean_resp + sem_resp,
                        color=state_colors[state], alpha=0.2)
axs[1].axvline(0, color='gray', linestyle='--', linewidth=1)
axs[1].set_xlabel('(s)')
axs[1].set_ylabel('Latent Aligned Response')
axs[1].set_title('Latent State-Aligned Responses')
axs[1].legend()

# Subplot 3: Cross-correlation of mean lever and latent responses per state
for state, data in aligned_responses_dict.items():
    lever_mean = np.mean(data['lever'], axis=0)
    latent_mean = np.mean(data['latent'], axis=0)
    # Center signals
    lever_mean_centered = lever_mean - np.mean(lever_mean)
    latent_mean_centered = latent_mean - np.mean(latent_mean)
    cross_corr = correlate(latent_mean_centered,lever_mean_centered, mode='full')
    cross_corr /= np.max(np.abs(cross_corr))  # normalize
    lags = np.arange(-len(lever_mean) + 1, len(lever_mean)) * bin_size_ms
    axs[2].plot(lags, cross_corr, label=f'State {state}', color=state_colors[state])
axs[2].set_xlabel('Lag (s)')
axs[2].set_ylabel('Normalized Cross-correlation')
axs[2].set_title('Cross-correlation Lever vs Latent')
axs[2].legend()

plt.tight_layout()
filename = f"{fSave}_state_aligned_latent.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()

#%%
# pulls: 3 x timepoints binary array (1 = pull, 0 = no pull)
pull_1 = totalPullTimes[0, :]  # first pull row
inputs = np.concatenate((pull1,pull2,pull3),axis = 1)
inputs = pull3
inputs = inputs//bin_size_ms
event_matrix = np.zeros((n_trials, 250))
for trial in range(n_trials):
    for pull_time in inputs[trial]:
        event_matrix[trial, pull_time:pull_time+1] = 1  # Or use real-valued feature
event_matrix = np.reshape(event_matrix,-1)

aligned_pull_prob_dict = {}
changes = np.diff(rslds_states, prepend=rslds_states[0]-1) != 0

for state in unique_states:
    onsets = np.where((rslds_states == state) & changes)[0]
    aligned_pulls = []
    for onset in onsets:
        start = onset - pre_window
        end = onset + post_window + 1  # slice end exclusive
        if (start >= 0) and (end <= len(event_matrix)):
            window = event_matrix[start:end]
            aligned_pulls.append(window)
    aligned_pulls = np.array(aligned_pulls)
    
    # Calculate probability of pull at each time point (mean across occurrences)
    pull_prob = np.mean(aligned_pulls, axis=0)
    aligned_pull_prob_dict[state] = pull_prob
    
time_axis = np.arange(-pre_window, post_window + 1) * bin_size_ms

plt.figure(figsize=(6,4))
for state, pull_prob in aligned_pull_prob_dict.items():
    plt.plot(time_axis, pull_prob, label=f'State {state}', color=state_colors[state])

plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('(ms)')
plt.ylabel('Probability of Pull')
plt.title('Pull Probability Aligned to State Onsets (Pull 1)')
plt.legend()
plt.tight_layout()
plt.show()
#%%
# pulls: 3 x timepoints binary array (1 = pull, 0 = no pull)
totalPullTimes = np.array([pull1,pull2,pull3-500,pull3])//bin_size_ms
totalPullTimes = totalPullTimes.squeeze()
for n in range(totalPullTimes.shape[1]):
     totalPullTimes[:,n]= totalPullTimes[:,n]+(250*n)

# Pull 1 row (binary vector)
unique_states = np.unique(rslds_states)
pre_window = 50
post_window = 200
window_len = pre_window + post_window + 1
time_axis = np.arange(-pre_window, post_window + 1) * bin_size_ms
num_pulls = 1  # e.g. 3 pulls
avg_rel_time = []
fig, axs = plt.subplots(1, num_pulls, figsize=(5 * num_pulls, 4), sharey=True)

for pull_num in range(num_pulls):
    aligned_states_around_pulls = []
    pulls = totalPullTimes[pull_num, :]
    for idx in pulls:
        start = idx - pre_window
        end = idx + post_window + 1
        if start >= 0 and end <= len(rslds_states):
            window = rslds_states[start:end]
            aligned_states_around_pulls.append(window)
    aligned_states_around_pulls = np.array(aligned_states_around_pulls)
    # Calculate average relative times of subsequent pulls within the window
    align_pull_times = totalPullTimes[pull_num, :]  # reference align pull times
    
    for next_pull_num in range(pull_num + 1, 4):
        next_pull_times = totalPullTimes[next_pull_num, :]
        # Compute relative times of next pulls relative to current aligned pull
        relative_times = next_pull_times - align_pull_times
        # Keep only those that fall within the plotting window
        valid_times = np.mean(relative_times[(relative_times >= -pre_window) & (relative_times <= post_window)]) * bin_size_ms
        avg_rel_time.append(valid_times)  # convert to ms

    if len(aligned_states_around_pulls) > 0:
        state_probabilities = {state: np.mean(aligned_states_around_pulls == state, axis=0) for state in unique_states}
    else:
        state_probabilities = {state: np.zeros(window_len) for state in unique_states}
    # Normalization: subtract mean prewindow probability per state
    '''
    for state in unique_states:
        prewindow_mean = np.mean(state_probabilities[state][:pre_window])
        state_probabilities[state] = state_probabilities[state] - prewindow_mean
    '''
    ax = axs[pull_num] if num_pulls > 1 else axs
    for i, state in enumerate(unique_states):
        ax.plot(time_axis, state_probabilities[state], label=f'State {state}', color=state_colors[i])
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    for n in range(len(avg_rel_time)):
        if n == len(avg_rel_time)-1:
            ax.axvline(avg_rel_time[n], color='blue', linestyle='--', linewidth=1, alpha=0.7)
        else:
            ax.axvline(avg_rel_time[n], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('(ms)')
    if pull_num == 0:
        ax.set_ylabel('State Probability')
    ax.set_title(f'Pull {pull_num + 1}')
    if pull_num == 3:
        ax.set_title(f'Reward')
    ax.legend()
plt.tight_layout()
filename = f"{fSave}_pull_aligned_state.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()
# %% Plot each linear system
from ssm.plots import plot_dynamics_2d
# Iterate over all discrete states
num_states = slds.K  # Number of discrete states
lim = abs(latent_dynamics).max(axis=0) + 50 # Define limits based on latent dynamics
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])
import seaborn as sns
color_names = ["windows blue", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)

num_states = slds.dynamics.As.shape[0]
fig, axs = plt.subplots(1, num_states, figsize=(6 * num_states, 6))

for k in range(num_states):
    dynamics_matrix = np.squeeze(slds.dynamics.As[k, :, :])[:2, :2]
    bias_vector = np.squeeze(slds.dynamics.bs[k, :])[:2]

    ax = axs[k] if num_states > 1 else axs  # Handle case if only one state
    
    plot_dynamics_2d(dynamics_matrix, bias_vector, mins=mins, maxs=maxs, color=colors[k], axis = axs[k])
    
    # Overlay latent dynamics trajectory if desired (example)
    # ax.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1)

    ax.set_title(f"Flow Field for State {k+1}")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")

plt.tight_layout()
filename = f"{fSave}_state_flow.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()
# %%
# %%
from ssm.plots import plot_most_likely_dynamics
def make_trials(neural_data,trial_time):
        time_total, n_neurons = neural_data.shape
        spike_data_trial = np.zeros_like(neural_data)
        # Calculate number of trials automatically
        n_trials = time_total // trial_time

        # Check if the time dimension is cleanly divisible by 600
        if time_total % trial_time != 0:
            # Truncate data to make it cleanly divisible
            spike_data_trial = neural_data[:n_trials*trial_time, :]
            print(f"Warning: Truncated {time_total - n_trials*trial_time} time points")

        # First reshape to (n_trials, 600, n_neurons)
        reshaped = spike_data_trial.reshape(n_trials, trial_time, n_neurons)

        # Then transpose to get (600, n_neurons, n_trials)
        spike_data_trials = np.transpose(reshaped, (1, 2, 0))

        # Now take the mean across trials (axis=2, not axis=3 as Python is 0-indexed)
        trial_average = np.mean(spike_data_trials, axis=2)

        return spike_data_trials,trial_average
_,latent_average = make_trials(inferred_latent_dynamics,250)
latent_average = latent_average
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(inferred_latent_dynamics).max(axis=0)*1.3
totalPullTimes = np.array([0,np.mean(pull1),np.mean(pull2),np.mean(pull3),np.mean(pull3)+500])//bin_size_ms
totalPullTimes = totalPullTimes.astype(int)
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(inferred_latent_dynamics[:,0], inferred_latent_dynamics[:,1],'-k', lw=1,alpha = 0.3)
plt.plot(latent_average[:,0], latent_average[:,1],'-k', lw=2,alpha = 0.5)
plt.scatter(latent_average[totalPullTimes,0],latent_average[totalPullTimes,1],s=24,c='green')
plt.title("Sequence Dynamics")
filename = f"{fSave}likely_state.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename}")
plt.show()
# %% Here we want to retrain the model to extract the 
# latent contineous states and their corresponding eigen values 
#%% Load in expanded data
#loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\DLS\rslds_models\Day9_DLS_warpedSpks_rsldsModel_expand.npz",allow_pickle=True)
loaded = np.load(r"D:\SQLever\Ephys\WarpedSpikes\M1\rslds_models\Day6_M1_warpedSpks_rsldsModel_expand.npz",allow_pickle=True)
rsldsData = loaded['rsldsData'].item()  # .item() gets the actual dictionary
q_elbos = rsldsData['q_elbos']
slds_expand = rsldsData['rslds_model']
rslds_states = rsldsData['discrete_states']
latent_dynamics = groundTruthData['latent_dynamics']
inferred_latent_dynamics = rsldsData['latent_states']
q_lem = rsldsData['q_lem']
q_lem_y = rsldsData['inferred_spikes']
# %%
from ssm.plots import plot_dynamics_2d

num_states = slds_expand.K  # Number of discrete states
lim = abs(latent_dynamics).max(axis=0) + 50  # Define limits based on latent dynamics
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])
import seaborn as sns
color_names = ["windows blue", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)

for k in range(num_states):
    # Extract dynamics for state k
    dynamics_matrix = np.squeeze(slds_expand.dynamics.As[k, :, :])
    dynamics_matrix = dynamics_matrix[:2,:2]
    bias_vector = np.squeeze(slds_expand.dynamics.bs[k, :])
    bias_vector = bias_vector[:2]
    # Create a new figure for each state's flow field
    plt.figure(figsize=(6, 6))
    plot_dynamics_2d(dynamics_matrix, bias_vector, mins=mins, maxs=maxs,color=colors[k])
    
    # Overlay the latent dynamics trajectory
    #plt.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1)
    
    # Add title and labels
    plt.title(f"Flow Field for State {k+1}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()
# %% Time constant of dynamic matrix eigenvalues
from scipy.linalg import eig
eigenVal = eig(slds_expand.dynamics.As[0,:,:])
x = np.real(eigenVal[0])
y = np.imag(eigenVal[0])
# Color by position in array (dimension index)
indices = np.arange(len(eigenVal[0]))
# Normalize for colormap (optional, but recommended)
norm_indices = (indices - indices.min()) / (indices.max() - indices.min())

plt.figure(figsize=(6, 6))
sc = plt.scatter(x, y, c=norm_indices, cmap='RdBu',
                 edgecolors=[0.4,0.4,0.4],
                 linewidths=0.1, s=40)
#plt.xlim(0.1,1)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Eigenvalues')
# Create colorbar
cb = plt.colorbar(sc, label='Dimension index')

# Set colorbar ticks and labels to match dimension indices
num_dims = len(eigenVal[0])
tick_locs = np.linspace(0, 1, num_dims)  # positions in normalized range
cb.set_ticks(tick_locs)
cb.set_ticklabels([str(i) for i in range(num_dims)])
filename = f"{fSave}eigenValues.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename}")
plt.show()
#%% Calculate eigen values from each discrete states
from scipy.linalg import eig
# Here we only grab the L1 and L2 dim for visualization to the model
plt.figure(figsize=(6, 6))
indices = np.arange(num_states)
norm_indices = (indices - indices.min()) / (indices.max() - indices.min())
scatter_handles = []

for k in range(num_states):
    dynamics_matrix = np.squeeze(slds.dynamics.As[k, :, :])[:2, :2]
    eigenVal = eig(dynamics_matrix)
    x = np.real(eigenVal[0])
    y = np.imag(eigenVal[0])
    color_val = norm_indices[k]
    sc = plt.scatter(x, y, c=colors[k], cmap=state_colors,
                     edgecolors=[0.4,0.4,0.4],
                     linewidths=0.1, s=40)
    scatter_handles.append(sc)  # Optional: gather for colorbar

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Eigenvalues')
cb = plt.colorbar(sc, label='Dimension index')

num_dims = num_states
tick_locs = np.linspace(0, 1, num_dims)
cb.set_ticks(tick_locs)
cb.set_ticklabels([str(i) for i in range(num_dims)])
filename = f"{fSave}eigenValues.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename}")
plt.show()






#%% Plot out time constants of eigenvalues
time_constant = np.abs(1/np.log(np.abs(eigenVal[0])))
dim = np.arange(len(time_constant))+1
plt.figure(figsize=(6, 6))
plt.bar(dim,time_constant)
# %% Behavioural annotation of states
# Here we want to figure out how each state coincodes
# with the probability of a specific behaviour 
# that the animal is undergoing

sqFig.plot_state_probability(rslds_states,filename = f"{fSave}_state_proportion.pdf")
# Assuming:
# rslds_states: 1D array of states (length T)
# latent_dynamics: 2D array with shape (T, features), you align on latent_dynamics[:, 2]
# pre_window, post_window defined as in your code
from scipy.signal import correlate
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
unique_states = sorted(set(rslds_states))  # Get all unique states, sorted for consistent order
pre_window = 50
post_window = 200
window_len = pre_window + post_window + 1

aligned_responses_dict = {}  # key: state, value: list of aligned windows
latent_aligned_responses_dict = {}
lever_aligned_responses_dict = {}
changes = np.diff(rslds_states, prepend=rslds_states[0]-1) != 0
aligned_responses_dict = {}
changes = np.diff(rslds_states, prepend=rslds_states[0]-1) != 0

for state in unique_states:
    onsets = np.where((rslds_states == state) & changes)[0]
    latent_windows = []
    lever_windows = []
    for onset in onsets:
        start = onset - pre_window
        end = onset + post_window + 1  # slice end exclusive
        
        # Check bounds
        if (start >= 0) and (end <= len(latent_dynamics[:, 0])) and (end <= len(leverTrace)):
            latent_window = latent_dynamics[:, 0][start:end]
            lever_window = leverTrace[start:end]
            latent_windows.append(latent_window)
            lever_windows.append(lever_window)
    
    aligned_responses_dict[state] = {
        'latent': np.array(latent_windows),
        'lever': np.array(lever_windows)
    }
# Define colors to match your donut chart
 # Colors (customize as needed)
state_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# Example: suppose you have a list of aligned responses (one for each state)
# aligned_responses_dict[state] = np.array(num_trials x window_len)
# window_len, pre_window as defined earlier

time_axis = np.arange(-pre_window, post_window + 1) * bin_size_ms

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Subplot 1: Lever aligned responses with SEM
for state, data in aligned_responses_dict.items():
    lever_responses = data['lever']
    mean_resp = np.mean(lever_responses, axis=0)
    sem_resp = np.std(lever_responses, axis=0) / np.sqrt(lever_responses.shape[0])
    axs[0].plot(time_axis, mean_resp, label=f'State {state}', color=state_colors[state])
    axs[0].fill_between(time_axis, mean_resp - sem_resp, mean_resp + sem_resp,
                        color=state_colors[state], alpha=0.2)
axs[0].axvline(0, color='gray', linestyle='--', linewidth=1)
axs[0].set_xlabel('(s)')
axs[0].set_ylabel('Lever Aligned Response')
axs[0].set_title('Lever State-Aligned Responses')
axs[0].legend()

# Subplot 2: Latent aligned responses with SEM
for state, data in aligned_responses_dict.items():
    latent_responses = data['latent']
    mean_resp = np.mean(latent_responses, axis=0)
    sem_resp = np.std(latent_responses, axis=0) / np.sqrt(latent_responses.shape[0])
    axs[1].plot(time_axis, mean_resp, label=f'State {state}', color=state_colors[state])
    axs[1].fill_between(time_axis, mean_resp - sem_resp, mean_resp + sem_resp,
                        color=state_colors[state], alpha=0.2)
axs[1].axvline(0, color='gray', linestyle='--', linewidth=1)
axs[1].set_xlabel('(s)')
axs[1].set_ylabel('Latent Aligned Response')
axs[1].set_title('Latent State-Aligned Responses')
axs[1].legend()

# Subplot 3: Cross-correlation of mean lever and latent responses per state
for state, data in aligned_responses_dict.items():
    lever_mean = np.mean(data['lever'], axis=0)
    latent_mean = np.mean(data['latent'], axis=0)
    # Center signals
    lever_mean_centered = lever_mean - np.mean(lever_mean)
    latent_mean_centered = latent_mean - np.mean(latent_mean)
    cross_corr = correlate(latent_mean_centered,lever_mean_centered, mode='full')
    cross_corr /= np.max(np.abs(cross_corr))  # normalize
    lags = np.arange(-len(lever_mean) + 1, len(lever_mean)) * bin_size_ms
    axs[2].plot(lags, cross_corr, label=f'State {state}', color=state_colors[state])
axs[2].set_xlabel('Lag (s)')
axs[2].set_ylabel('Normalized Cross-correlation')
axs[2].set_title('Cross-correlation Lever vs Latent')
axs[2].legend()

plt.tight_layout()
filename = f"{fSave}_state_aligned_latent.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()

#%%
# pulls: 3 x timepoints binary array (1 = pull, 0 = no pull)
pull_1 = totalPullTimes[0, :]  # first pull row
inputs = np.concatenate((pull1,pull2,pull3),axis = 1)
inputs = pull3
inputs = inputs//bin_size_ms
event_matrix = np.zeros((n_trials, 250))
for trial in range(n_trials):
    for pull_time in inputs[trial]:
        event_matrix[trial, pull_time:pull_time+1] = 1  # Or use real-valued feature
event_matrix = np.reshape(event_matrix,-1)

aligned_pull_prob_dict = {}
changes = np.diff(rslds_states, prepend=rslds_states[0]-1) != 0

for state in unique_states:
    onsets = np.where((rslds_states == state) & changes)[0]
    aligned_pulls = []
    for onset in onsets:
        start = onset - pre_window
        end = onset + post_window + 1  # slice end exclusive
        if (start >= 0) and (end <= len(event_matrix)):
            window = event_matrix[start:end]
            aligned_pulls.append(window)
    aligned_pulls = np.array(aligned_pulls)
    
    # Calculate probability of pull at each time point (mean across occurrences)
    pull_prob = np.mean(aligned_pulls, axis=0)
    aligned_pull_prob_dict[state] = pull_prob
    
time_axis = np.arange(-pre_window, post_window + 1) * bin_size_ms

plt.figure(figsize=(6,4))
for state, pull_prob in aligned_pull_prob_dict.items():
    plt.plot(time_axis, pull_prob, label=f'State {state}', color=state_colors[state])

plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('(ms)')
plt.ylabel('Probability of Pull')
plt.title('Pull Probability Aligned to State Onsets (Pull 1)')
plt.legend()
plt.tight_layout()
plt.show()
#%%
# pulls: 3 x timepoints binary array (1 = pull, 0 = no pull)
totalPullTimes = np.array([pull1,pull2,pull3-500,pull3])//bin_size_ms
totalPullTimes = totalPullTimes.squeeze()
for n in range(totalPullTimes.shape[1]):
     totalPullTimes[:,n]= totalPullTimes[:,n]+(250*n)

# Pull 1 row (binary vector)
unique_states = np.unique(rslds_states)
pre_window = 50
post_window = 200
window_len = pre_window + post_window + 1
time_axis = np.arange(-pre_window, post_window + 1) * bin_size_ms
num_pulls = 1  # e.g. 3 pulls
avg_rel_time = []
fig, axs = plt.subplots(1, num_pulls, figsize=(5 * num_pulls, 4), sharey=True)

for pull_num in range(num_pulls):
    aligned_states_around_pulls = []
    pulls = totalPullTimes[pull_num, :]
    for idx in pulls:
        start = idx - pre_window
        end = idx + post_window + 1
        if start >= 0 and end <= len(rslds_states):
            window = rslds_states[start:end]
            aligned_states_around_pulls.append(window)
    aligned_states_around_pulls = np.array(aligned_states_around_pulls)
    # Calculate average relative times of subsequent pulls within the window
    align_pull_times = totalPullTimes[pull_num, :]  # reference align pull times
    
    for next_pull_num in range(pull_num + 1, 4):
        next_pull_times = totalPullTimes[next_pull_num, :]
        # Compute relative times of next pulls relative to current aligned pull
        relative_times = next_pull_times - align_pull_times
        # Keep only those that fall within the plotting window
        valid_times = np.mean(relative_times[(relative_times >= -pre_window) & (relative_times <= post_window)]) * bin_size_ms
        avg_rel_time.append(valid_times)  # convert to ms

    if len(aligned_states_around_pulls) > 0:
        state_probabilities = {state: np.mean(aligned_states_around_pulls == state, axis=0) for state in unique_states}
    else:
        state_probabilities = {state: np.zeros(window_len) for state in unique_states}
    # Normalization: subtract mean prewindow probability per state
    '''
    for state in unique_states:
        prewindow_mean = np.mean(state_probabilities[state][:pre_window])
        state_probabilities[state] = state_probabilities[state] - prewindow_mean
    '''
    ax = axs[pull_num] if num_pulls > 1 else axs
    for i, state in enumerate(unique_states):
        ax.plot(time_axis, state_probabilities[state], label=f'State {state}', color=state_colors[i])
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    for n in range(len(avg_rel_time)):
        if n == len(avg_rel_time)-1:
            ax.axvline(avg_rel_time[n], color='blue', linestyle='--', linewidth=1, alpha=0.7)
        else:
            ax.axvline(avg_rel_time[n], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('(ms)')
    if pull_num == 0:
        ax.set_ylabel('State Probability')
    ax.set_title(f'Pull {pull_num + 1}')
    if pull_num == 3:
        ax.set_title(f'Reward')
    ax.legend()
plt.tight_layout()
filename = f"{fSave}_pull_aligned_state.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()
# %%
dynamics_matrix = np.array([(0.9, 0),(0.0,.1)])
bias_vector = np.array([0,0])

from scipy.linalg import eig
a1, a2 = eigenVal  # eigenvalues less than 1 for stability
x1_0, x2_0 = 30, -35
x = np.array([x1_0, x2_0])
trajectory = [x.copy()]
k = np.arange(0, 100)
for _ in range(int(k[-1])):
    x = dynamics_matrix @ x + bias_vector  # matrix multiplication for coupled system
    trajectory.append(x.copy())

trajectory = np.array(trajectory)
lim = abs(trajectory).max(axis=0)*1.3
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])
# Create a new figure for each state's flow field
plt.figure(figsize=(6, 6))
plot_dynamics_2d(dynamics_matrix, bias_vector,mins=mins,maxs= maxs)

# Overlay the latent dynamics trajectory
#plt.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1)
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory', color='orange')

# Add title and labels
plt.title(f"Flow Field for State")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()
# %%
num_states = slds_expand.dynamics.As.shape[2]
fig= plt.subplots(1, 1, figsize=(6, 6))

k = 2
dynamics_matrix = np.squeeze(slds_expand.dynamics.As[k, :, :])[:2, :2]
bias_vector = np.squeeze(slds_expand.dynamics.bs[k, :])[:2]
dynamics_matrix = np.array([(0.99,0),(0,0.99)])
bias_vector = [0,0]
x1_0, x2_0 = -60, -60
x = np.array([x1_0, x2_0])
trajectory = [x.copy()]
k = np.arange(0, 2000)
for _ in range(int(k[-1])):
    x = dynamics_matrix @ x + bias_vector  # matrix multiplication for coupled system
    trajectory.append(x.copy())

trajectory = np.array(trajectory)
lim = abs(trajectory).max(axis=0)*1.3
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])

plot_dynamics_2d(dynamics_matrix, bias_vector, mins=mins, maxs=maxs)

# Overlay latent dynamics trajectory if desired (example)
# ax.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1)
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory', color='orange')


ax.set_xlabel("Latent Dimension 1")
ax.set_ylabel("Latent Dimension 2")

plt.tight_layout()
filename = f"{fSave}_state_flow.pdf"
plt.show()
# %%
