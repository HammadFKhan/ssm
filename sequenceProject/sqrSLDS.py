#%% 
# Load in the warped structure that we have. Format it into the way we want to run the rslds pipeline through it. 
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
#%%


mat_file_path = r"D:\SQLever\Ephys\WarpedSpikes\M1\Day6_M1_warpedSpks.mat"

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

fSave = 'Figures\Day6M1.pdf'
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

#%% Show Plots by PC loading weights

from hammad.Fig_SimSpike import plot_spikes_pca, plot_state_transitions

plot_spikes_pca(binned_spike_data,pca,latent_dynamics)
#%%

# 3. rSlds initialization
num_states = 3
obs_dim = binned_spike_data.shape[1]  # Get 3 from PCA components
latent_dim = 3
# Create the model and initialize its parameters

slds = SLDS(obs_dim, num_states, latent_dim, emissions="poisson_orthog", transitions="recurrent",emission_kwargs=dict(link="softplus"))
binned_spike_data = binned_spike_data.astype(np.int32)
assert binned_spike_data.dtype == int
slds.initialize(binned_spike_data,verbose=1)
# Fit the model using Laplace-EM with a structured variational posterior
q_lem_elbos, q_lem = slds.fit(binned_spike_data, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               num_iters=50,initialize=False)

# Get the posterior mean of the continuous states
q_lem_x = q_lem.mean_continuous_states[0]

# Find the permutation that matches the true and inferred states
rslds_states = slds.most_likely_states(q_lem_x, binned_spike_data)

# Smooth the data under the variational posterior
q_lem_y = slds.smooth(q_lem_x, binned_spike_data)
# Plot ELBO of the model
plt.figure()
plt.plot(q_lem_elbos[1:], label="Laplace-EM")

plt.legend(loc="lower right")
#%%
import getFigures as sqFig
sqFig.plot_rslds_states(rslds_states,num_states,filename = "rslds_Discretestate.pdf")

# %%
sqFig.plot_trajectory_states
# %%
sqFig.plot_trail_pca(latent_dynamics,num_timepoints_per_trial = 250,filename='trialPCA.pdf')

# %%
# %% Plot out inferred latent dynamics
inferred_latent_dynamics =  np.zeros_like(q_lem.mean_continuous_states[0], dtype='int32')
for n in range(q_lem.mean_continuous_states[0].shape[1]):
        inferred_latent_dynamics[:, n] = gaussian_filter1d(q_lem.mean_continuous_states[0][:, n]*100, 5)
inferred_latent_dynamics = inferred_latent_dynamics/100

sqFig.plot_inferred_spks(binned_spike_data,q_lem_y)
sqFig.plot_inferred_latent_dynamics(latent_dynamics,inferred_latent_dynamics)
# %%
from ssm.plots import plot_most_likely_dynamics
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
q_lem_scaled = inferred_latent_dynamics[:150,:]*2
lim = abs(q_lem_scaled).max(axis=0)+1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(q_lem_scaled[:,0], q_lem_scaled[:,1], '-k', lw=1)

plt.title("Most Likely Dynamics, Laplace-EM")
# %%
sqFig.plot_inferred_population(binned_spike_data,q_lem_y)

# %%
# Assuming spike_data is your 2D array with shape (time_total, neurons)
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

trial_time = 250
_,spike_trial_average = make_trials(binned_spike_data,trial_time)
_,inferred_spike_average = make_trials(q_lem_y,trial_time)
_,latent_average = make_trials(inferred_latent_dynamics,trial_time)
# Create publication-quality figure with better dimensions
fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)


# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(spike_trial_average), 
                   aspect='auto', 
                   cmap='plasma',  # Scientific colormap that's colorblind-friendly
                   interpolation='none',
                   vmin = 0, vmax = 8)  # Preserves exact values
axs[0].set_title("Actual Trial-averaged Spike", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Neurons", fontsize=12)

# Add gridlines to help identify neuron positions
axs[0].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Custom colorbar with proper positioning
cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
cbar1.set_label("Spike Count", fontsize=12)
cbar1.ax.tick_params(labelsize=10)

# Plot smoothed spikes with different colormap for visual distinction
im2 = axs[1].imshow(np.transpose(inferred_spike_average), 
                   aspect='auto', 
                   cmap='plasma',  # Different colormap to distinguish from raw data
                   interpolation='none', vmin = 0, vmax = 8)
axs[1].set_title("Inferred Trial-averaged Spike", fontsize=14, fontweight='bold')
axs[1].set_xlabel("Time Bins", fontsize=12)
axs[1].set_ylabel("Neurons", fontsize=12)

# Add gridlines
axs[1].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
# Custom colorbar for smooth data
cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
cbar2.set_label("Inferred Spike Count", fontsize=12)
cbar2.ax.tick_params(labelsize=10)

# Add proper tick formatting for both axes
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_yticks(np.arange(0, np.transpose(binned_spike_data).shape[0], 5))
    ax.set_yticklabels([f"{i}" for i in range(0, np.transpose(binned_spike_data).shape[0], 5)])

# Adjust spacing between subplots
plt.tight_layout()
filename = "TrialAveraged_Spike_state.pdf"
#plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()

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
leverTrace = np.reshape(all_traces_array, all_traces_array.shape[0]*all_traces_array.shape[1])
leverTrace = leverTrace[1::bin_size_ms]
# Create figure with 2 vertical subplots
sqFig.plot_traj_spk_video(q_lem_y,x,leverTrace,
                        max_limit = 20000,
                        start_interval = 20,
                        end_interval = 5,
                        speedup_duration = 2500,
                        fname = "traj_video.mp4")


#%%
transition_matrix = slds.transitions.transition_matrix
sqFig.plot_transition_matrix(transition_matrix)

# %%

sqFig.plot_state_probability(rslds_states)

#%% Save data
fname = 'rsldsPerforamanceM1'

# Import required libraries
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score
import os

# Calculate ARI scores correctly
#rslds_ari = adjusted_rand_score(true_states, rslds_states)  # Fixed: using rslds_states instead of xhat_lem

# Create dictionaries with correct metrics
groundTruthData = {
    'binned_spike_data': binned_spike_data,
    'latent_dynamics': latent_dynamics
}

rsldsData = {
    'latent_states': q_lem_x,
    'discrete_states': rslds_states,
    'inferred_spikes':q_lem_y,
    'transition_matrix': slds.transitions.transition_matrix,
    'q_elbos': q_lem_elbos
}

# Combine all dictionaries into a single dictionary with nested structure
allData = {
    'groundTruth': groundTruthData,
    'rslds': rsldsData
}

# Save to a single .mat file
fileName = f"{fname}.mat"
savemat(fileName, allData)

print(f"Model data saved: {os.path.exists(fileName)}")
