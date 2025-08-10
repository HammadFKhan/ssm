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


mat_file_path = r"D:\SequenceProject\WarpedSpikes\M1\Day6_M1_warpedSpks.mat"

with h5py.File(mat_file_path, 'r') as f:
    # Access the 'warpedSpks' dataset (the MATLAB struct)
    warpedSpks = f['warpedSpks']
    print(f.keys())
    
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
slds.initialize(binned_spike_data)
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
from matplotlib.colors import ListedColormap
import seaborn as sns

# Create figure
fig, axs = plt.subplots(1, 1, figsize=(8, 2), sharex=True)
colors = sns.color_palette("viridis", num_states)
state_cmap = ListedColormap(colors)
# Plot inferred states with better colormap and colorbar
im2 = axs.imshow(rslds_states[None, :]+1, aspect="auto", cmap=state_cmap, 
                        vmin=1, vmax=num_states, interpolation='none',
                        extent=[0, len(rslds_states), -0.5, 0.5])


axs.set_ylabel("RSLDS Inferred $z$", fontsize=12)
axs.yaxis.set_ticks([])  # Remove y-axis ticks
cbar2 = fig.colorbar(im2, ax=axs, orientation="vertical", fraction=0.046, pad=0.04)
cbar2.set_label("State", fontsize=10)
cbar2.ax.tick_params(labelsize=10)


# Add shared x-axis label
axs.set_xlabel("Time Bins", fontsize=12)
axs.set_xlim(0,5000)
# Adjust layout for better spacing
plt.tight_layout()
filename = "rslds_Discretestate.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
# Show the plot
plt.show()

# %%
def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)

    return ax

ax3 = plt.subplot(111)
plot_trajectory(rslds_states, q_lem_y, ax=ax3)
plt.title("Inferred, Laplace-EM")
plt.tight_layout()
# %%
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Setup
num_timepoints_per_trial = 250
numPC_show = 6
colors = sns.color_palette("viridis", numPC_show)
state_cmap = ListedColormap(colors)
num_total_timepoints, _ = latent_dynamics.shape
num_trials = num_total_timepoints // num_timepoints_per_trial

# Prepare subplots: numPC_show rows, 2 columns
fig, axs = plt.subplots(numPC_show, 2, figsize=(12, 2 * numPC_show), sharex=True)
comp_combine = []
for comp in range(numPC_show):
    comp_all_time = latent_dynamics[:, comp]
    comp_by_trial = comp_all_time[:num_trials * num_timepoints_per_trial].reshape(num_trials, num_timepoints_per_trial)
    x = np.linspace(-3.5, 1.5, num_timepoints_per_trial)
    comp_combine.append(comp_by_trial) 
    # First column: Overlay all trials in faint color, then the mean in black
    for trial in range(num_trials):
        axs[comp, 0].plot(x, comp_by_trial[trial, :], alpha=0.25, color=colors[comp])
    axs[comp, 0].plot(x, comp_by_trial.mean(0), alpha=1, color='black', linewidth=2)
    axs[comp, 0].set_ylabel(f'Latent {comp+1}', fontsize=12)
    if comp == numPC_show - 1:
        axs[comp, 0].set_xlabel('Time', fontsize=12)
    axs[comp, 0].set_title('All Trials + Mean')
    
    # Second column: Only mean
    axs[comp, 1].plot(x, comp_by_trial.mean(0), alpha=1, color='black', linewidth=2)
    if comp == numPC_show - 1:
        axs[comp, 1].set_xlabel('Time', fontsize=12)
    axs[comp, 1].set_title('Mean Only')

filename = fSave
plt.tight_layout()
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename}")

# %%
# %% Plot out inferred latent dynamics
plt.figure(figsize=(6, 6))

# Create two subplots (2 rows, 1 column)
ax1 = plt.subplot(211)  # First subplot (top)
ax2 = plt.subplot(212)  # Second subplot (bottom)
inferred_latent_dynamics =  np.zeros_like(q_lem.mean_continuous_states[0], dtype='int32')
for n in range(q_lem.mean_continuous_states[0].shape[1]):
        inferred_latent_dynamics[:, n] = gaussian_filter1d(q_lem.mean_continuous_states[0][:, n]*100, 5)
inferred_latent_dynamics = inferred_latent_dynamics/100

# Plot data on each subplot
ax1.plot(inferred_latent_dynamics[:,0], '-k', lw=1)
ax1.plot(latent_dynamics[:,0], '-r', lw=1)

ax2.plot(inferred_latent_dynamics[:,1], '-k', lw=1)
ax2.plot(latent_dynamics[:,1], '-r', lw=1)


# Set x-axis limits for both subplots (optional)
ax1.set_xlim(0, 1000)
ax2.set_xlim(0, 1000)

plt.tight_layout()  # Improves spacing
plt.show()

plt.figure(figsize=(6, 6))

# Create two subplots (2 rows, 1 column)
ax1 = plt.subplot(111)  # First subplot (top)
inferred_spike_dynamics =  q_lem_y

# Plot data on each subplot
ax1.plot(inferred_spike_dynamics[:,:1], '-r', lw=1)
ax1.plot(binned_spike_data[:,:1], '-k', lw=1)

# Plot the smoothed observations
N = 3
plt.figure(figsize=(8,4))
plt.plot(binned_spike_data[:,:N] + N * np.arange(N), '-k', lw=2)
plt.plot(inferred_spike_dynamics[:,:N] + N * np.arange(N), '-', lw=2)
plt.ylabel("$y$")
plt.xlabel("time")
plt.xlim(0, 1000)

# Set x-axis limits for both subplots (optional)
ax1.set_xlim(0, 1000)

plt.tight_layout()  # Improves spacing
plt.show()

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
# Create publication-quality figure with better dimensions
fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)


# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(binned_spike_data), 
                   aspect='auto', 
                   cmap='plasma',  # Scientific colormap that's colorblind-friendly
                   interpolation='none',
                   vmin = 0, vmax = 8)  # Preserves exact values
axs[0].set_title("Spike Data", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Neurons", fontsize=12)

# Add gridlines to help identify neuron positions
axs[0].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Custom colorbar with proper positioning
cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
cbar1.set_label("Spike Count", fontsize=12)
cbar1.ax.tick_params(labelsize=10)

# Plot smoothed spikes with different colormap for visual distinction
im2 = axs[1].imshow(np.transpose(q_lem_y), 
                   aspect='auto', 
                   cmap='plasma',  # Different colormap to distinguish from raw data
                   interpolation='none', vmin = 0, vmax = 8)
axs[1].set_title("Inferred Spike Counts", fontsize=14, fontweight='bold')
axs[1].set_xlabel("Time Bins", fontsize=12)
axs[1].set_ylabel("Neurons", fontsize=12)

# Add gridlines
axs[1].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
axs[1].set_xlim(0,5500)
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


# Optional: Add a super title
fig.suptitle("Neural Population Activity", fontsize=16, y=1.02)
filename = "Inferred_Spike_state.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()

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
_,inferred_spike_average = make_trials(inferred_spike_dynamics,trial_time)
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
