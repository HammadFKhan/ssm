#%%
import autograd.numpy as np
import autograd.numpy.random as npr
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from ssm.util import rle, find_permutation
from matplotlib.colors import ListedColormap
import seaborn as sns
from ssm import SLDS

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm  # Import tqdm for loading bar
from scipy.io import savemat
from scipy.io import loadmat

mat = loadmat(r"C:\Users\khan332\Documents\GitHub\ssm\sequenceProject\neuralData_2d.mat",squeeze_me=True)
DeltaFoverF = mat['neuralData_2d'].T
print(f"Generated data shape: {DeltaFoverF.shape}")

# Parameters
sigma = 5  # Smoothing parameter (in bins)
bin_size_ms = 1  # Bin size in milliseconds

def compute_binned_spike_data(spike_counts, sigma, bin_size_ms):
    """
    Compute continuous firing rates from binned spike data using Gaussian smoothing.
    """
    # Check input dimensions
    if len(spike_counts.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {spike_counts.shape}")
    
    # 2. Bin the data into 20ms bins
    n_neurons = spike_counts.shape[1]
    n_timebins = spike_counts.shape[0]
    bin_size = bin_size_ms
    n_bins = n_timebins // bin_size
    binned_spike_data = np.zeros((n_bins, n_neurons))
    for i in range(n_bins):
        binned_spike_data[i] = spike_counts[i * bin_size:(i + 1) * bin_size].sum(axis=0)
    print("Binned data shape:", binned_spike_data.shape)

    # Convert to Hz (spikes/second) by scaling
    scale_factor = bin_size_ms  # Convert to Hz
    smoothed_spike_data = np.zeros_like(binned_spike_data)
    # Apply Gaussian smoothing to each neuron individually
    for i in range(n_neurons):
        # Explicitly use array indexing
        current_neuron = binned_spike_data[:, i].copy()  # Get copy of this neuron's data
        # Scale first, then smooth
        smoothed_spike_data[:,i] = gaussian_filter1d(current_neuron * scale_factor, sigma=sigma)
    
    return smoothed_spike_data

# Compute firing rates - make sure binned_spike_data is shape (neurons, time)
binned_DeltaFoverF = compute_binned_spike_data(DeltaFoverF, sigma, bin_size_ms)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Choose number of components for latent space (2-3 is good for visualization)
n_components = 10

# Fit PCA to the spike count data
scaler = StandardScaler(with_std=True)
smoothed_spikes_standardized = scaler.fit_transform(binned_DeltaFoverF)

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

plot_spikes_pca(binned_DeltaFoverF,pca,latent_dynamics)
# %%
# Create publication-quality figure with better dimensions
fig, axs = plt.subplots(1, 1, figsize=(6, 8), sharex=True)


# Plot binned spike counts
im1 = axs.imshow(np.transpose(DeltaFoverF), 
                   aspect='auto', 
                   cmap='plasma',  # Scientific colormap that's colorblind-friendly
                   interpolation='none',
                   vmin=0,vmax=12
                   )  # Preserves exact values
axs.set_title("Calcium Data", fontsize=14, fontweight='bold')
axs.set_ylabel("Neurons", fontsize=12)

# Add gridlines to help identify neuron positions
axs.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Custom colorbar with proper positioning
cbar1 = fig.colorbar(im1, ax=axs, fraction=0.046, pad=0.04)
cbar1.set_label("Fluorescence Val", fontsize=12)
cbar1.ax.tick_params(labelsize=10)

# Adjust spacing between subplots
plt.tight_layout()


# Optional: Add a super title
fig.suptitle("Neural Population Activity", fontsize=16, y=1.02)
filename = "Inferred_Spike_state.pdf"
#plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()

# %%

num_timepoints = 151
num_trials = 168
numPC_show = 4
fig, axs = plt.subplots(numPC_show, 1, figsize=(6, 8), sharex=True,sharey=False)
colors = sns.color_palette("viridis", numPC_show)
state_cmap = ListedColormap(colors)
for n in range(numPC_show):
    pc_all_time_points = latent_dynamics[:, n]
    # Reshape PC1 into (num_trials, time_points_per_trial)
    pc_by_trial = pc_all_time_points.reshape(num_trials, num_timepoints)
    # Iterate through each row (which is a trial) in the reshaped pc1_by_trial
    for i in range(num_trials):
        trial_pc_data = pc_by_trial[i, :] # Get PC1 data for the current trial
        time_points_in_trial = np.arange(num_timepoints) # Time index relative to trial start
        
        axs[n].plot(time_points_in_trial, trial_pc_data, alpha=0.25,color=state_cmap(n))
    axs[n].plot(time_points_in_trial, pc_by_trial.mean(0) , alpha=1,color='black')    
    axs[n].set_ylabel('PC', fontsize=14)
    axs[n].grid(axis='y', linestyle='--', alpha=0.6)

axs[n].set_xlabel('Time', fontsize=14)
plt.show()
# %%

# Extract PC1 and PC2 data
num_timepoints = 151
num_trials = 168

pc1_all = latent_dynamics[:, 0].reshape(num_trials, num_timepoints)
pc2_all = latent_dynamics[:, 1].reshape(num_trials, num_timepoints)

# Create figure with two subplots
fig = plt.figure(figsize=(8, 8))

# Left subplot: 2D trajectory in PC1-PC2 space
ax1 = plt.subplot(1, 1, 1)

# Plot mean trajectory with time-based color gradient
mean_pc1 = pc1_all.mean(axis=0)
mean_pc2 = pc2_all.mean(axis=0)

# Create color-coded trajectory for mean
time_colors = np.linspace(0, 1, num_timepoints)
for t in range(num_timepoints-1):
    ax1.plot([mean_pc1[t], mean_pc1[t+1]], [mean_pc2[t], mean_pc2[t+1]], 
             color=plt.cm.plasma(time_colors[t]), linewidth=3, alpha=0.8)

# Add start and end markers
ax1.scatter(mean_pc1[0], mean_pc2[0], s=100, color='green', 
           marker='o', label='Start', zorder=5, edgecolor='black', linewidth=2)
ax1.scatter(mean_pc1[-1], mean_pc2[-1], s=100, color='red', 
           marker='s', label='End', zorder=5, edgecolor='black', linewidth=2)

ax1.set_xlabel('PC1', fontsize=14, fontweight='bold')
ax1.set_ylabel('PC2', fontsize=14, fontweight='bold')
ax1.set_title('Neural State Trajectory\n(PC1-PC2 Space)', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
plt.show()

# %%
