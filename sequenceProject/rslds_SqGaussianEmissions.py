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

#%%
# 3. rSlds initialization
num_states = 5
obs_dim = binned_DeltaFoverF.shape[1]  # Get 3 from PCA components
latent_dim = 2
# Create the model and initialize its parameters

slds = SLDS(obs_dim, num_states, latent_dim, emissions="gaussian_orthog", transitions="recurrent")
binned_spike_data = binned_DeltaFoverF.astype(np.int32)
assert binned_spike_data.dtype == int
sub_spike = binned_spike_data
slds.initialize(sub_spike)
# Fit the model using Laplace-EM with a structured variational posterior
q_lem_elbos, q_lem = slds.fit(sub_spike, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               num_iters=50,initialize=False)

# Get the posterior mean of the continuous states
q_lem_x = q_lem.mean_continuous_states[0]

# Find the permutation that matches the true and inferred states
rslds_states = slds.most_likely_states(q_lem_x, sub_spike)

# Smooth the data under the variational posterior
q_lem_y = slds.smooth(q_lem_x, sub_spike)

# %% Plot ELBO of the model
plt.figure()
plt.plot(q_lem_elbos[1:], label="Laplace-EM")

plt.legend(loc="lower right")

# %% Plot out discrete states with sleep states
from matplotlib.colors import ListedColormap
import seaborn as sns

# Create figure
fig, axs = plt.subplots(2, 1, figsize=(8, 2), sharex=True)
colors = sns.color_palette("viridis", num_states)
state_cmap = ListedColormap(colors)
# Plot inferred states with better colormap and colorbar
im2 = axs[0].imshow(rslds_states[None, :]+1, aspect="auto", cmap=state_cmap, 
                        vmin=1, vmax=num_states, interpolation='none',
                        extent=[0, len(rslds_states), -0.5, 0.5])


axs[0].set_ylabel("RSLDS Inferred $z$", fontsize=12)
axs[0].yaxis.set_ticks([])  # Remove y-axis ticks
cbar2 = fig.colorbar(im2, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)
cbar2.set_label("State", fontsize=10)
cbar2.ax.tick_params(labelsize=10)

axs[1].plot(sleepStates,lw=2)
# Add shared x-axis label
axs[0].set_xlabel("Time Bins", fontsize=12)
axs[0].set_xlim(15000,30000)
# Adjust layout for better spacing
plt.tight_layout()
filename = "rslds_Discretestate.pdf"
#plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
# Show the plot
plt.show()
# %% Compare to inferred calcium traces
fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(binned_DeltaFoverF), 
                   aspect='auto', 
                   cmap='plasma',  # Scientific colormap that's colorblind-friendly
                   interpolation='none',
                   )  # Preserves exact values
axs[0].set_title("True Calcium Activity", fontsize=14, fontweight='bold')
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
                   interpolation='none')
axs[1].set_title("Inferred Calcium Activity", fontsize=14, fontweight='bold')
axs[1].set_xlabel("Time Bins", fontsize=12)
axs[1].set_ylabel("Neurons", fontsize=12)

# Add gridlines
axs[1].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
axs[1].set_xlim(0,30000)
# Custom colorbar for smooth data
cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
cbar2.set_label("Inferred Spike Count", fontsize=12)
cbar2.ax.tick_params(labelsize=10)

# Add proper tick formatting for both axes
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_yticks(np.arange(0, np.transpose(binned_spike_data).shape[0], 20))
    ax.set_yticklabels([f"{i}" for i in range(0, np.transpose(binned_spike_data).shape[0], 20)])

# Adjust spacing between subplots
plt.tight_layout()


# Optional: Add a super title
fig.suptitle("Neural Population Activity", fontsize=16, y=1.02)
filename = "Inferred_Spike_state.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Example data: replace with your actual array
arr = rslds_states   # If states are 0-indexed

# Count occurrences of each unique value
unique, counts = np.unique(arr, return_counts=True)
proportions = counts / counts.sum()

# Consistent color palette (tab10 is colorblind-friendly and distinct)
from matplotlib import cm
num_states = len(unique)
colors = plt.cm.tab10.colors[:num_states]

# Prepare labels with percentages
labels = [f"State {u}" for u in unique]

fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# --- Donut plot (pie chart) ---
wedges, texts, autotexts = ax[0].pie(
    proportions,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='w')
)
ax[0].set_title('Proportion of States', fontsize=16)

# --- Bar plot (time spent in each state) ---
bars = ax[1].bar(unique, counts / 5, color=colors, edgecolor='black', width=0.7)
ax[1].set_xlabel('State (dimension)', fontsize=14)
ax[1].set_ylabel('Time spent (s)', fontsize=14)
ax[1].set_title('Time spent in each state', fontsize=16)
ax[1].set_xticks(unique)
ax[1].grid(axis='y', linestyle='--', alpha=0.6)

# Optional: Add individual data points as scatter (if you have trial-wise data)
# For illustration, here's how you'd add jittered scatter points:
# for i, count in enumerate(counts):
#     y = np.random.normal(loc=count/5, scale=2, size=8)  # replace with actual data
#     ax[1].scatter([unique[i]]*len(y), y, color=colors[i], edgecolor='k', zorder=3)

plt.tight_layout()
filename = "Mouse3_state.pdf"
#plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()

# %%
#%% Save data
fname = 'rslds5StatesSleep'

# Import required libraries
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score
import os

# Calculate ARI scores correctly
#rslds_ari = adjusted_rand_score(true_states, rslds_states)  # Fixed: using rslds_states instead of xhat_lem

# Create dictionaries with correct metrics
groundTruthData = {
    'CalciumData': binned_DeltaFoverF,
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
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm

# Assuming you have these arrays
# rslds_states: your existing state array (already +1 for indexing from 1)
# SleepStates: array with 1=Wake, 2=NREM, 3=REM

# Create a combined DataFrame for analysis
df = pd.DataFrame({
    'DiscreteState': rslds_states+1,
    'SleepState': sleepStates
})

# Map sleep state numbers to labels for better readability
sleep_labels = {1: 'Wake', 2: 'NREM', 3: 'REM'}
df['SleepStateLabel'] = df['SleepState'].map(sleep_labels)

# Get unique states
unique_states = np.sort(df['DiscreteState'].unique())
unique_sleep_states = np.sort(df['SleepState'].unique())

# Define consistent colors for discrete states
num_states = len(unique_states)
colors = plt.cm.tab10.colors[:num_states]
color_map = {state: colors[i] for i, state in enumerate(unique_states)}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [1, 1.5]})

# 1. Pie chart of overall state distribution
state_counts = df['DiscreteState'].value_counts().sort_index()
state_proportions = state_counts / state_counts.sum()

wedges, texts, autotexts = axes[0, 0].pie(
    state_proportions,
    labels=[f'State {s}' for s in state_proportions.index],
    colors=[color_map[s] for s in state_proportions.index],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='w')
)
axes[0, 0].set_title('Overall Proportion of States', fontsize=14)

# 2. Cross-tabulation: Calculate proportion of each discrete state within each sleep state
cross_tab = pd.crosstab(
    df['SleepStateLabel'], 
    df['DiscreteState'], 
    normalize='index'
) * 100  # For percentages

cross_tab.plot(
    kind='bar', 
    stacked=True,
    ax=axes[0, 1], 
    color=[color_map[s] for s in cross_tab.columns]
)
axes[0, 1].set_title('Proportion of Discrete States within Each Sleep State', fontsize=14)
axes[0, 1].set_xlabel('Sleep State', fontsize=12)
axes[0, 1].set_ylabel('Proportion (%)', fontsize=12)
axes[0, 1].legend(title='Discrete State')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.3)

# 3. Time spent in each discrete state (raw counts)
total_counts = df['DiscreteState'].value_counts().sort_index()
axes[1, 0].bar(
    [f'State {s}' for s in total_counts.index],
    total_counts.values / 5,  # Assuming 5 Hz sampling as in your previous code
    color=[color_map[s] for s in total_counts.index],
    edgecolor='black'
)
axes[1, 0].set_title('Time Spent in Each Discrete State', fontsize=14)
axes[1, 0].set_xlabel('Discrete State', fontsize=12)
axes[1, 0].set_ylabel('Time (s)', fontsize=12)
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.3)

# 4. Cross-tabulation: Calculate time spent in each discrete state for each sleep state
# Group by both sleep state and discrete state, then count
grouped_counts = df.groupby(['SleepStateLabel', 'DiscreteState']).size().unstack(fill_value=0)


# Convert counts to time (seconds) assuming 5 Hz sampling
grouped_time = grouped_counts / 5  

# Plot grouped bar chart
grouped_time.plot(
    kind='bar',
    ax=axes[1, 1],
    color=[color_map[s] for s in grouped_time.columns]
)
axes[1, 1].set_title('Time Spent in Discrete States by Sleep State', fontsize=14)
axes[1, 1].set_xlabel('Sleep State', fontsize=12)
axes[1, 1].set_ylabel('Time (s)', fontsize=12)
axes[1, 1].legend(title='Discrete State')
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("Sleep_State_Analysis.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.show()
#%%
