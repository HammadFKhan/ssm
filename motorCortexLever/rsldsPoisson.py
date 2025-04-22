"""
rSLDS Poisson model for spiking data
===============================
"""
# %%
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

#%%
spike_data = np.load(r'C:\Users\khan332\Documents\GitHub\ssm\hammad\spikeData.npy')
# Transpose data here
spike_data = spike_data.T
# Check the shape of the loaded data
print(f"Generated data shape: {spike_data.shape}")

# Parameters
sigma = 4  # Smoothing parameter (in bins)
bin_size_ms = 10  # Bin size in milliseconds

def compute_binned_spike_data(spike_counts, sigma, bin_size_ms):
    """
    Compute continuous firing rates from binned spike data using Gaussian smoothing.
    """
    # Check input dimensions
    if len(spike_counts.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {spike_counts.shape}")
    
    # 2. Bin the data into 20ms bins
    n_neurons = spike_data.shape[1]
    n_timebins = spike_data.shape[0]
    bin_size = bin_size_ms
    n_bins = n_timebins // bin_size
    binned_spike_data = np.zeros((n_bins, n_neurons))
    for i in range(n_bins):
        binned_spike_data[i] = spike_data[i * bin_size:(i + 1) * bin_size].sum(axis=0)
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
sub_spike = binned_spike_data[:10000,:]
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
# %%
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
axs.set_xlim(0,1400)
# Adjust layout for better spacing
plt.tight_layout()
filename = "rslds_Discretestate.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
# Show the plot
plt.show()


# %%

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

ax3 = plt.subplot(133)
plot_trajectory(rslds_states, q_lem_y, ax=ax3)
plt.title("Inferred, Laplace-EM")
plt.tight_layout()

#%% Save data
fname = 'rsldsPerforamanceDataMouse2'

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
#%% Save as a video
from PIL import Image
import io
from ssm.plots import plot_most_likely_dynamics

# Example data (replace with your actual q_lem_scaled)
# q_lem_scaled = np.random.randn(100, 2)
q_lem_scaled = inferred_latent_dynamics[:160,:]*10
frames = []
for i in range(2, len(q_lem_scaled)+1):  # start from 2 to show a line
    fig, ax = plt.subplots()
    lim = abs(q_lem_scaled).max(axis=0)+1
    plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
    ax.plot(q_lem_scaled[:i,0], q_lem_scaled[:i,1], '-k', lw=1)
    ax.set_xlim(np.min(q_lem_scaled[:,0]), np.max(q_lem_scaled[:,0]))
    ax.set_ylim(np.min(q_lem_scaled[:,1]), np.max(q_lem_scaled[:,1]))
    ax.set_xlabel('q_lem_scaled[:,0]')
    ax.set_ylabel('q_lem_scaled[:,1]')
    plt.tight_layout()
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    # Open with PIL and convert to 'P' mode for GIF
    img = Image.open(buf).convert('P')
    frames.append(img)
    buf.close()
    # Save as GIF
frames[0].save(
    'timeseries_animation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=40,   # milliseconds per frame
    loop=0         # loop forever
)
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
axs[1].set_xlim(0,2500)
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

#%% Show Trial averaged data

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

trial_time = 150
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
#%%

plt.figure(figsize=(6,6))
ax = plt.subplot(111)
q_lem_scaled = latent_average*35
lim = abs(q_lem_scaled).max(axis=0)+1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(q_lem_scaled[:,0], q_lem_scaled[:,1], '-k', lw=2)
plt.scatter(q_lem_scaled[0,0],q_lem_scaled[0,1],62)
plt.scatter(q_lem_scaled[74,0],q_lem_scaled[74,1],62)
plt.scatter(q_lem_scaled[88,0],q_lem_scaled[88,1],62)
plt.scatter(q_lem_scaled[100,0],q_lem_scaled[100,1],62)
plt.title("Most Likely Dynamics, Laplace-EM")
plt.tight_layout()
filename = "TrialAveraged_Trajectory_state.pdf"
#plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()
#%%

sim_states, sim_latent, sim_observations = slds.sample(1500, with_noise=1)
sim_latent_smooth = np.zeros_like(sim_latent, dtype='int32')
for n in range(sim_latent.shape[1]):
    sim_latent_smooth[:, n] = gaussian_filter1d(sim_latent[:, n]*1000, 5)

sim_latent = sim_latent_smooth/1000   
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(sim_latent).max(axis=0) + 1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(sim_latent[:,0], sim_latent[:,1], '-k', lw=1)

plt.title("Simulated Dynamics")
makeVid=False
if makeVid:
    from PIL import Image
    import io
    from ssm.plots import plot_most_likely_dynamics

    # Example data (replace with your actual q_lem_scaled)
    # q_lem_scaled = np.random.randn(100, 2)
    q_lem_scaled = sim_latent[:160,:]*10
    frames = []
    for i in range(2, len(q_lem_scaled)+1):  # start from 2 to show a line
        fig, ax = plt.subplots()
        lim = abs(q_lem_scaled).max(axis=0)+1
        plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
        ax.plot(q_lem_scaled[:i,0], q_lem_scaled[:i,1], '-k', lw=1)
        ax.set_xlim(np.min(q_lem_scaled[:,0]), np.max(q_lem_scaled[:,0]))
        ax.set_ylim(np.min(q_lem_scaled[:,1]), np.max(q_lem_scaled[:,1]))
        ax.set_xlabel('q_lem_scaled[:,0]')
        ax.set_ylabel('q_lem_scaled[:,1]')
        plt.tight_layout()
        
        # Save to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        # Open with PIL and convert to 'P' mode for GIF
        img = Image.open(buf).convert('P')
        frames.append(img)
        buf.close()
        # Save as GIF
    frames[0].save(
        'timeseries_animation_simulated.gif',
        save_all=True,
        append_images=frames[1:],
        duration=40,   # milliseconds per frame
        loop=0         # loop forever
    )

# Trial average simulated data
trial_time = 150
# Assuming spike_data is your 2D array with shape (time_total, neurons)
def make_trials(neural_data,trial_time):
    neural_data = neural_data.astype(np.int32)
    time_total, n_neurons = neural_data.shape
    spike_data_trial = np.zeros_like(neural_data)
    # Calculate number of trials automatically
    n_trials = time_total // trial_time

    # Check if the time dimension is cleanly divisible by 600
    if time_total % trial_time != 0:
        # Truncate data to make it cleanly divisible
        spike_data_trial = neural_data[:n_trials*trial_time, :]
        print(f"Warning: Truncated {time_total - n_trials*trial_time} time points")
    else:
        spike_data_trial = neural_data
    # First reshape to (n_trials, 600, n_neurons)
    reshaped = spike_data_trial.reshape(n_trials, trial_time, n_neurons)

    # Then transpose to get (600, n_neurons, n_trials)
    spike_data_trials = np.transpose(reshaped, (1, 2, 0))

    # Now take the mean across trials (axis=2, not axis=3 as Python is 0-indexed)
    trial_average = np.mean(spike_data_trials, axis=2)

    return trial_average
sim_trial_average = make_trials(sim_observations,trial_time)
fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)


# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(sim_trial_average), 
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
plt.show()

#%% Transition matrix
plt.figure(figsize=(10, 8))
transition_matrix = slds.transitions.transition_matrix
# Create the heatmap with custom styling
ax = sns.heatmap(
    transition_matrix, 
    annot=True,           # Show values in cells
    fmt=".2f",            # Format to 2 decimal places
    cmap="PuRd",        # Blue-green colormap
    cbar=True,            # Show color scale
    square=True,          # Make cells square
    linewidths=0.5,       # Add thin grid lines
    linecolor="white",    # White grid lines
    xticklabels=["State 1", "State 2", "State 3"],
    yticklabels=["State 1", "State 2", "State 3"]
)

# Add descriptive labels and title
plt.title("Transition Matrix from rSLDS", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("To State", fontsize=14, labelpad=10)
plt.ylabel("From State", fontsize=14, labelpad=10)

# Adjust text color based on background for better readability
for text in ax.texts:
    value = float(text.get_text())
    if value > 0.5:  # Dark background needs white text
        text.set_color('white')

# Add colorbar label
cbar = ax.collections[0].colorbar
cbar.set_label("Transition Probability", fontsize=12, labelpad=10)

plt.tight_layout()
filename = "Transition_state.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
#%%
sim_states, sim_latent, sim_observations = slds.sample(1000,with_noise=False)

sim_latent_smooth = np.zeros_like(sim_latent, dtype='int32')
for n in range(sim_latent.shape[1]):
    sim_latent_smooth[:, n] = gaussian_filter1d(sim_latent[:, n]*1000, 10)

sim_latent_smooth = sim_latent_smooth/1000   
from ssm.plots import plot_dynamics_2d
import matplotlib.pyplot as plt
import numpy as np

# Iterate over all discrete states
num_states = slds.K  # Number of discrete states
lim = abs(sim_latent_smooth).max(axis=0) + 4  # Define limits based on latent dynamics
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])
import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)

for k in range(num_states):
    # Extract dynamics for state k
    dynamics_matrix = np.squeeze(slds.dynamics.As[k, :, :])
    bias_vector = np.squeeze(slds.dynamics.bs[k, :])

    # Create a new figure for each state's flow field
    plt.figure(figsize=(6, 6))
    plot_dynamics_2d(dynamics_matrix, bias_vector, mins=mins, maxs=maxs,color=colors[k])
    
    # Overlay the latent dynamics trajectory
    #plt.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1)
    
    # Add title and labels
    plt.title(f"Flow Field for State {k+1}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    filename = f"flowfield2_{k}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()

# %% Simulate observations based on Poisson emissions
# Create publication-quality figure with better dimensions
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)


# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(binned_spike_data), 
                   aspect='auto', 
                   cmap='plasma',  # Scientific colormap that's colorblind-friendly
                   interpolation='none',)  # Preserves exact values
axs[0].set_title("Binned Spike Counts", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Neurons", fontsize=12)

# Add gridlines to help identify neuron positions
axs[0].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Custom colorbar with proper positioning
cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
cbar1.set_label("Spike Count", fontsize=12)
cbar1.ax.tick_params(labelsize=10)

# Plot smoothed spikes with different colormap for visual distinction
im2 = axs[1].imshow(np.transpose(sim_observations), 
                   aspect='auto', 
                   cmap='plasma',  # Different colormap to distinguish from raw data
                   interpolation='none')
axs[1].set_title("Simulated Spike Counts", fontsize=14, fontweight='bold')
axs[1].set_xlabel("Time Bins", fontsize=12)
axs[1].set_ylabel("Neurons", fontsize=12)

# Add gridlines
axs[1].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Custom colorbar for smooth data
cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
cbar2.set_label("Simulated Spike Count", fontsize=12)
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

plt.show()

#%% Calculating firing rates of binned data
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Parameters
sigma = 40  # Smoothing parameter (in bins)
bin_size_ms = 10  # Bin size in milliseconds

def compute_binned_spike_data(spike_counts, sigma, bin_size_ms):
    """
    Compute continuous firing rates from binned spike data using Gaussian smoothing.
    """
    # Check input dimensions
    if len(spike_counts.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {spike_counts.shape}")
    
    n_timepoints,n_neurons = spike_counts.shape
    binned_spike_data = np.zeros_like(spike_counts, dtype=float)
    
    # Diagnostic: Check if neurons have different patterns
    neuron_sums = np.sum(spike_counts, axis=1)
    if np.allclose(neuron_sums, neuron_sums[0], rtol=1e-5):
        print("WARNING: All neurons have nearly identical spike counts!")
    
    # Convert to Hz (spikes/second) by scaling
    scale_factor = 1000 / bin_size_ms  # Convert to Hz
    
    # Apply Gaussian smoothing to each neuron individually
    for i in range(n_neurons):
        # Explicitly use array indexing
        current_neuron = spike_counts[:, i].copy()  # Get copy of this neuron's data
        # Scale first, then smooth
        binned_spike_data[:, i] = gaussian_filter1d(current_neuron * scale_factor, sigma=sigma)
    
    return binned_spike_data

# Compute firing rates - make sure binned_spike_data is shape (neurons, time)
binned_spike_data = compute_binned_spike_data(binned_spike_data, sigma, 10)

#%% State occupation data analysis
# Make a circle plot of proportion of states
import numpy as np
import matplotlib.pyplot as plt

# Example data: replace this with your actual array
arr = rslds_states

# Count occurrences of each unique value
unique, counts = np.unique(arr, return_counts=True)
proportions = counts / counts.sum()

# Prepare labels with percentages
labels = [f"{u} ({p*100:.0f}%)" for u, p in zip(unique, proportions)]

# Colors (customize as needed)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create a donut plot
fig, ax = plt.subplots(figsize=(3,3))
wedges, texts, autotexts = ax.pie(
    proportions, 
    labels=('State 1','State 2','State 3'), 
    colors=colors[:len(unique)], 
    autopct='%1.1f%%',
    startangle=90, 
    wedgeprops=dict(width=0.4)

)

ax.set_title('Mouse 2')
plt.tight_layout()
filename = "Mouse3_state.pdf"
#plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
plt.show()

# %%
