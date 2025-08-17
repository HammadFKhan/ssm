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
#mat_file_path = r"D:\SQLever\Ephys\WarpedSpikes\DLS\Day9_DLS_warpedSpks.mat"

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

#%% Show Plots by PC loading weights

from hammad.Fig_SimSpike import plot_spikes_pca, plot_state_transitions

plot_spikes_pca(binned_spike_data,pca,latent_dynamics,filename=f"{fSave}latent_weights.pdf")
#%%

# 3. rSlds initialization
num_states = 3
obs_dim = binned_spike_data.shape[1]  # Get 3 from PCA components
latent_dim = 2
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
from ssm.plots import plot_most_likely_dynamics
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
q_lem_scaled = inferred_latent_dynamics[:150,:]*2
lim = abs(q_lem_scaled).max(axis=0)+1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(q_lem_scaled[:,0], q_lem_scaled[:,1], '-k', lw=1)

plt.title("Most Likely Dynamics, Laplace-EM")
# %%
sqFig.plot_inferred_population(binned_spike_data,q_lem_y,filename = f"{fSave}Inferred_Spike_state.pdf")

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
leverTrace = np.reshape(all_traces_array, all_traces_array.shape[0]*all_traces_array.shape[1])
leverTrace = leverTrace[1::bin_size_ms]
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

#%% Save data
fname = 'rsldsPerforamanceM1Day6'

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

# Save multiple arrays and objects to an .npz file
np.savez(f"{fname}.npz", rsldsData,groundTruthData)
print(f"Model data saved: {os.path.exists(fileName)}")
# %% Plot each linear system
from ssm.plots import plot_dynamics_2d
# Iterate over all discrete states
num_states = slds.K  # Number of discrete states
lim = abs(latent_dynamics).max(axis=0) + 4  # Define limits based on latent dynamics
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])
import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)

for k in range(num_states):
    # Extract dynamics for state k
    dynamics_matrix = np.squeeze(slds.dynamics.As[k, :, :])
    dynamics_matrix = dynamics_matrix[:2,:2]
    bias_vector = np.squeeze(slds.dynamics.bs[k, :])
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
    filename = f"flowfield2_{k}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()
# %%
def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=30, nypts=30,
    alpha=0.8, ax=None, figsize=(3, 3)):
    import seaborn as sns
    color_names = ["windows blue", "red", "amber", "faded green"]
    colors = sns.xkcd_palette(color_names)
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    log_Ps = model.transitions.log_transition_matrices(
        xy, np.zeros((nxpts * nypts, 0)), np.ones_like(xy, dtype=bool), None)
    z = np.argmax(log_Ps[:, 0, :], axis=-1)
    z = np.concatenate([[z[0]], z])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax