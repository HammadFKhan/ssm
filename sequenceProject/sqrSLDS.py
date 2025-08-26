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


#mat_file_path = r"D:\SQLever\Ephys\WarpedSpikes\M1\Day6_M1_warpedSpks.mat"
mat_file_path = r"D:\SQLever\Ephys\WarpedSpikes\DLS\Day10_DLS_warpedSpks.mat"

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
from matplotlib.colors import ListedColormap
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
axs[1].plot(inferred_latent_dynamics[:,0], '-k', lw=1)
axs[1].plot(inferred_latent_dynamics[:,1], '-r', lw=1)

# Plot data on each subplot
axs[2].plot(leverTrace, '-k', lw=1)
totalPullTimes = np.array([pull1,pull2,pull3])//bin_size_ms
totalPullTimes = totalPullTimes.squeeze()
ymin = np.max(leverTrace)
for n in range(280):
     axs[2].vlines(totalPullTimes[:,n]+(250*n), ymin, ymin+0.04, colors='k', linestyles='-')
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
leverTrace = gaussian_filter1d(leverTrace*100, 5)/100
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
    'q_elbos': q_lem_elbos,
    'rslds_model': slds
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
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
q_lem_scaled = inferred_latent_dynamics*10
latent_average_scaled = latent_average
lim = abs(latent_average_scaled).max(axis=0)+1
totalPullTimes = np.array([0,np.mean(pull1),np.mean(pull2),np.mean(pull3),np.mean(pull3)+500])//bin_size_ms
totalPullTimes = totalPullTimes.astype(int)
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
#plt.plot(q_lem_scaled, q_lem_scaled,'-k', lw=1,alpha = 0.3)
plt.plot(latent_average_scaled[:,0], latent_average_scaled[:,1],'-k', lw=2,alpha = 0.5)
plt.scatter(latent_average_scaled[totalPullTimes,0],latent_average_scaled[totalPullTimes,1],s=24,c='green')
plt.title("Sequence Dynamics")
filename = f"{fSave}likely_state.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
print(f"Figure saved as {filename}")
plt.show()
# %% Here we want to retrain the model to extract the 
# latent contineous states and their corresponding eigen values 
# 3. rSlds initialization
num_states = 3
obs_dim = binned_spike_data.shape[1]  # Get 3 from PCA components
latent_dim = 8
# Create the model and initialize its parameters

slds_expand = SLDS(obs_dim, num_states, latent_dim, emissions="poisson_orthog", transitions="recurrent",emission_kwargs=dict(link="softplus"))
binned_spike_data = binned_spike_data.astype(np.int32)
assert binned_spike_data.dtype == int
slds_expand.initialize(binned_spike_data,verbose=1)
# Fit the model using Laplace-EM with a structured variational posterior
q_lem_elbos_expand,q_lem_expanded = slds_expand.fit(binned_spike_data, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               num_iters=50,initialize=False)

# Plot ELBO of the model
plt.figure()
plt.plot(q_lem_elbos_expand[1:], label="Laplace-EM")

plt.legend(loc="lower right")

#% Save data
fname = 'rsldsPerforamanceDLSDay10_expand'

# Import required libraries
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score
import os

# Calculate ARI scores correctly
#rslds_ari = adjusted_rand_score(true_states, rslds_states)  # Fixed: using rslds_states instead of xhat_lem


rsldsData = {
    'q_elbos': q_lem_elbos_expand,
    'rslds_model': slds_expand
}



# Save multiple arrays and objects to an .npz file
fileName = f"{fname}.npz"
np.savez(f"{fname}.npz", rsldsData)
print(f"Model data saved: {os.path.exists(fileName)}")
# %%
from ssm.plots import plot_dynamics_2d

num_states = slds_expand.K  # Number of discrete states
lim = abs(latent_dynamics).max(axis=0) + 4  # Define limits based on latent dynamics
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])
import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
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
eigenVal = eig(slds_expand.dynamics.A)
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
# %%
dynamics_matrix = np.array([(0.2, 0),(0,1)])
print(dynamics_matrix)
bias_vector = np.array([0,0])
mins = (-50, -50)
maxs = (100, 40)
# Create a new figure for each state's flow field
plt.figure(figsize=(6, 6))
plot_dynamics_2d(dynamics_matrix, bias_vector,
                 mins=(-40,-40),
                 maxs=(100,40))

# Overlay the latent dynamics trajectory
#plt.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1)

# Add title and labels
plt.title(f"Flow Field for State")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()

from scipy.linalg import eig
eigenVal = eig(dynamics_matrix)
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


plt.show()