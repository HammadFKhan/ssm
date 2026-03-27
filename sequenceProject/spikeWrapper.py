"""
This script performs decomposes analysis on matlab spike data.

Loads in spiking and behavior data. It then transforms the data into a 3d array for further analysis. 


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

mat_file_path = r"Y:\Hammad\Ephys\SeqProject\WarpedSpikes\M1\Day6_M1_warpedSpks.mat"


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
# %% Plot out the data
import getFigures as sqFig

sqFig.plot_spikes_pca(binned_spike_data,pca,latent_dynamics)

# %%
