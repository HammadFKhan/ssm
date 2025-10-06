"""
10/3/2025 Hammad Khan
New version of the batch processing for rslds model. We needed to do an overall
to incorporate trial types other than the hit sequence trials. Originally, we would parse
out the warped structure we got from the affine methods, but it is a trivial step when we want
to fit the linear dynamical system model. So instead we just build the function that we want in
matlab where the trials types can be transferred into a 2d array of concatenate trials in 
chronological order. 

For example if we are building a model that contains hit and miss trials, we concatenate the neural 
data together with correct ordering and then run the rslds model for the whole neural space. 

"""
#%% Load in the warped structure that we have. Format it into the way we want to run the rslds pipeline through it. 
import autograd.numpy as np
import autograd.numpy.random as npr


from ssm import SLDS

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm  # Import tqdm for loading bar
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score
npr.seed(0)
import h5py
import os
import glob
#%% Here we point to a folder and then run the model training to save to the folder. 

def train_rslds_from_mat(mat_file_path, save_path):
    with h5py.File(mat_file_path, 'r') as f:
        # Access the 'warpedSpks' dataset (the MATLAB struct)
        warpedSpks = f['rsldsSpikes']
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
        struct = f['rsldsSpikes' ]
        print("Fields in the struct:")
        print(list(struct.keys()))  # Should list all fields, including 'warpedSpikes'

  
        # Typically, the struct array contains references
        # Get the first struct in the array (adjust the index for your case)
        struct_ref = 'rsldsSpikes' # or [0] if 1D
        
        # Dereference to get the struct group
        struct = f[struct_ref]
        
        print("Fields in the struct:")
        print(list(struct.keys()))  # Should list all fields, including 'warpedSpikes'
        
        # Access the 'warpedSpikes' field
        task_spikes_refs = struct['taskSpikes'][:]
        
        # 3. Handle the dimensions of the cell array.
        # A 1xN MATLAB cell array will be a (1, N) NumPy array of references.
        # You need to flatten it to iterate over the individual cell references.
        task_spikes_refs_flat = task_spikes_refs.flatten()

        # 4. Loop through the references to dereference and access the data.
        # This will give you a list of the contents of each cell.
        rslds_spk = []
        for ref in task_spikes_refs_flat:
            # Dereference the object reference and read the data
            cell_content = f[ref][:]
            rslds_spk.append(cell_content)

        # 5. Process the extracted cell data.
        # For a list of 3D arrays, this list will contain each array.
        if rslds_spk:
            print("\n--- Extracted Data ---")
            print("Number of cells:", len(rslds_spk))
            print("Shape of the first cell's array:", rslds_spk[0].shape)
            
            # Dereference again to get the actual 3D array data
            #print(f"Shape of warpedSpikes array: {warpedSpikes_data.shape}")
            # Now, warpedSpikes_data is a NumPy array with your 3D data

    # Load in warp data and wrangle it into what we need it to be
    #warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\DLS\Day9_DLS_warpedSpks_rslds.npy")
    #warp_spk = np.load(r"D:\SequenceProject\WarpedSpikes\M1\Day6_M1_warpedSpks_rslds.npy")
    spk = rslds_spk[3]
    spk_ref = np.transpose(spk, (2,1,0))
    [n_trials,n_time,n_neurons] = spk_ref.shape
    print("\n--- Extracted Data ---")
    print("Number of trials:", n_trials)
    print("Time for each trial:", n_time)
    print("Number of neurons:", n_neurons)
    #%%
    spike_data = spk_ref.reshape(n_time*n_trials,n_neurons)
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
    # Print ELBO of the model
    print('Elbos model performance:',q_lem_elbos[-1:])


    #% Save data
    # Import required libraries
    from scipy.io import savemat
    import os

    # Calculate ARI scores correctly
    #rslds_ari = adjusted_rand_score(true_states, rslds_states)  # Fixed: using rslds_states instead of xhat_lem

    # Create dictionaries with correct metrics
    groundTruthData = {
        'binned_spike_data': binned_spike_data,
        'latent_dynamics': latent_dynamics,
        'bin_size': bin_size_ms,
        'originalFilepath': mat_file_path
    }

    rsldsData = {
        'q_lem': q_lem,
        'latent_states': q_lem_x,
        'discrete_states': rslds_states,
        'inferred_spikes':q_lem_y,
        'transition_matrix': slds.transitions.transition_matrix,
        'q_elbos': q_lem_elbos,
        'rslds_model': slds
    }


    # Save to a single .mat file
    fileName  = save_path+'_rsldsModel.npz'
    print('Saving file to',fileName)
    # Save multiple arrays and objects to an .npz file
    np.savez(fileName, rsldsData=rsldsData,groundTruthData=groundTruthData)
    print(f"Model data saved: {os.path.exists(fileName)}")

    # %% Here we want to retrain the model to extract the 
    # latent contineous states and their corresponding eigen values 
    # 3. rSlds initialization
    num_states = 3
    obs_dim = binned_spike_data.shape[1]  # Get 3 from PCA components
    latent_dim = 5
    # Create the model and initialize its parameters

    slds_expand = SLDS(obs_dim, num_states, latent_dim, emissions="poisson_orthog", transitions="recurrent",emission_kwargs=dict(link="softplus"))
    binned_spike_data = binned_spike_data.astype(np.int32)
    assert binned_spike_data.dtype == int
    slds_expand.initialize(binned_spike_data,verbose=1)
    # Fit the model using Laplace-EM with a structured variational posterior
    q_lem_elbos,q_lem= slds_expand.fit(binned_spike_data, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                num_iters=50,initialize=False)

    # Plot ELBO of the model
    print('Elbos model performance:',q_lem_elbos[-1:])

    # Get the posterior mean of the continuous states
    q_lem_x = q_lem.mean_continuous_states[0]

    # Find the permutation that matches the true and inferred states
    rslds_states = slds_expand.most_likely_states(q_lem_x, binned_spike_data)

    # Smooth the data under the variational posterior
    q_lem_y = slds_expand.smooth(q_lem_x, binned_spike_data)
    #% Save data
    # Import required libraries
    from scipy.io import savemat
    import os

    # Calculate ARI scores correctly
    #rslds_ari = adjusted_rand_score(true_states, rslds_states)  # Fixed: using rslds_states instead of xhat_lem

    # Create dictionaries with correct metrics

    rsldsData = {
        'q_lem': q_lem,
        'latent_states': q_lem_x,
        'discrete_states': rslds_states,
        'inferred_spikes':q_lem_y,
        'transition_matrix': slds.transitions.transition_matrix,
        'q_elbos': q_lem_elbos,
        'rslds_model': slds_expand
    }

    # Save to a single .mat file
    fileName  = save_path+'_rsldsModel_expand.npz'
    print('Saving file to',fileName)

    # Save multiple arrays and objects to an .npz file
    np.savez(fileName, rsldsData=rsldsData)
    print(f"Model data saved: {os.path.exists(fileName)}")

def batch_train_rslds(directory_path):
    # Get all .mat files in the directory
    mat_files = glob.glob(os.path.join(directory_path, '*.mat'))

    # Create a folder for saving models
    models_dir = os.path.join(directory_path, 'rslds_models')
    os.makedirs(models_dir, exist_ok=True)

    # Iterate through all mat files and train
    for mat_file in mat_files:
        # Building the output model file path
        base_name = os.path.splitext(os.path.basename(mat_file))[0]
        save_path = os.path.join(models_dir, base_name)
        
        # Train the model from mat file
        train_rslds_from_mat(mat_file, save_path)
        
    print(f'Trained and saved {len(mat_files)} models to {models_dir}')


batch_train_rslds(r'Y:\Hammad\Ephys\SeqProject\ForceField\rsldsSpks_sessions')