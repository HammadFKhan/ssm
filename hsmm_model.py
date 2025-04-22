"""
Hidden Semi-Markov Model (HSMM)
===============================
"""
# %%
import autograd.numpy as np
import autograd.numpy.random as npr
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from ssm.util import rle, find_permutation
from ssm import HSMM
from ssm import HMM
from ssm import SLDS

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm  # Import tqdm for loading bar
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score

npr.seed(0)

#%% Simulate Neural Data
   

def simulate_spike_trains(n_states=3, n_neurons=50, repetitions=5, 
                         base_firing_rates=None, transition_matrix=None,
                         duration_params=None):
    """
    Simulate spike trains with variable durations and custom transitions.

    Parameters:
    - n_states: int, number of states (e.g., Rest, Movement, Post-Rest)
    - n_neurons: int, number of neurons in the population
    - state_durations: list of int, duration of each state in time bins
    - repetitions: int, number of times the state sequence is repeated
    - base_firing_rates: list or None, base firing rates for each state (defaults to [5,30,15])
    - transition_steps: Number of time bins for rate transitions between states


    Returns:
    - spike_trains: np.ndarray, simulated spike counts (time bins x neurons)
    - true_states: np.ndarray, true state labels for each time bin
    """
    # Default parameters
    if base_firing_rates is None:
        base_firing_rates = [5, 30, 15]
    
    # Define custom transition matrix (without self-transitions)
    if transition_matrix is None:
        transition_matrix = np.array([
            [0.0, 0.7, 0.3],  # From state 0 to states 1,2
            [0.6, 0.0, 0.4],  # From state 1 to states 0,2
            [0.8, 0.2, 0.0]   # From state 2 to states 0,1
        ])
    
    # Define variable duration distributions for each state
    if duration_params is None:
        duration_params = [
            {'distribution': 'poisson', 'mean': 100},      # State 0: ~50 time bins
            {'distribution': 'negbinom', 'r': 5, 'p': 0.1},  # State 1: ~45 time bins
            {'distribution': 'poisson', 'mean': 30}       # State 2: ~30 time bins
        ]
    
    # Define firing rates with variability
    state_rates = np.array([
        [base_firing_rates[i]] * n_neurons for i in range(n_states)
    ]) + np.random.rand(n_states, n_neurons) * 10  # Add variability
    
    # Generate sequence of states with variable durations
    states_sequence = []
    current_state = 0  # Start with state 0
    
    for _ in range(repetitions):
        for i in range(n_states):
            # Sample next state using transition probabilities
            if i > 0:
                next_probs = transition_matrix[current_state].copy()
                next_probs = next_probs / next_probs.sum()  # Normalize
                current_state = np.random.choice(n_states, p=next_probs)
            
            # Sample duration for this state
            params = duration_params[current_state]
            if params['distribution'] == 'poisson':
                duration = max(1, np.random.poisson(params['mean']))
            elif params['distribution'] == 'negbinom':
                duration = max(1, np.random.negative_binomial(params['r'], params['p']))
            
            # Add state to sequence
            states_sequence.extend([current_state] * duration)
    
    # Convert to arrays
    true_states = np.array(states_sequence)
    
    # Generate spike counts
    spike_trains = np.zeros((len(true_states), n_neurons))
    for t, state in enumerate(true_states):
        spike_trains[t] = np.random.poisson(state_rates[state])
    
    return spike_trains, true_states, state_rates

def simulate_enhanced_spike_trains(n_states=3, n_neurons=50, repetitions=5, 
                                   base_firing_rates=None, transition_matrix=None,
                                   duration_params=None):
    """
    Simulate realistic spike trains with advanced neurophysiological features.
    """
    # Default parameters
    if base_firing_rates is None:
        base_firing_rates = [5, 30, 15]
    
    if transition_matrix is None:
        transition_matrix = np.array([
            [0.0, 0.7, 0.3],  # From state 0 to states 1,2
            [0.6, 0.0, 0.4],  # From state 1 to states 0,2
            [0.8, 0.2, 0.0]   # From state 2 to states 0,1
        ])
    
    if duration_params is None:
        duration_params = [
            {'distribution': 'poisson', 'mean': 100},      
            {'distribution': 'negbinom', 'r': 5, 'p': 0.1},  
            {'distribution': 'poisson', 'mean': 30}       
        ]
    
    # FEATURE 1: Heterogeneous neural population
    n_state_selective = int(n_neurons * 0.6)
    n_mixed = int(n_neurons * 0.2)
    n_noise = n_neurons - n_state_selective - n_mixed
    
    # FEATURE 2: Create neuron IDs by type
    neuron_types = np.zeros(n_neurons, dtype=int)
    neuron_types[n_state_selective:n_state_selective+n_mixed] = 1
    neuron_types[n_state_selective+n_mixed:] = 2
    
    # FEATURE 3: Create realistic base firing rates with greater variance
    state_rates = np.zeros((n_states, n_neurons))
    
    # State-selective neurons
    for n in range(n_state_selective):
        preferred_state = n % n_states
        for s in range(n_states):
            if s == preferred_state:
                state_rates[s, n] = base_firing_rates[s] * (1 + 0.5 * np.random.randn())
            else:
                state_rates[s, n] = base_firing_rates[0] * 0.2 * (1 + 0.5 * np.random.randn())
    
    # Mixed-selectivity neurons
    for n in range(n_state_selective, n_state_selective + n_mixed):
        responsive_states = np.random.choice(n_states, size=2, replace=False)
        for s in range(n_states):
            if s in responsive_states:
                state_rates[s, n] = base_firing_rates[s] * (0.7 + 0.3 * np.random.rand())
            else:
                state_rates[s, n] = base_firing_rates[0] * 0.3 * (1 + 0.5 * np.random.rand())
    
    # Noise neurons - FIXED to show more activity
    for n in range(n_state_selective + n_mixed, n_neurons):
        baseline = base_firing_rates[1] * (0.3 + 0.3 * np.random.rand())  # Higher baseline using state 1 rate
        for s in range(n_states):
            state_rates[s, n] = baseline * (0.9 + 0.3 * np.random.rand())
    
    # NEW FEATURE: Add overlapping PC structure to neurons
    for n in range(n_neurons // 3):
        # These neurons will have similar activity across states
        for s in range(n_states):
            state_rates[s, n] = base_firing_rates[1] * (0.8 + 0.4 * np.random.rand())
    
    # NEW FEATURE: Add nonlinear relationships between neurons
    for n1 in range(n_neurons // 4):
        for n2 in range(n_neurons // 4, n_neurons // 2):
            for s in range(n_states):
                # Create multiplicative interactions
                interaction = 0.2 * state_rates[s, n1] * state_rates[s, n2] / base_firing_rates[s]
                state_rates[s, n1] += interaction * (0.5 + 0.5 * np.random.rand())
    
    # Generate sequence of states with variable durations
    states_sequence = []
    durations_sequence = []
    current_state = 0
    
    # FEATURE 4: Track actual state value
    true_state_values = []
    
    # NEW FEATURE: Create sub-states
    sub_states_sequence = []
    
    for _ in tqdm(range(repetitions), desc="Repetitions Progress", leave=True):
        for i in range(n_states):
            # Sample next state using transition probabilities
            if i > 0:
                next_probs = transition_matrix[current_state].copy()
                next_probs = next_probs / next_probs.sum()
                current_state = np.random.choice(n_states, p=next_probs)
            
            # Sample duration for this state
            params = duration_params[current_state]
            if params['distribution'] == 'poisson':
                duration = max(5, np.random.poisson(params['mean']))
            elif params['distribution'] == 'negbinom':
                duration = max(5, np.random.negative_binomial(params['r'], params['p']))
            
            # Add state to sequence
            states_sequence.extend([current_state] * duration)
            durations_sequence.append(duration)
            
            # Add continuous state value
            true_state_values.extend([current_state] * duration)
            
            # Add sub-state structure
            sub_state = 0
            for _ in range(duration):
                if np.random.rand() < 0.05:  # 5% chance to switch sub-state
                    sub_state = 1 - sub_state
                sub_states_sequence.append(sub_state)
    
    # Convert to arrays
    true_states = np.array(states_sequence)
    sub_states = np.array(sub_states_sequence)
    
    # FEATURE 6: Generate correlated spike counts
    spike_trains = np.zeros((len(true_states), n_neurons))
    
    # Create correlation matrix for neurons
    neuron_correlations = np.eye(n_neurons)
    
    # Add correlations between neurons of the same type
    for n1 in range(n_neurons):
        for n2 in range(n1+1, n_neurons):
            if neuron_types[n1] == neuron_types[n2]:
                if neuron_types[n1] == 0:
                    corr = 0.2 + 0.1 * np.random.rand()
                elif neuron_types[n1] == 1:
                    corr = 0.1 + 0.1 * np.random.rand()
                else:
                    corr = 0.3 + 0.2 * np.random.rand()
                
                neuron_correlations[n1, n2] = corr
                neuron_correlations[n2, n1] = corr
    
    # FEATURE 7: Generate spikes with realistic transitions
    transition_time = 15 
    prev_state = true_states[0]
    for t in tqdm(range(len(true_states)), desc="Spike Train Generation", leave=True):
        current_state = true_states[t]
        current_sub_state = sub_states[t]
        
        # Detect state transitions
        if t > 0 and current_state != prev_state:
            transition_start = t
            
            # Generate transition period
            for trans_t in range(t, min(t + transition_time, len(true_states))):
                if true_states[trans_t] == current_state:
                    alpha = (trans_t - transition_start) / transition_time
                    blended_rates = state_rates[prev_state] * (1-alpha) + state_rates[current_state] * alpha
                    
                    # Generate spikes with correlated noise
                    mean_vec = blended_rates
                    noise = np.random.multivariate_normal(np.zeros(n_neurons), 
                                                         neuron_correlations * np.outer(blended_rates, blended_rates) * 0.1)
                    
                    lambda_t = np.maximum(0.1, mean_vec + noise)
                    spike_trains[trans_t] = np.random.poisson(lambda_t)
                    
        # Normal state (not in transition)
        else:
            # Apply sub-state modulation
            if current_sub_state == 1:
                # Alter firing pattern for sub-state
                mask = np.random.choice([0, 1], n_neurons, p=[0.7, 0.3])
                sub_state_rates = state_rates[current_state].copy()
                sub_state_rates[mask == 1] *= 1.5  # Increase rates for some neurons
            else:
                sub_state_rates = state_rates[current_state].copy()
            
            # Add state-dependent noise
            if current_state == 0:
                noise_scale = 0.15
            else:
                noise_scale = 0.1
                
            # Generate correlated noise
            noise = np.random.multivariate_normal(np.zeros(n_neurons), 
                                                neuron_correlations * np.outer(sub_state_rates, 
                                                                            sub_state_rates) * noise_scale)
            # Add this after calculating all rate parameters:
            max_realistic_rate = 50  # Maximum biologically plausible firing rate

            # Apply rate limiting to base state rates
            for s in range(n_states):
                state_rates[s] = np.minimum(state_rates[s], max_realistic_rate)

            # And modify the spike generation section:
            lambda_t = np.minimum(max_realistic_rate, np.maximum(0.1, sub_state_rates + noise))
            spike_trains[t] = np.random.poisson(lambda_t)
        
        prev_state = current_state
    
    # Return all relevant data
    return spike_trains, true_states, sub_states, state_rates, true_state_values

# Example usage with your parameters
n_states = 3
n_neurons = 100
repetitions = 25  # Generate 5 repetitions of the state sequence
base_firing_rates = [5, 60, 15, 30, 25]  # Your specified rates


duration_params = [
    {'distribution': 'poisson', 'mean': 500},      # State 0: ~50 time bins
    {'distribution': 'negbinom', 'r': 10, 'p': 0.1},  # State 1: ~45 time bins
    {'distribution': 'poisson', 'mean': 100},       # State 2: ~30 time bins
    {'distribution': 'negbinom', 'r': 5, 'p': 0.019},  # State 1: ~45 time bins
    {'distribution': 'negbinom', 'r': 3, 'p': 0.008}  # State 1: ~45 time bins
]


transition_matrix = np.array([
    [0.0, 0.4, 0.3, 0.2, 0.1],  # From state 0 to states 1,2
    [0.6, 0.0, 0.2, 0.1, 0.1],  # From state 1 to states 0,2
    [0.4, 0.2, 0.0, 0.3, 0.1],   # From state 2 to states 0,1
    [0.1, 0.2, 0.5, 0.0, 0.2],  # From state 2 to states 0,1
    [0.3, 0.2, 0.2, 0.3, 0.0]   # From state 2 to states 0,1
])

'''
spike_counts, true_states,true_state_rates = simulate_spike_trains(
    n_states, 
    n_neurons, 
    repetitions,
    base_firing_rates,
    transition_matrix,
    duration_params
)
'''

'''
duration_params = [
    {'distribution': 'poisson', 'mean': 1000},   #~400ms   
    {'distribution': 'negbinom', 'r': 5, 'p': 0.0146},  #~250ms  
    {'distribution': 'poisson', 'mean': 400}       #~250ms 
]
'''
base_firing_rates = [5, 15, 7, 20, 24]

spike_counts, true_states, sub_states, state_rates, true_state_values = simulate_enhanced_spike_trains(n_states, 
                                                                                                       n_neurons, 
                                                                                                       repetitions
                                                                                                       )

print(f"Generated data shape: {spike_counts.shape}")
print(f"Total duration: {len(true_states)} time bins")

# 2. Bin the data into 20ms bins
n_timebins = len(true_states)
bin_size = 3
n_bins = n_timebins // bin_size
binned_spike_counts = np.zeros((n_bins, n_neurons))
binned_true_states = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    binned_spike_counts[i] = spike_counts[i * bin_size:(i + 1) * bin_size].sum(axis=0)
true_states = true_states[:n_bins * bin_size:bin_size]
assert len(binned_spike_counts) == len(binned_true_states)
# 3. Apply Gaussian smoothing
smoothed_spike_counts = np.zeros_like(binned_spike_counts)
# Assuming spike_counts has shape (n_timebins, n_neurons)
# Choose sigma (kernel width) based on your time bin size
sigma = 3 # Start with 3 time bins width, adjust based on your data

# Apply smoothing to each neuron's time series
smoothed_spikes = np.zeros_like(binned_spike_counts, dtype=float)
for n in range(binned_spike_counts.shape[1]):
    smoothed_spikes[:, n] = gaussian_filter1d(binned_spike_counts[:, n], sigma)
    
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Choose number of components for latent space (2-3 is good for visualization)
n_components = 10

# Fit PCA to the spike count data
scaler = StandardScaler()
smoothed_spikes_standardized = scaler.fit_transform(smoothed_spikes)

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
from hammad.Fig_SimSpike import plot_spikes_pca, plot_state_transitions

transition_matrix = np.array([
        [0.0, 0.7, 0.3],  # From state 0 to states 1,2
        [0.6, 0.0, 0.4],  # From state 1 to states 0,2
        [0.8, 0.2, 0.0]   # From state 2 to states 0,1
    ])
plot_spikes_pca(binned_spike_counts,pca,latent_dynamics)

plot_state_transitions(transition_matrix, latent_dynamics, true_states,min_max_bounds=(-20, 20), plot_limit=1000)

#%%

# 3. HSMM & HMM initialization
num_states = 3
obs_dim = n_components  # Get 3 from PCA components
hsmm = HSMM(num_states, obs_dim,observations="gaussian")
hmm = HMM(num_states, obs_dim, observations="gaussian")


# 4. Fit to latent dynamics
hsmm_em_lls = hsmm.fit(latent_dynamics, method="em")
hmm_em_lls = hmm.fit(latent_dynamics, method="em")

# 5. Analyze results
hsmm.permute(find_permutation(true_states, hsmm.most_likely_states(latent_dynamics)))
states = hsmm.most_likely_states(latent_dynamics)

hmm.permute(find_permutation(true_states, hmm.most_likely_states(latent_dynamics)))
Hmm_states = hmm.most_likely_states(latent_dynamics)

hsmm_ll = hsmm.log_likelihood(latent_dynamics)
hmm_ll = hmm.log_likelihood(latent_dynamics)

# Create the model and initialize its parameters
slds = SLDS(n_neurons, num_states, 2, emissions="poisson_orthog", transitions="recurrent",emission_kwargs=dict(link="softplus"))
binned_spike_counts = binned_spike_counts.astype(np.int32)
assert binned_spike_counts.dtype == int
slds.initialize(binned_spike_counts)
# Fit the model using Laplace-EM with a structured variational posterior
q_lem_elbos, q_lem = slds.fit(binned_spike_counts, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               num_iters=100,initialize=False)

# Get the posterior mean of the continuous states
q_lem_x = q_lem.mean_continuous_states[0]

# Find the permutation that matches the true and inferred states
slds.permute(find_permutation(true_states, slds.most_likely_states(q_lem_x, binned_spike_counts)))
rslds_states = slds.most_likely_states(q_lem_x, binned_spike_counts)

# Smooth the data under the variational posterior
q_lem_y = slds.smooth(q_lem_x, binned_spike_counts)


#%% 
# Call the function with your data
def evaluate_models_by_pcs(latent_dynamics, true_states, max_pcs=None):
    # Determine max number of PCs to use
    if max_pcs is None:
        max_pcs = latent_dynamics.shape[1]
    
    # Initialize results storage
    results = {
        'num_pcs': [],
        'hsmm_lls': [],
        'hmm_lls': [],
        'rslds_lls':[],
        'hsmm_states': [],
        'hmm_states': [],
        'rslds_states':[],
        'hsmm_ari': [],
        'hmm_ari': [],
        'rslds_ari': []
    }
    
    # Loop through different numbers of PCs
    for n_pcs in range(2, max_pcs + 1):
        print(f"Evaluating models with {n_pcs} PCs...")
        
        # Use the first n_pcs principal components
        input_dynamics = latent_dynamics[:, :n_pcs]
        
        # Initialize models with correct dimensions
        num_states = 3
        hsmm = HSMM(num_states, n_pcs, observations="gaussian")
        hmm = HMM(num_states, n_pcs, observations="gaussian")
        
        # Fit models
        hsmm_em_lls = hsmm.fit(input_dynamics, method="em")
        hmm_em_lls = hmm.fit(input_dynamics, method="em")
        
        # Get predicted states using correct dimensions
        hsmm_states = hsmm.most_likely_states(input_dynamics)
        hmm_states = hmm.most_likely_states(input_dynamics)
        
        # Calculate log-likelihoods
        hsmm_ll = hsmm.log_likelihood(input_dynamics)
        hmm_ll = hmm.log_likelihood(input_dynamics)

        # Create the model and initialize its parameters
        n_neurons = binned_spike_counts.shape[1]
        slds = SLDS(n_pcs, num_states, 1, emissions="gaussian_orthog", transitions="recurrent")

        slds.initialize(input_dynamics)
        # Fit the model using Laplace-EM with a structured variational posterior
        q_lem_elbos, q_lem = slds.fit(input_dynamics, method="laplace_em",
                                    variational_posterior="structured_meanfield",
                                    num_iters=100,initialize=False)

        # Get the posterior mean of the continuous states
        q_lem_x = q_lem.mean_continuous_states[0]

        # Find the permutation that matches the true and inferred states
        slds.permute(find_permutation(true_states, slds.most_likely_states(q_lem_x, input_dynamics)))
        rslds_states = slds.most_likely_states(q_lem_x, input_dynamics)

        # Smooth the data under the variational posterior
        q_lem_y = slds.smooth(q_lem_x, input_dynamics)

        
        # Calculate Adjusted Rand Index for state alignment with ground truth
        hsmm_ari = adjusted_rand_score(true_states, hsmm_states)
        hmm_ari = adjusted_rand_score(true_states, hmm_states)
        rslds_ari = adjusted_rand_score(true_states,rslds_states)
        
        # Store results
        results['num_pcs'].append(n_pcs)
        results['hsmm_lls'].append(hsmm_ll)
        results['hmm_lls'].append(hmm_ll)
        results['rslds_lls'].append(np.max(q_lem_elbos))
        results['hsmm_states'].append(hsmm_states)
        results['hmm_states'].append(hmm_states)
        results['rslds_states'].append(rslds_states)
        results['hsmm_ari'].append(hsmm_ari)
        results['hmm_ari'].append(hmm_ari)
        results['rslds_ari'].append(rslds_ari)
        
        print(f"  HSMM Log-Likelihood: {hsmm_ll:.2f}, ARI: {hsmm_ari:.2f}")
        print(f"  HMM Log-Likelihood: {hmm_ll:.2f}, ARI: {hmm_ari:.2f}")
    
    # Save results to MATLAB file
    savemat('model_performance_by_pcs.mat', results)
    print("Results saved to 'model_rslds_performance_by_pcs.mat'")
    
    return results
# Call the function with your data
def evaluate_models_by_states(latent_dynamics, true_states, max_states=None):
    """
    Evaluate HSMM, HMM, and rSLDS model performance with varying numbers of discrete states.
    
    Args:
        latent_dynamics: PCA-reduced neural data (n_timepoints, n_pcs)
        true_states: Ground truth state labels
        max_states: Maximum number of states to evaluate (default: 5)
    
    Returns:
        Dictionary of performance metrics for each model and state count
    """
    # Determine max number of states to evaluate
    if max_states is None:
        max_states = 5
    
    # Get number of PCs in the data
    n_pcs = latent_dynamics.shape[1]
    
    # Initialize results storage
    results = {
        'num_states': [],
        'hsmm_lls': [],
        'hmm_lls': [],
        'rslds_lls':[],
        'hsmm_states': [],
        'hmm_states': [],
        'rslds_states':[],
        'hsmm_ari': [],
        'hmm_ari': [],
        'rslds_ari': []
    }
    
    # Loop through different numbers of states
    for n_states in range(2, max_states + 1):  # Start at 2 states minimum
        print(f"Evaluating models with {n_states} discrete states...")
        
        # Initialize models with correct dimensions
        hsmm = HSMM(n_states, n_pcs, observations="gaussian")
        hmm = HMM(n_states, n_pcs, observations="gaussian")
        
        # Fit HMM and HSMM models
        hsmm_em_lls = hsmm.fit(latent_dynamics, method="em")
        hmm_em_lls = hmm.fit(latent_dynamics, method="em")
        
        # Get predicted states
        hsmm_states = hsmm.most_likely_states(latent_dynamics)
        hmm_states = hmm.most_likely_states(latent_dynamics)
        
        # Calculate log-likelihoods
        hsmm_ll = hsmm.log_likelihood(latent_dynamics)
        hmm_ll = hmm.log_likelihood(latent_dynamics)

        # Create and fit the rSLDS model
        latent_dim = 2  # Use 3D latent space or match n_pcs if smaller
        slds = SLDS(n_pcs, n_states, latent_dim, 
                    emissions="poisson_orthog", 
                    transitions="recurrent",
                    emission_kwargs=dict(link="softplus"))

        slds.initialize(latent_dynamics,
                        verbose=1,
                        num_init_iters=50,
                        discrete_state_init_method="random",
                        num_init_restarts=1)
        
        q_lem_elbos, q_lem = slds.fit(latent_dynamics, 
                                    method="laplace_em",
                                    variational_posterior="structured_meanfield",
                                    num_iters=50, 
                                    initialize=False)

        # Get continuous states and most likely discrete states
        q_lem_x = q_lem.mean_continuous_states[0]
        rslds_states = slds.most_likely_states(q_lem_x, latent_dynamics)

        # Smooth the data under the variational posterior
        q_lem_y = slds.smooth(q_lem_x, latent_dynamics)
        
        # Calculate Adjusted Rand Index
        hsmm_ari = adjusted_rand_score(true_states, hsmm_states)
        hmm_ari = adjusted_rand_score(true_states, hmm_states)
        rslds_ari = adjusted_rand_score(true_states, rslds_states)
        
        # Store results
        results['num_states'].append(n_states)
        results['hsmm_lls'].append(hsmm_ll)
        results['hmm_lls'].append(hmm_ll)
        results['rslds_lls'].append(float(np.max(q_lem_elbos)))
        results['hsmm_states'].append(hsmm_states)
        results['hmm_states'].append(hmm_states)
        results['rslds_states'].append(rslds_states)
        results['hsmm_ari'].append(hsmm_ari)
        results['hmm_ari'].append(hmm_ari)
        results['rslds_ari'].append(rslds_ari)
        
        # Print performance metrics
        print(f"  HSMM Log-Likelihood: {hsmm_ll:.2f}, ARI: {hsmm_ari:.2f}")
        print(f"  HMM Log-Likelihood: {hmm_ll:.2f}, ARI: {hmm_ari:.2f}")
        print(f"  rSLDS ELBO: {np.max(q_lem_elbos):.2f}, ARI: {rslds_ari:.2f}")
    
    # Save results to MATLAB file
    savemat('model_performance_by_states.mat', results)
    print("Results saved to 'model_performance_by_states.mat'")
    return results

#results = evaluate_models_by_pcs(latent_dynamics, true_states)
assert binned_spike_counts.dtype == int
results= evaluate_models_by_states(binned_spike_counts[0:4000,:], true_states, max_pcs=n_components)
results_state = evaluate_models_by_states(binned_spike_counts[0:4000,:], true_states, max_states=6)
#%% Plot log-likelihoods
# Plot combined figure with all comparisons
from hammad.Fig_SimSpike import plot_combined_model_comparison,plot_hyperparam_model_performance

fig = plot_combined_model_comparison(
    results_pc=results,          # Your PC-based results
    results_state=results_state, # Your state-based results
    n_pcs=n_components,
    figsize=(15, 12),
    palette='colorblind'
)

plot_hyperparam_model_performance(latent_dynamics, true_states, states, Hmm_states, rslds_states,
                                      results_state, results, pca, hsmm_em_lls, hmm_em_lls, q_lem_elbos,
                                      n_states=None)
# %%
# Create figure
fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

# Plot true states with better colormap and colorbar
im1 = axs[0].imshow(true_states[None, :], aspect="auto", cmap="viridis", vmin=0, vmax=n_states)
axs[0].set_ylabel("True $z$", fontsize=12)
axs[0].yaxis.set_ticks([])  # Remove y-axis ticks
cbar1 = fig.colorbar(im1, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)
cbar1.set_label("State", fontsize=10)
cbar1.ax.tick_params(labelsize=10)

# Plot inferred states with better colormap and colorbar
im2 = axs[1].imshow(states[None, :], aspect="auto", cmap="viridis", vmin=0, vmax=n_states)
axs[1].set_ylabel("HSMM Inferred $z$", fontsize=12)
axs[1].yaxis.set_ticks([])  # Remove y-axis ticks
cbar2 = fig.colorbar(im2, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)
cbar2.set_label("State", fontsize=10)
cbar2.ax.tick_params(labelsize=10)

# Plot inferred states with better colormap and colorbar
im2 = axs[2].imshow(Hmm_states[None, :], aspect="auto", cmap="viridis", vmin=0, vmax=n_states)
axs[2].set_ylabel("HMM Inferred $z$", fontsize=12)
axs[2].yaxis.set_ticks([])  # Remove y-axis ticks
cbar2 = fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)
cbar2.set_label("State", fontsize=10)
cbar2.ax.tick_params(labelsize=10)

# Plot inferred states with better colormap and colorbar
im2 = axs[3].imshow(rslds_states[None, :], aspect="auto", cmap="viridis", vmin=0, vmax=n_states)
axs[3].set_ylabel("RSLDS Inferred $z$", fontsize=12)
axs[3].yaxis.set_ticks([])  # Remove y-axis ticks
cbar2 = fig.colorbar(im2, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)
cbar2.set_label("State", fontsize=10)
cbar2.ax.tick_params(labelsize=10)


# Add shared x-axis label
axs[3].set_xlabel("Time Bins", fontsize=12)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
# %%
# Plot log likelihoods (fit model is typically better)
plt.figure()
plt.plot(hsmm_em_lls, ls='-', label="HSMM (EM)")
plt.plot(hmm_em_lls, ls='-', label="HMM (EM)")
plt.legend(loc="lower right")
plt.plot(q_lem_elbos[1:], label="Laplace-EM")

plt.legend(loc="lower right")


print(hsmm_ll)
print(hmm_ll)

#%% Plot the true and inferred duration distributions
n_states = 3
true_state_seq, durations = rle(rslds_states)
inf_states, inf_durations = rle(rslds_states)
max_duration = max(np.max(durations), np.max(inf_durations))
dd = np.arange(max_duration, step=1)

plt.figure(figsize=(3 * n_states, 9))
for k in range(n_states):
    # Plot the durations of the true states
    plt.subplot(3, n_states, k+1)
    plt.hist(durations[true_state_seq == k] - 1, dd, density=True)
    if k == n_states - 1:
        plt.legend(loc="lower right")
    plt.title("State {} (N={})".format(k+1, np.sum(rslds_states == k)))

    # Plot the durations of the inferred states
    plt.subplot(3, n_states, n_states+k+1)
    plt.hist(inf_durations[inf_states == k] - 1, dd, density=True)
    plt.plot(dd, nbinom.pmf(dd, hsmm.transitions.rs[k], 1 - hsmm.transitions.ps[k]),
             '-r', lw=2, label='hsmm inf.')
    if k == n_states - 1:
        plt.legend(loc="lower right")
    plt.title("State {} (N={})".format(k+1, np.sum(inf_states == k)))

    # Plot the durations of the inferred states
    plt.subplot(3, n_states, n_states+k+1)
    plt.hist(inf_durations[inf_states == k] - 1, dd, density=True)
    plt.plot(dd, nbinom.pmf(dd, slds.transitions.rs[k], 1 - slds.transitions.ps[k]),
             '-r', lw=2, label='rslds inf.')
    if k == n_states - 1:
        plt.legend(loc="lower right")
    plt.title("State {} (N={})".format(k+1, np.sum(inf_states == k)))


plt.tight_layout()

plt.show()

#%%
# Plot PCA trajectories with state colors
plt.figure(figsize=(4, 4))


for state in range(n_states):
    mask = (true_states == state)
    plt.scatter(latent_dynamics[mask, 0], latent_dynamics[mask, 1], 
                label=f'State {state}', alpha=0.4, s=1)

# Add connected line to show temporal trajectory
plt.plot(latent_dynamics[:, 0], latent_dynamics[:, 1], 'k-', alpha=0.2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('True States in PCA Space')
plt.legend()

plt.figure(figsize=(4, 4))


for state in range(n_states):
    mask = (states == state)
    plt.scatter(latent_dynamics[mask, 0], latent_dynamics[mask, 1], 
                label=f'State {state}', alpha=0.4, s=1)

# Add connected line to show temporal trajectory
plt.plot(latent_dynamics[:, 0], latent_dynamics[:, 1], 'k-', alpha=0.2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('HSMM States in PCA Space')
plt.legend()

# Plot out the smoothed inference data from slds where both the contineous and discrete are mapped

plt.figure(figsize=(4, 4))

for state in range(n_states):
    mask = (rslds_states == state)
    plt.scatter(q_lem_y[mask, 0], q_lem_y[mask, 1], 
                label=f'State {state}', alpha=0.4, s=1)

# Add connected line to show temporal trajectory
plt.plot(q_lem_y[:, 0], q_lem_y[:, 1], 'k-', alpha=0.2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('SLDS States in Latent Space')
plt.legend()
# %%
import seaborn as sns

# Assuming you already have your true_transition_matrix and inferred_transition_matrix
transition_matrix = np.array([
            [0.0, 0.7, 0.3],  # From state 0 to states 1,2
            [0.6, 0.0, 0.4],  # From state 1 to states 0,2
            [0.8, 0.2, 0.0]   # From state 2 to states 0,1
        ])
inferred_transition_matrix = hsmm.transitions.Ps
inferred_transition_matrix_slds = np.abs(slds.transitions.log_Ps)
# Create a side-by-side plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

# Plot the true transition matrix
sns.heatmap(transition_matrix, 
            ax=ax1, 
            cmap='YlGnBu',  # Blue-green colormap
            annot=True,     # Show values in cells
            fmt='.2f',      # Format as 2 decimal places
            square=True,    # Make cells square
            vmin=0, vmax=1, # Set consistent color scale
            cbar_kws={'label': 'Transition probability'})

ax1.set_title('True Transition Matrix', fontsize=14)
ax1.set_xlabel('To State')
ax1.set_ylabel('From State')

# Plot the inferred transition matrix
sns.heatmap(inferred_transition_matrix, 
            ax=ax2, 
            cmap='YlGnBu',
            annot=True,
            fmt='.2f',
            square=True,
            vmin=0, vmax=1,
            cbar_kws={'label': 'Transition probability'})

ax2.set_title('Inferred Transition Matrix', fontsize=14)
ax2.set_xlabel('To State')
ax2.set_ylabel('From State')

# Plot the inferred transition matrix
sns.heatmap(inferred_transition_matrix_slds, 
            ax=ax3, 
            cmap='YlGnBu',
            annot=True,
            fmt='.2f',
            square=True,
            vmin=0, vmax=1,
            cbar_kws={'label': 'Transition probability'})

ax3.set_title('Inferred Transition Matrix', fontsize=14)
ax3.set_xlabel('To State')
ax3.set_ylabel('From State')


# If you have state names, you can use them as tick labels

plt.tight_layout()
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

ax3 = plt.subplot(133)
plot_trajectory(rslds_states, q_lem_y, ax=ax3)
plt.title("Inferred, Laplace-EM")
plt.tight_layout()
# %%
def plot_rslds_dynamics_with_trajectories(
    model, latent_states, data, q_posterior=None, colors=['r', 'g', 'b', 'c', 'm', 'y', 'k'],
    ax=None, figsize=(6, 6), trajectory_alpha=0.7,
    plot_state_colors=True):
    """
    Plot rSLDS dynamics with neural trajectories overlaid.
    
    Args:
        model: The rSLDS model
        latent_states: Latent state trajectories (T x D) or list of trajectories
        data: Original data used for inference
        q_posterior: Variational posterior from Laplace-EM (if applicable)
        colors: Colors for different discrete states
        trajectory_alpha: Transparency of trajectory lines
        plot_state_colors: Whether to color trajectories by their discrete state
    """
    K = model.K  # Number of discrete states
    assert model.D == 2, "This function only works for 2D latent states"
    
    # Adjust grid limits to match trajectory scale
    x_min, x_max = latent_states[:, 0].min(), latent_states[:, 0].max()
    y_min, y_max = latent_states[:, 1].min(), latent_states[:, 1].max()
    
    # Add margins to the grid limits
    margin = 1  # 50% margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    xlim = (x_min - margin * x_range, x_max + margin * x_range)
    ylim = (y_min - margin * y_range, y_max + margin * y_range)
    
    # Create a grid for the vector field
    nxpts = 30
    nypts = 30
    x = np.linspace(xlim[0], xlim[1], nxpts)
    y = np.linspace(ylim[0], ylim[1], nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    
    # Get the probability of each state at each xy location
    log_Ps = model.transitions.log_transition_matrices(
        xy, np.zeros((nxpts * nypts, 0)), np.ones_like(xy, dtype=bool), None)
    z = np.argmax(log_Ps[:, 0, :], axis=-1)  # Assign each grid point to a state
    z = np.concatenate([[z[0]], z]) 
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # Plot the vector field for each discrete state
    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum() > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=0.8)

    # Add the trajectories
    if isinstance(latent_states, list):
        # Multiple trajectories
        for traj_idx, traj in enumerate(latent_states):
            if traj.shape[1] > 2:
                traj = traj[:, :2]  # Use only first two dimensions
            
            if plot_state_colors:
                if q_posterior is not None:
                    traj_z = model.most_likely_states(traj, data[traj_idx])
                else:
                    traj_z = model.most_likely_states(data[traj_idx])
                
                for k in range(K):
                    state_mask = traj_z == k
                    if not np.any(state_mask):
                        continue
                    
                    state_changes = np.diff(np.concatenate([[0], state_mask.astype(int), [0]]))
                    starts = np.where(state_changes == 1)[0]
                    ends = np.where(state_changes == -1)[0]
                    
                    for start, end in zip(starts, ends):
                        ax.plot(traj[start:end, 0], traj[start:end, 1],
                                '-', color=colors[k % len(colors)],
                                alpha=trajectory_alpha, linewidth=1.5)
            else:
                ax.plot(traj[:, 0], traj[:, 1], '-', color='k',
                        alpha=trajectory_alpha, linewidth=1.5)
    else:
        # Single trajectory
        if latent_states.shape[1] > 2:
            latent_states = latent_states[:, :2]
        
        if plot_state_colors:
            if q_posterior is not None:
                states = model.most_likely_states(latent_states, data)
            else:
                states = model.most_likely_states(data)
            
            for k in range(K):
                state_mask = states == k
                if not np.any(state_mask):
                    continue
                
                state_changes = np.diff(np.concatenate([[0], state_mask.astype(int), [0]]))
                starts = np.where(state_changes == 1)[0]
                ends = np.where(state_changes == -1)[0]
                
                for start, end in zip(starts, ends):
                    ax.plot(latent_states[start:end, 0], latent_states[start:end, 1],
                            '-', color=colors[k % len(colors)],
                            alpha=trajectory_alpha, linewidth=1.5)
        else:
            ax.plot(latent_states[:, 0], latent_states[:, 1], '-', color='k',
                    alpha=trajectory_alpha, linewidth=1.5)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()


# Plot with both continuous states and data
plot_rslds_dynamics_with_trajectories(
    slds,                  # model
    q_lem_y,               # latent states from posterior
    latent_dynamics,        # original data
    q_posterior=q_lem,      # posterior object
)
plt.title("rSLDS Dynamics with Neural Trajectories")
plt.show()
#%% Save data
fname = 'modelPerforamanceDataFinalWorking'

# Import required libraries
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score
import os

# Calculate ARI scores correctly
hsmm_ari = adjusted_rand_score(true_states, states)
hmm_ari = adjusted_rand_score(true_states, Hmm_states)
rslds_ari = adjusted_rand_score(true_states, rslds_states)  # Fixed: using rslds_states instead of xhat_lem

# Create dictionaries with correct metrics
groundTruthData = {
    'binned_spike_counts': binned_spike_counts,
    'smoothed_spikes': smoothed_spikes,
    'true_states': true_states,
    'latent_dynamics': latent_dynamics
}

hsmmData = {
    'latent_states': states,
    'transition_matrix': hsmm.transitions.transition_matrix,
    'log_likelihood': hsmm_ll,
    'em_learning_curve': hsmm_em_lls,
    'adjusted_score': hsmm_ari  # Correctly using hsmm_ari
}

hmmData = {
    'latent_states': Hmm_states,
    'transition_matrix': hmm.transitions.transition_matrix,
    'log_likelihood': hmm_ll,
    'em_learning_curve': hmm_em_lls,
    'adjusted_score': hmm_ari  # Fixed: using hmm_ari instead of rslds_ari
}

rsldsData = {
    'latent_states': q_lem_y,
    'discrete_states': rslds_states,
    'transition_matrix': slds.transitions.transition_matrix,
    'log_likelihood': q_lem_elbos,  # Added this
    'neural_trajectories': latent_dynamics,
    'q_elbos': q_lem_elbos,
    'adjusted_score': rslds_ari  # Fixed: using rslds_ari instead of hsmm_ari
}

# Combine all dictionaries into a single dictionary with nested structure
allData = {
    'groundTruth': groundTruthData,
    'hsmm': hsmmData,
    'hmm': hmmData,
    'rslds': rsldsData
}

# Save to a single .mat file
fileName = f"{fname}.mat"
savemat(fileName, allData)

print(f"Model data saved: {os.path.exists(fileName)}")
print(f"HSMM ARI: {hsmm_ari:.3f}, HMM ARI: {hmm_ari:.3f}, rSLDS ARI: {rslds_ari:.3f}")
# %%
from ssm.plots import plot_most_likely_dynamics

sim_states, sim_latent, sim_observations = slds.sample(4000, with_noise=0)
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(sim_latent).max(axis=0) + 1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(sim_latent[:,0], sim_latent[:,1], '-k', lw=1)

plt.title("Simulated Dynamics")
# %% Plot out inferrred latent dynamics
plt.figure(figsize=(6, 6))

# Create two subplots (2 rows, 1 column)
ax1 = plt.subplot(211)  # First subplot (top)
ax2 = plt.subplot(212)  # Second subplot (bottom)
inferred_latent_dynamics =  np.zeros_like(q_lem.mean_continuous_states[0], dtype='int32')
for n in range(q_lem.mean_continuous_states[0].shape[1]):
    inferred_latent_dynamics[:, n] = gaussian_filter1d(q_lem.mean_continuous_states[0][:, n]*100, 10)
inferred_latent_dynamics = inferred_latent_dynamics/100
# Plot data on each subplot
ax1.plot(inferred_latent_dynamics, '-k', lw=1)
ax2.plot(latent_dynamics[:, :2], '-r', lw=1)


# Set x-axis limits for both subplots (optional)
ax1.set_xlim(0, 1000)
ax2.set_xlim(0, 1000)

plt.tight_layout()  # Improves spacing
plt.show()
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
q_lem_scaled = inferred_latent_dynamics[:1200]
lim = abs(q_lem_scaled).max(axis=0)+1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(q_lem_scaled[:,0], q_lem_scaled[:,1], '-k', lw=1)

plt.title("Most Likely Dynamics, Laplace-EM")
# %%
# Create publication-quality figure with better dimensions
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)


# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(binned_spike_counts), 
                   aspect='auto', 
                   cmap='viridis',  # Scientific colormap that's colorblind-friendly
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
im2 = axs[1].imshow(np.transpose(q_lem_y), 
                   aspect='auto', 
                   cmap='plasma',  # Different colormap to distinguish from raw data
                   interpolation='none')
axs[1].set_title("Inferred Spike Counts", fontsize=14, fontweight='bold')
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
    ax.set_yticks(np.arange(0, np.transpose(binned_spike_counts).shape[0], 5))
    ax.set_yticklabels([f"{i}" for i in range(0, np.transpose(binned_spike_counts).shape[0], 5)])

# Adjust spacing between subplots
plt.tight_layout()

# Optional: Add a super title
fig.suptitle("Neural Population Activity", fontsize=16, y=1.02)

plt.show()
#%%
sim_states, sim_latent, sim_observations = slds.sample(20000,with_noise=False)

sim_latent_smooth = np.zeros_like(sim_latent, dtype='int32')
for n in range(sim_latent.shape[1]):
    sim_latent_smooth[:, n] = gaussian_filter1d(sim_latent[:, n], 10)
    
from ssm.plots import plot_dynamics_2d
import matplotlib.pyplot as plt
import numpy as np

# Iterate over all discrete states
num_states = slds.K  # Number of discrete states
lim = abs(sim_latent_smooth).max(axis=0) + 4  # Define limits based on latent dynamics
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])

for k in range(num_states):
    # Extract dynamics for state k
    dynamics_matrix = np.squeeze(slds.dynamics.As[k, :, :])
    bias_vector = np.squeeze(slds.dynamics.bs[k, :])

    # Create a new figure for each state's flow field
    plt.figure(figsize=(6, 6))
    plot_dynamics_2d(dynamics_matrix, bias_vector, mins=mins, maxs=maxs)
    
    # Overlay the latent dynamics trajectory
    plt.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1)
    
    # Add title and labels
    plt.title(f"Flow Field for State {k}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    
    plt.show()

# %%
from ssm.plots import plot_dynamics_2d
import matplotlib.pyplot as plt
import numpy as np

# Define limits based on your latent dynamics
lim = abs(sim_latent_smooth).max(axis=0) + 4
mins = (-lim[0], -lim[1])
maxs = (lim[0], lim[1])

# Create a single figure for all flow fields
plt.figure(figsize=(8, 8))

# Iterate over all discrete states
for k in range(slds.K):  # slds.K is the number of discrete states
    # Extract dynamics for state k
    dynamics_matrix = np.squeeze(slds.dynamics.As[k, :, :])
    bias_vector = np.squeeze(slds.dynamics.bs[k, :])
    
    # Plot the vector field for state k
    plot_dynamics_2d(dynamics_matrix, bias_vector, mins=mins, maxs=maxs, npts =20,color=f"C{k}")

# Overlay the latent dynamics trajectory
plt.plot(sim_latent_smooth[:, 0], sim_latent_smooth[:, 1], '-k', lw=1, label="Latent Trajectory")

# Add labels and legend
plt.title("Combined Flow Fields for All States")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.legend()
plt.show()
# %%
# Create publication-quality figure with better dimensions
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)


# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(binned_spike_counts), 
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
axs[1].set_title("Inferred Spike Counts", fontsize=14, fontweight='bold')
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
    ax.set_yticks(np.arange(0, np.transpose(binned_spike_counts).shape[0], 5))
    ax.set_yticklabels([f"{i}" for i in range(0, np.transpose(binned_spike_counts).shape[0], 5)])

# Adjust spacing between subplots
plt.tight_layout()

# Optional: Add a super title
fig.suptitle("Neural Population Activity", fontsize=16, y=1.02)

plt.show()
# %%
