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
repetitions = 50  # Generate 5 repetitions of the state sequence
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
bin_size = 1
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
sigma = 25 # Start with 3 time bins width, adjust based on your data

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
latent_dynamics_smoothed = np.zeros_like(latent_dynamics, dtype=float)

'''
for n in range(latent_dynamics.shape[1]):
    latent_dynamics_smoothed[:, n] = gaussian_filter1d(latent_dynamics[:, n], 1)
latent_dynamics = latent_dynamics_smoothed  
'''   
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
im2 = axs[1].imshow(np.transpose(smoothed_spikes), 
                   aspect='auto', 
                   cmap='plasma',  # Different colormap to distinguish from raw data
                   interpolation='none')
axs[1].set_title("Smoothed Spike Counts", fontsize=14, fontweight='bold')
axs[1].set_xlabel("Time Bins", fontsize=12)
axs[1].set_ylabel("Neurons", fontsize=12)

# Add gridlines
axs[1].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Custom colorbar for smooth data
cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
cbar2.set_label("Smoothed Spike Count", fontsize=12)
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

plt.figure()
plt.plot(latent_dynamics)
plt.show()

plt.figure(figsize=(4, 4))
# Create a colormap from true states
colors = plt.cm.viridis(true_states/max(true_states))

plt.scatter(latent_dynamics[:, 0], latent_dynamics[:, 1], 
            c=colors, s=30, alpha=0.7)
plt.colorbar(label='State')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Neural Population Activity in Latent Space')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter3D(latent_dynamics[:, 0], latent_dynamics[:, 1], latent_dynamics[:, 2], c=colors, s=30,)
cbar = fig.colorbar(scatter)
cbar.set_label('State')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('Neural Trajectories in Latent Space')
plt.show()

#%% Show Plots by PC loading weights
# Create publication-quality figure with better dimensions and PC weight plot
fig, axs = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 1]})

# Plot sorted binned spike counts
im1 = axs[0].imshow(np.transpose(binned_spike_counts[:, sort_idx]), 
                   aspect='auto', 
                   cmap='viridis',
                   interpolation='none')
axs[0].set_title("Binned Spike Counts", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Neurons (sorted by PC weights)", fontsize=12)

# Create a separate axis for PC weights
# Plot PC weights as horizontal bar charts
for i in range(3):
    color = f'C{i}'
    axs[1].barh(np.arange(len(sort_idx)), 
                pc_weights[i, sort_idx], 
                height=0.8, 
                alpha=0.6, 
                color=color,
                label=f'PC{i+1}')

axs[1].set_title("PC Weights", fontsize=14, fontweight='bold')
axs[1].set_ylim(axs[0].get_ylim())  # Match y-axis limits
axs[1].set_yticks([])  # Remove y-ticks for cleaner look
axs[1].legend(loc='upper right')
axs[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)  # Zero line

# Custom colorbar with proper positioning
cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
cbar1.set_label("Spike Count", fontsize=12)

# Add proper tick formatting
axs[0].tick_params(axis='both', which='major', labelsize=10)

# Create similar plot for smoothed spikes (in a separate figure)
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 1]})


im2 = axs2[0].imshow(np.transpose(smoothed_spikes[:, sort_idx]), 
                    aspect='auto', 
                    cmap='plasma',
                    interpolation='none')
axs2[0].set_title("Smoothed Spike Counts", fontsize=14, fontweight='bold')
axs2[0].set_xlabel("Time Bins", fontsize=12)
axs2[0].set_ylabel("Neurons (sorted by PC weights)", fontsize=12)

# Plot PC weights again in the second figure
for i in range(3):
    color = f'C{i}'
    axs2[1].barh(np.arange(len(sort_idx)), 
                pc_weights[i, sort_idx], 
                height=0.8, 
                alpha=0.6, 
                color=color,
                label=f'PC{i+1}')

axs2[1].set_title("PC Weights", fontsize=14, fontweight='bold')
axs2[1].set_ylim(axs2[0].get_ylim())
axs2[1].set_yticks([])
axs2[1].legend(loc='upper right')
axs2[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Custom colorbar
cbar2 = fig2.colorbar(im2, ax=axs2[0], fraction=0.046, pad=0.04)
cbar2.set_label("Smoothed Spike Count", fontsize=12)

plt.tight_layout()
plt.show()
#%%

# 3. HSMM & HMM initialization
num_states = 3
obs_dim = n_neurons  # Get 3 from PCA components

# Create the model and initialize its parameters
slds = SLDS(n_neurons, num_states, 2, emissions="poisson_orthog", transitions="recurrent",emission_kwargs=dict(link="softplus"))

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
# %% Plot ELBO of the model
plt.figure()
plt.plot(q_lem_elbos[1:], label="Laplace-EM")

plt.legend(loc="lower right")
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
plt.savefig('transition_matrix_comparison.png', dpi=300, bbox_inches='tight')
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

#%% Save data
fname = 'rsldsPerforamanceData'

# Import required libraries
from scipy.io import savemat
from sklearn.metrics import adjusted_rand_score
import os

# Calculate ARI scores correctly
rslds_ari = adjusted_rand_score(true_states, rslds_states)  # Fixed: using rslds_states instead of xhat_lem

# Create dictionaries with correct metrics
groundTruthData = {
    'binned_spike_counts': binned_spike_counts,
    'smoothed_spikes': smoothed_spikes,
    'true_states': true_states,
    'latent_dynamics': latent_dynamics
}

rsldsData = {
    'latent_states': q_lem_y,
    'discrete_states': rslds_states,
    'transition_matrix': slds.transitions.transition_matrix,
    'log_likelihood': q_lem_elbos,  # Added this
    'q_elbos': q_lem_elbos,
    'adjusted_score': rslds_ari  # Fixed: using rslds_ari instead of hsmm_ari
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
print(f"rSLDS ARI: {rslds_ari:.3f}")

# %% Plot out inferrred latent dynamics
plt.figure(figsize=(6, 6))

# Create two subplots (2 rows, 1 column)
ax1 = plt.subplot(211)  # First subplot (top)
ax2 = plt.subplot(212)  # Second subplot (bottom)
inferred_latent_dynamics =  np.zeros_like(q_lem.mean_continuous_states[0], dtype='int32')
for n in range(q_lem.mean_continuous_states[0].shape[1]):
    inferred_latent_dynamics[:, n] = gaussian_filter1d(q_lem.mean_continuous_states[0][:, n], 20)

# Plot data on each subplot
ax1.plot(inferred_latent_dynamics, '-k', lw=1)
ax2.plot(latent_dynamics[:, :2], '-r', lw=1)


# Set x-axis limits for both subplots (optional)
ax1.set_xlim(0, 1000)
ax2.set_xlim(0, 1000)

plt.tight_layout()  # Improves spacing
plt.show()

# %%
from ssm.plots import plot_most_likely_dynamics
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
q_lem_scaled = inferred_latent_dynamics
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

sim_states, sim_latent, sim_observations = slds.sample(1000, with_noise=1)
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(sim_latent).max(axis=0) + 1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(sim_latent[:,0], sim_latent[:,1], '-k', lw=1)

plt.title("Simulated Dynamics")


#%%
sim_states, sim_latent, sim_observations = slds.sample(9000,with_noise=False)

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

# %% Simulate observations based on Poisson emissions
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
    ax.set_yticks(np.arange(0, np.transpose(binned_spike_counts).shape[0], 5))
    ax.set_yticklabels([f"{i}" for i in range(0, np.transpose(binned_spike_counts).shape[0], 5)])

# Adjust spacing between subplots
plt.tight_layout()

# Optional: Add a super title
fig.suptitle("Neural Population Activity", fontsize=16, y=1.02)

plt.show()