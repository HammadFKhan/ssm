
#%%
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


#%%
def simulate_spike_trains(n_states, n_neurons, state_durations, repetitions, base_firing_rates=None,transition_steps=None):
    """
    Simulate spike trains for a neural population with repetitive states.

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
    # Set default firing rates
    if base_firing_rates is None:
        base_firing_rates = [5, 30, 15]
    if transition_steps is None:
        transition_steps = 1
    
    # Calculate total time bins including transitions
    states_per_rep = len(state_durations)
    total_timebins = (sum(state_durations) + transition_steps * (states_per_rep - 1)) * repetitions
    
    # Initialize arrays
    spike_trains = np.zeros((total_timebins, n_neurons))
    true_states = np.zeros(total_timebins, dtype=int)
    state_rates = np.array([base_firing_rates] * n_neurons).T * (1 + np.random.rand(n_states, n_neurons) * 0.2)

    current_time = 0
    for _ in range(repetitions):
        prev_rates = None
        
        for state_idx, (duration, target_rates) in enumerate(zip(state_durations, state_rates)):
            # Add transition from previous state
            if prev_rates is not None and transition_steps > 0:
                for t in range(transition_steps):
                    alpha = t / transition_steps
                    current_rates = prev_rates * (1 - alpha) + target_rates * alpha
                    spike_trains[current_time] = np.random.poisson(current_rates)
                    true_states[current_time] = true_states[current_time - 1]  # Maintain previous state label
                    current_time += 1
            
            # Add main state duration
            spike_trains[current_time:current_time+duration] = np.random.poisson(target_rates, (duration, n_neurons))
            true_states[current_time:current_time+duration] = state_idx
            current_time += duration
            prev_rates = target_rates

    # Trim unused preallocated space
    return spike_trains[:current_time], true_states[:current_time]


# Example usage with your parameters
n_states = 3
n_neurons = 50
state_durations = [30, 25,30]  # Rest, Movement, Post-Rest durations in time bins
repetitions = 10  # Generate 5 repetitions of the state sequence
base_firing_rates = [5, 30,15]  # Your specified rates
transition_steps=2  # 20-bin transitions between states
spike_counts, true_states = simulate_spike_trains(
    n_states, 
    n_neurons, 
    state_durations, 
    repetitions,
    base_firing_rates,
    transition_steps
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
sigma = 10 # Start with 3 time bins width, adjust based on your data

# Apply smoothing to each neuron's time series
smoothed_spikes = np.zeros_like(binned_spike_counts, dtype=float)
for n in range(binned_spike_counts.shape[1]):
    smoothed_spikes[:, n] = gaussian_filter1d(binned_spike_counts[:, n], sigma)
    
from sklearn.decomposition import PCA

# Choose number of components for latent space (2-3 is good for visualization)
n_components = 2

# Fit PCA to the spike count data
pca = PCA(n_components=n_components)
latent_trajectories = pca.fit_transform(smoothed_spikes)

# Use the projected data for HMM observations
observations = latent_trajectories

# %%
# Create publication-quality figure with better dimensions
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Plot binned spike counts
im1 = axs[0].imshow(np.transpose(binned_spike_counts), 
                   aspect='auto', 
                   cmap='viridis',  # Scientific colormap that's colorblind-friendly
                   interpolation='none')  # Preserves exact values
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
plt.plot(observations)
plt.show()

plt.figure(figsize=(4, 4))
# Create a colormap from true states
colors = plt.cm.viridis(true_states/max(true_states))

plt.scatter(latent_trajectories[:, 0], latent_trajectories[:, 1], 
            c=colors, s=30, alpha=0.7)
plt.colorbar(label='State')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Neural Population Activity in Latent Space')

#%%
# Configure HMM with Gaussian emissions (approximates summed Poisson)
model = hmm.GaussianHMM(
    n_components=n_states,
    covariance_type="diag",
    n_iter=20
)
model.fit(observations)

# Infer states
predicted_states = model.predict(observations)
#%%
# Plot PCA trajectories with state colors
plt.figure(figsize=(4, 4))


for state in range(n_states):
    mask = (predicted_states == state)
    plt.scatter(latent_trajectories[mask, 0], latent_trajectories[mask, 1], 
                label=f'State {state}', alpha=0.4, s=20)

# Add connected line to show temporal trajectory
plt.plot(latent_trajectories[:, 0], latent_trajectories[:, 1], 'k-', alpha=0.2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('HMM States in PCA Space')
plt.legend()

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import permutations

def find_best_permutation(true_states, pred_states):
    n_states = max(max(true_states), max(pred_states)) + 1
    perms = list(permutations(range(n_states)))
    
    best_acc = 0
    best_perm = None
    for perm in perms:
        remapped_pred = np.array([perm[s] for s in pred_states])
        acc = np.mean(remapped_pred == true_states)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
            
    return best_perm, best_acc

[best_perm,best_acc] = find_best_permutation(true_states, predicted_states)

# Create remapped predicted states using the best permutation
remapped_predicted = np.array([best_perm[s] for s in predicted_states])

# Plot with aligned state labels
# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

# Plot true states
axs[0].plot(true_states, label="True States", alpha=0.7, linewidth=3, color='blue')
axs[0].set_ylabel("State")
axs[0].set_title("True States")
axs[0].legend()

# Plot aligned predicted states
axs[1].plot(remapped_predicted, label="Predicted States (Aligned)", alpha=0.7, linewidth=3, color='orange')
axs[1].set_xlabel("Time (bins)")
axs[1].set_ylabel("State")
axs[1].set_title("Predicted States (Aligned)")
axs[1].legend()

plt.tight_layout()
plt.show()

print(f"Best accuracy: {best_acc:.2f}")

cM = confusion_matrix(true_states, remapped_predicted)
plt.figure()
plt.imshow(cM)
plt.colorbar()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Get the transition matrix from your fitted model
transition_matrix = model.transmat_

# Create a heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, cmap="YlGnBu", 
            xticklabels=["State 0", "State 1", "State 2"],
            yticklabels=["State 0", "State 1", "State 2"])
plt.title("HMM State Transition Probabilities")
plt.xlabel("To State")
plt.ylabel("From State")
plt.tight_layout()
plt.show()

# %%
import networkx as nx

# Create directed graph
G = nx.DiGraph()
states = ["Rest", "Movement", "Post-Movement"]  # Can rename to match your interpretation
for i in range(len(states)):
    G.add_node(states[i])

# Add edges with transition probabilities
for i in range(len(states)):
    for j in range(len(states)):
        if transition_matrix[i, j] > 0.00001:  # Only show non-negligible transitions
            G.add_edge(states[i], states[j], weight=transition_matrix[i, j], 
                      label=f"{transition_matrix[i, j]:.2f}")

# Plot
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G)
edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis("off")
plt.title("Neural State Transition Network")
plt.tight_layout()
plt.show()

# %%
