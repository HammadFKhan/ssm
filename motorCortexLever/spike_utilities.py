
from scipy.ndimage import gaussian_filter1d
import numpy as np


def compute_binned_spike_data(spike_data, sigma = 5, bin_size_ms = 10):
    """
    Compute continuous firing rates from binned spike data using Gaussian smoothing.
    """
    # Check input dimensions
    if len(spike_data.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {spike_data.shape}")
    
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

def bin_lever_trace(leverTrace, bin_size_ms=10):
    n_timebins = len(leverTrace)
    n_bins = n_timebins // bin_size_ms
    n_timebins_trimmed = n_bins * bin_size_ms
    leverTrace = leverTrace[:n_timebins_trimmed]
    # mean or some other stat per bin
    leverTrace_binned = leverTrace.reshape(n_bins, bin_size_ms).mean(axis=1)
    print("Binned data shape:", leverTrace_binned.shape)
    return leverTrace_binned
