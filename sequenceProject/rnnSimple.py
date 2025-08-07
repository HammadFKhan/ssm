#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os # For saving files


import RNN_utilities as RNU
# Define common parameters

# Training hyperparameters
epochs = 5000
initial_lr = 0.001
lr_reduction_factor = 0.95
lr_reduction_patience = 1000

params = {
    'trialsN': 100,
    'T': 150,
    'input_padding_length': 25,
    't_signal_input': np.linspace(0, 1, 100),
    'base_amplitude_input': 3.0,
    'noise_mean_input': 0.1,
    'noise_std_dev_input': 0.5,
    'amplitude_jitter_std_input': 0.5,
    'phase_jitter_std_input': 0.5 * np.pi,
    't_trajectory': np.linspace(0, 2 * np.pi, 150),
    'base_a_traj': 5.0,
    'base_b_traj': 3.0,
    'base_c_traj': 2.0,
    'a_jitter_std_traj': 1.0,
    'b_jitter_std_traj': 1.0,
    'c_jitter_std_traj': 1.0,
    'phase_jitter_std_traj': 0.001,
    'noise_std_traj_xy': 0.5,
    'noise_std_traj_z': 0.05,
    'target_smoothing_sigma': 5,
    'base_frequency_input': 0.5 # New default value
}
# Frequencies to test
base_frequencies_to_test = [0.5]

all_training_results = {}

for freq in base_frequencies_to_test:
    print(f"\n--- Starting training for base_frequency_input: {freq} ---")

    # 1. Generate Data for current frequency
    inputs, targets, trajectories_np_unsmoothed = RNU.generate_data(params,base_frequency_input=freq)

    # 2. Re-initialize Model for each frequency
    model = RNU.SimpleRNN(input_size=1, hidden_size=100, output_size=3)

    # 3. Train Model
    trained_model, output_val, train_losses, val_losses, X_val, y_val = RNU.train_model(
        model, inputs, targets, epochs, initial_lr, lr_reduction_factor, lr_reduction_patience
    )

    # Store results
    all_training_results[freq] = {
        'model': trained_model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'output_val': output_val,
        'inputs': inputs,
        'targets': targets,
        'X_val': X_val,
        'y_val': y_val
    }

    # 4. Plot Results
    #plot_results(inputs, targets, output_val, train_losses, val_losses, freq)

    # 5. Save Model Data (Optional, if you want to save each model's output)
    from scipy.io import savemat
    output_n_current = output_val.detach().cpu().numpy()
    targets_np_current = targets.detach().cpu().numpy()
    input_np_current = inputs.detach().cpu().numpy()

    RNN_output_data = {
        'RNN_output': output_n_current,
        'input': input_np_current,
        'target': targets_np_current
    }
    file_name_mat = f"RNN_model_data_freq_{str(freq).replace('.', '_')}.mat"
    savemat(file_name_mat, {'RNN_data': RNN_output_data})
    print(f"Model data for frequency {freq} saved: {os.path.exists(file_name_mat)}")

print("\n--- All training runs completed ---")

# %%
