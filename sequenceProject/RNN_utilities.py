
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


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

# --- 1. Data Generation Function ---
# The function now takes a dictionary of parameters and can be overridden by keyword arguments
def generate_data(params, **kwargs):
    """
    Generates input signals and corresponding 3D target trajectories.
    Uses a dictionary for parameters and allows for specific overrides via kwargs.
    """
    # Create a copy of the params dictionary and update with any keyword arguments
    p = params.copy()
    p.update(kwargs)
    
    input_signals_list = []
    trajectories_list = []

    for _ in range(p['trialsN']):
        # --- Generate shared jitter parameters for the current trial ---
        current_amplitude_jitter_input = np.random.normal(0, p['amplitude_jitter_std_input'])
        current_phase_jitter_input = np.random.normal(0, p['phase_jitter_std_input'])

        current_a_traj = p['base_a_traj'] + np.random.normal(0, p['a_jitter_std_traj'])
        current_b_traj = p['base_b_traj'] + np.random.normal(0, p['b_jitter_std_traj'])
        current_c_traj = p['base_c_traj'] + np.random.normal(0, p['c_jitter_std_traj'])
        current_phase_shift_traj = np.random.normal(0, p['phase_jitter_std_traj'])

        # --- Construct Input Signal for the current trial ---
        current_amplitude_input = max(0.1, p['base_amplitude_input'] + current_amplitude_jitter_input)
        current_phase_input = current_phase_jitter_input

        noise_input = np.random.normal(p['noise_mean_input'], p['noise_std_dev_input'], size=len(p['t_signal_input']))
        signal_input = current_amplitude_input * np.sin(2 * 2 * np.pi * p['base_frequency_input'] * p['t_signal_input'] + current_phase_input) + noise_input
        
        padding_input = np.zeros(p['input_padding_length'])
        input_signal_single_trial = np.concatenate([padding_input, signal_input, padding_input])
        input_signals_list.append(input_signal_single_trial)

        # --- Construct Target Trajectory for the current trial ---
        x_traj = current_a_traj * np.cos(2 * p['base_frequency_input'] * p['t_trajectory'] + current_phase_shift_traj) + p['noise_std_traj_xy'] * np.random.randn(p['T'])
        y_traj = current_b_traj * np.sin(2 * p['base_frequency_input'] * p['t_trajectory'] + current_phase_shift_traj) + p['noise_std_traj_xy'] * np.random.randn(p['T'])
        z_traj = current_c_traj * np.sin(2 * p['base_frequency_input'] * (p['t_trajectory'] + current_phase_shift_traj)) + \
                 0.5 * np.sin(8 * (p['t_trajectory'] + current_phase_shift_traj)) + \
                 p['noise_std_traj_z'] * np.random.randn(p['T'])

        # Apply smoothing to the target trajectory components before appending to the list
        x_traj_smoothed = gaussian_filter1d(x_traj, sigma=p['target_smoothing_sigma'])
        y_traj_smoothed = gaussian_filter1d(y_traj, sigma=p['target_smoothing_sigma'])
        z_traj_smoothed = gaussian_filter1d(z_traj, sigma=p['target_smoothing_sigma'])
        
        trajectories_list.append(np.vstack([x_traj_smoothed, y_traj_smoothed, z_traj_smoothed]))

    input_signal_batch = np.array(input_signals_list)
    trajectories_np = np.stack(trajectories_list, axis=2)

    inputs = torch.tensor(input_signal_batch, dtype=torch.float32).reshape(p['trialsN'], p['T'], 1)
    targets_np_transposed = np.transpose(trajectories_np, (2, 1, 0))
    targets = torch.tensor(targets_np_transposed, dtype=torch.float32)

    return inputs, targets,trajectories_np
# --- 2. RNN Model Definition ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        rnn_out, _ = self.rnn(x, h0)
        output = self.linear(rnn_out)
        return output

# --- 3. Training Function ---
def train_model(model, inputs, targets, epochs, initial_lr, lr_reduction_factor, lr_reduction_patience,
                test_size=0.2, random_state=42):
    """
    Trains the RNN model and returns the trained model and loss histories.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Data Splitting
    X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=test_size, random_state=random_state)

    print(f"Training input shape: {X_train.shape}, Training target shape: {y_train.shape}")
    print(f"Validation input shape: {X_val.shape}, Validation target shape: {y_val.shape}")

    train_losses = []
    val_losses = []

    pbar = tqdm(range(epochs), desc="Training RNN")
    for epoch_idx in pbar:
        # Training Phase
        model.train()
        optimizer.zero_grad()
        output_train = model(X_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())

        # Validation Phase
        model.eval()
        with torch.no_grad():
            output_val = model(X_val)
            loss_val = criterion(output_val, y_val)
        val_losses.append(loss_val.item())

        # Learning rate reduction logic
        if (epoch_idx + 1) % lr_reduction_patience == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_reduction_factor
                tqdm.write(f"Epoch {epoch_idx+1}: Learning rate reduced to {param_group['lr']:.6f}")

        pbar.set_postfix_str(f"Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")

    return model, output_val, train_losses, val_losses, X_val, y_val

# --- 4. Plotting Function ---
def plot_results(inputs, targets, model_output_val, train_losses, val_losses, current_frequency, filename_prefix="RNN_results"):
    """
    Plots training/validation loss and sample input/output/target comparisons.
    """
    # Plotting Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Input Freq: {current_frequency})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting Sample Input/Output/Target
    num_trials_to_plot = min(3, inputs.shape[0]) # Plot up to 3 trials
    time = range(inputs.shape[1])

    fig, axs = plt.subplots(num_trials_to_plot, 3, figsize=(15, 5 * num_trials_to_plot))
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    for trial_idx in range(num_trials_to_plot):
        # Model Input
        axs[trial_idx, 0].plot(time, inputs[trial_idx, :, 0].detach().cpu().numpy(), linestyle='--', color='blue')
        axs[trial_idx, 0].plot(time, inputs[trial_idx, :, 1].detach().cpu().numpy(), linestyle='--', color='blue')
        axs[trial_idx, 0].set_title('Model Input')
        axs[trial_idx, 0].set_ylabel(f'Trial {trial_idx+1}')

        # RNN Output
        axs[trial_idx, 1].plot(time, model_output_val[trial_idx, :, 0].detach().cpu().numpy(), linestyle='--', label='Dim 1', color='blue')
        axs[trial_idx, 1].plot(time, model_output_val[trial_idx, :, 1].detach().cpu().numpy(), linestyle='--', label='Dim 2', color='orange')
        axs[trial_idx, 1].plot(time, model_output_val[trial_idx, :, 2].detach().cpu().numpy(), linestyle='--', label='Dim 3', color='green')
        axs[trial_idx, 1].set_title('RNN Output')
        if trial_idx == 0:
            axs[trial_idx, 1].legend()

        # Target Output
        axs[trial_idx, 2].plot(time, targets[trial_idx, :, 0].detach().cpu().numpy(), linestyle='--', label='Dim 1', color='blue')
        axs[trial_idx, 2].plot(time, targets[trial_idx, :, 1].detach().cpu().numpy(), linestyle='--', label='Dim 2', color='orange')
        axs[trial_idx, 2].plot(time, targets[trial_idx, :, 2].detach().cpu().numpy(), linestyle='--', label='Dim 3', color='green')
        axs[trial_idx, 2].set_title('Target Output')
        if trial_idx == 0:
            axs[trial_idx, 2].legend()

    plt.xlabel('Timepoints')
    fig.suptitle(f'Input/Output/Target Comparison (Input Freq: {current_frequency})', y=1.02, fontsize=16)
    plt.tight_layout()
    # Save figure
    filename = f"{filename_prefix}_freq_{str(current_frequency).replace('.', '_')}_comparison.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Figure saved as {filename}")
    plt.show()

    # Plot 3D RNN Output Trajectories
    output_n = model_output_val.detach().cpu().numpy()
    fig_3d_output = plt.figure(figsize=(8, 8))
    ax_3d_output = fig_3d_output.add_subplot(111, projection='3d')
    for i in range(output_n.shape[0]):
        ax_3d_output.plot(output_n[i, :, 0], output_n[i, :, 1], output_n[i, :, 2], alpha=0.8)
    ax_3d_output.set_xlabel('PC 1')
    ax_3d_output.set_ylabel('PC 2')
    ax_3d_output.set_zlabel('PC 3')
    ax_3d_output.set_title(f'RNN Output 3D Trajectories (Input Freq: {current_frequency})')
    plt.tight_layout()
    # Save figure
    filename_3d_output = f"{filename_prefix}_freq_{str(current_frequency).replace('.', '_')}_3d_output.pdf"
    plt.savefig(filename_3d_output, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Figure saved as {filename_3d_output}")
    plt.show()

    # Plot 3D Target Trajectories (Unsmoothed)
    targets_np_plot = targets.detach().cpu().numpy()
    fig_3d_target = plt.figure(figsize=(8, 8))
    ax_3d_target = fig_3d_target.add_subplot(111, projection='3d')
    for i in range(targets_np_plot.shape[0]):
        ax_3d_target.plot(targets_np_plot[i, :, 0], targets_np_plot[i, :, 1], targets_np_plot[i, :, 2], alpha=0.8)
    ax_3d_target.set_xlabel('PC 1')
    ax_3d_target.set_ylabel('PC 2')
    ax_3d_target.set_zlabel('PC 3')
    ax_3d_target.set_title(f'Target 3D Trajectories (Input Freq: {current_frequency})')
    plt.tight_layout()
    # Save figure
    filename_3d_target = f"{filename_prefix}_freq_{str(current_frequency).replace('.', '_')}_3d_target.pdf"
    plt.savefig(filename_3d_target, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Figure saved as {filename_3d_target}")
    plt.show()
