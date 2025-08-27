
"""
Call figures for RSLDS sq tasks

Parameters
----------
ylabel : str
    The label text.

labelpad : float, default: :rc:`axes.labelpad`
    Spacing in points from the Axes bounding box including ticks
    and tick labels.  If None, the previous value is left as is.

loc : {'bottom', 'center', 'top'}, default: :rc:`yaxis.labellocation`
    The label position. This is a high-level alternative for passing
    parameters *y* and *horizontalalignment*.

Other Parameters
----------------
**kwargs : `.Text` properties
    `.Text` properties control the appearance of the label.

See Also
--------
text : Documents the properties supported by `.Text`.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap


def plot_rslds_states(rslds_states,num_states,filename = "rslds_Discretestate.pdf"):
    """
    Call figures for RSLDS sq tasks

    Parameters
    ----------
    rslds_states : calculated stats after model narray

    num_states : # of states that user defined (int)

    filename: optional filename for pdf export
    See Also
    --------
    text : Documents the properties supported by `.Text`.
    """

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # Create figure
    fig, axs = plt.subplots(1, 1, figsize=(8, 2), sharex=True)
    colors = sns.color_palette("viridis", num_states)
    state_cmap = ListedColormap(colors)
    # Plot inferred states with better colormap and colorbar
    im2 = axs.imshow(rslds_states[None, :]+1, aspect="auto", cmap=state_cmap, 
                            vmin=1, vmax=num_states, interpolation='none',
                            extent=[0, len(rslds_states), -0.5, 0.5])


    axs.set_ylabel("RSLDS Inferred $z$", fontsize=12)
    axs.yaxis.set_ticks([])  # Remove y-axis ticks
    cbar2 = fig.colorbar(im2, ax=axs, orientation="vertical", fraction=0.046, pad=0.04)
    cbar2.set_label("State", fontsize=10)
    cbar2.ax.tick_params(labelsize=10)


    # Add shared x-axis label
    axs.set_xlabel("Time Bins", fontsize=12)
    axs.set_xlim(0,5000)
    # Adjust layout for better spacing
    plt.tight_layout()
    
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    # Show the plot
    plt.show()

def plot_trail_pca(latent_dynamics,num_timepoints_per_trial = 250,filename='trialPCA.pdf'):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # Setup
    numPC_show = 6
    colors = sns.color_palette("viridis", numPC_show)
    state_cmap = ListedColormap(colors)
    num_total_timepoints, _ = latent_dynamics.shape
    num_trials = num_total_timepoints // num_timepoints_per_trial

    # Prepare subplots: numPC_show rows, 2 columns
    fig, axs = plt.subplots(numPC_show, 2, figsize=(12, 2 * numPC_show), sharex=True)
    comp_combine = []
    for comp in range(numPC_show):
        comp_all_time = latent_dynamics[:, comp]
        comp_by_trial = comp_all_time[:num_trials * num_timepoints_per_trial].reshape(num_trials, num_timepoints_per_trial)
        x = np.linspace(-3.5, 1.5, num_timepoints_per_trial)
        comp_combine.append(comp_by_trial) 
        # First column: Overlay all trials in faint color, then the mean in black
        for trial in range(num_trials):
            axs[comp, 0].plot(x, comp_by_trial[trial, :], alpha=0.25, color=colors[comp])
        axs[comp, 0].plot(x, comp_by_trial.mean(0), alpha=1, color='black', linewidth=2)
        axs[comp, 0].set_ylabel(f'Latent {comp+1}', fontsize=12)
        if comp == numPC_show - 1:
            axs[comp, 0].set_xlabel('Time', fontsize=12)
        axs[comp, 0].set_title('All Trials + Mean')
        
        # Second column: Only mean
        axs[comp, 1].plot(x, comp_by_trial.mean(0), alpha=1, color='black', linewidth=2)
        if comp == numPC_show - 1:
            axs[comp, 1].set_xlabel('Time', fontsize=12)
        axs[comp, 1].set_title('Mean Only')

    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Figure saved as {filename}")

def plot_trajectory_states(rslds_states, latent_dynamics, ax=None, ls="-"):
            colors = sns.color_palette("viridis", max(rslds_states))
            zcps = np.concatenate(([0], np.where(np.diff(rslds_states))[0] + 1, [rslds_states.size]))
            if ax is None:
                fig = plt.figure(figsize=(4, 4))
                ax = fig.gca()
            for start, stop in zip(zcps[:-1], zcps[1:]):
                ax.plot(latent_dynamics[start:stop + 1, 0],
                        latent_dynamics[start:stop + 1, 1],
                        lw=1, ls=ls,
                        color=colors[rslds_states[start] % len(colors)],
                        alpha=1.0)
            plt.title("Inferred, Laplace-EM")
            plt.tight_layout()


def plot_inferred_latent_dynamics(latent_dynamics,inferred_latent_dynamics):

    plt.figure(figsize=(6, 6))
    # Create two subplots (2 rows, 1 column)
    ax1 = plt.subplot(211)  # First subplot (top)
    ax2 = plt.subplot(212)  # Second subplot (bottom)


    # Plot data on each subplot
    ax1.plot(inferred_latent_dynamics[:,0], '-k', lw=1)
    ax1.plot(latent_dynamics[:,0], '-r', lw=1)

    ax2.plot(inferred_latent_dynamics[:,1], '-k', lw=1)
    ax2.plot(latent_dynamics[:,1], '-r', lw=1)


    # Set x-axis limits for both subplots (optional)
    ax1.set_xlim(0, 1000)
    ax2.set_xlim(0, 1000)

    plt.tight_layout()  # Improves spacing
    plt.show()
def plot_inferred_spks(binned_spike_data,inferred_spike_dynamics):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(6, 6))

    # Create two subplots (2 rows, 1 column)
    ax1 = plt.subplot(111)  # First subplot (top)
    

    # Plot data on each subplot
    ax1.plot(inferred_spike_dynamics[:,:1], '-r', lw=1)
    ax1.plot(binned_spike_data[:,:1], '-k', lw=1)

    # Plot the smoothed observations
    N = 3
    plt.figure(figsize=(8,4))
    plt.plot(binned_spike_data[:,:N] + N * np.arange(N), '-k', lw=2)
    plt.plot(inferred_spike_dynamics[:,:N] + N * np.arange(N), '-', lw=2)
    plt.ylabel("$y$")
    plt.xlabel("time")
    plt.xlim(0, 1000)

    # Set x-axis limits for both subplots (optional)
    ax1.set_xlim(0, 1000)

    plt.tight_layout()  # Improves spacing
    plt.show()

def plot_inferred_population(binned_spike_data,inferred_spike_data,
                             x_min = 0,x_max = 5500,
                             filename = "Inferred_Spike_state.pdf"):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
     # Create publication-quality figure with better dimensions
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)


    # Plot binned spike counts
    im1 = axs[0].imshow(np.transpose(binned_spike_data), 
                    aspect='auto', 
                    cmap='plasma',  # Scientific colormap that's colorblind-friendly
                    interpolation='none',
                    vmin = 0, vmax = 8)  # Preserves exact values
    axs[0].set_title("Spike Data", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("Neurons", fontsize=12)

    # Add gridlines to help identify neuron positions
    axs[0].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # Custom colorbar with proper positioning
    cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Spike Count", fontsize=12)
    cbar1.ax.tick_params(labelsize=10)

    # Plot smoothed spikes with different colormap for visual distinction
    im2 = axs[1].imshow(np.transpose(inferred_spike_data), 
                    aspect='auto', 
                    cmap='plasma',  # Different colormap to distinguish from raw data
                    interpolation='none', vmin = 0, vmax = 8)
    axs[1].set_title("Inferred Spike Counts", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Time Bins", fontsize=12)
    axs[1].set_ylabel("Neurons", fontsize=12)

    # Add gridlines
    axs[1].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1].set_xlim(x_min,x_max)
    # Custom colorbar for smooth data
    cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Inferred Spike Count", fontsize=12)
    cbar2.ax.tick_params(labelsize=10)

    # Add proper tick formatting for both axes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_yticks(np.arange(0, np.transpose(binned_spike_data).shape[0], 5))
        ax.set_yticklabels([f"{i}" for i in range(0, np.transpose(binned_spike_data).shape[0], 5)])

    # Adjust spacing between subplots
    plt.tight_layout()


    # Optional: Add a super title
    fig.suptitle("Neural Population Activity", fontsize=16, y=1.02)
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_traj_spk_video(q_lem_y,latent_dynamics,lever_trace,
                        max_limit = 20000,
                        start_interval = 20,
                        end_interval = 5,
                        speedup_duration = 2500,
                        fname = "traj_video.mp4"):
    import matplotlib.animation as animation
    # Create figure with 2 vertical subplots
    fig, (ax_lever,ax_img, ax_traj ) = plt.subplots(3, 1, figsize=(5, 6), 
                                                    gridspec_kw={'height_ratios': [0.5, 4, 4]}, 
                                                    sharex=False)
    ax_lever.axis('off')
    plt.subplots_adjust(hspace=0.01)
    ax_lever.set_ylabel("Lever", fontsize=12)
    ax_lever.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    # Top subplot: image plot of spike counts
    im2 = ax_img.imshow(np.transpose(q_lem_y), aspect='auto', cmap='plasma',
                        interpolation='none', vmin=0, vmax=8)
    ax_lever.set_title("Inferred Spike Counts", fontsize=14, fontweight='bold')
    ax_img.set_xlabel("Time Bins", fontsize=12)
    ax_img.set_ylabel("Neurons", fontsize=12)
    ax_img.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # Initial x-axis window for the image plot
    window_width = 250
    ax_img.set_xlim(0, window_width)

    # Colorbar for the image plot
    cbar2 = fig.colorbar(im2, ax=ax_img, fraction=0.046, pad=0.04)
    cbar2.set_label("Inferred Spike Count", fontsize=12)
    cbar2.ax.tick_params(labelsize=10)

    # Bottom subplot: latent dynamics trajectory plot
    ax_traj.set_xlim(latent_dynamics[:,0].min(), latent_dynamics[:,0].max()//2)
    ax_traj.set_ylim(latent_dynamics[:,1].min(), latent_dynamics[:,1].max()//2)
    ax_traj.set_title("Latent Dynamics Trajectory", fontsize=14, fontweight='bold')
    ax_traj.set_xlabel("Dimension 1", fontsize=12)
    ax_traj.set_ylabel("Dimension 2", fontsize=12)

    # Light gray line for full history
    history_line, = ax_traj.plot([], [], lw=1, ls="-", alpha=0.3, color='gray')
    vline = ax_img.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

    lever_line, = ax_lever.plot(lever_trace, lw=2, ls="-", alpha=1, color='black')

    # Thicker blue line segment for current point + previous 5 points
    current_segment, = ax_traj.plot([], [], lw=3, ls="-", color='blue', alpha=1.0)

    plt.tight_layout()
    #frames = tqdm(range(max_limit - window_width))
    import time

    start_interval = 20  # starting interval (ms)
    end_interval = 5     # target interval (ms)
    speedup_duration = 2500  # number of frames to speed up over

    last_update = time.time()
    frame_index = 0
    def animate(i):
        global last_update, frame_index
        # Calculate current interval (linear decrease)
        #current_interval = max(end_interval, start_interval - (start_interval - end_interval) * (i / speedup_duration))
        # Wait until current_interval has passed
        #now = time.time()
        #if (now - last_update) < current_interval / 1000.0:
            # Not enough time has passed, skip updating animation for this frame
        #    return im2, current_segment, history_line, vline, lever_line

        #last_update = now  # update time
        # Animate image subplot: scroll x-axis window horizontally
        start = i
        mid = start+window_width//2
        end = i + window_width
        if end > max_limit:
            end = max_limit
            start = max_limit - window_width
        ax_img.set_xlim(start, end)
        ax_lever.set_xlim(start, end)
        # Move vertical line to current frame
        vline.set_xdata(mid)
        # Animate trajectory subplot
        start_traj = max(0, mid-5)
        current_segment.set_data(latent_dynamics[start_traj:mid+1, 0],
                                  latent_dynamics[start_traj:mid+1, 1])
        history_line.set_data(latent_dynamics[:mid, 0], latent_dynamics[:mid, 1])
        print(f"Animation progress: Frame {i+1} / {max_limit - window_width}")

        return im2, current_segment, history_line,vline,lever_line

    ani = animation.FuncAnimation(fig, animate,
                                frames=max_limit - window_width,
                                interval=10, blit=True)
    ani.save(fname)

    plt.show()

def plot_transition_matrix(transition_matrix, filename = "Transition_state.pdf"):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(10, 8))
    # Create the heatmap with custom styling
    ax = sns.heatmap(
        transition_matrix, 
        annot=True,           # Show values in cells
        fmt=".2f",            # Format to 2 decimal places
        cmap="PuRd",        # Blue-green colormap
        cbar=True,            # Show color scale
        square=True,          # Make cells square
        linewidths=0.5,       # Add thin grid lines
        linecolor="white",    # White grid lines
        xticklabels=["State 1", "State 2", "State 3"],
        yticklabels=["State 1", "State 2", "State 3"]
    )

    # Add descriptive labels and title
    plt.title("Transition Matrix from rSLDS", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("To State", fontsize=14, labelpad=10)
    plt.ylabel("From State", fontsize=14, labelpad=10)

    # Adjust text color based on background for better readability
    for text in ax.texts:
        value = float(text.get_text())
        if value > 0.5:  # Dark background needs white text
            text.set_color('white')

    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Transition Probability", fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)

def plot_state_probability(rslds_states,filename = "M1_state.pdf"):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
        # Count occurrences of each unique value
    unique, counts = np.unique(rslds_states, return_counts=True)
    proportions = counts / counts.sum()

    # Prepare labels with percentages
    labels = [f"{u} ({p*100:.0f}%)" for u, p in zip(unique, proportions)]

    # Colors (customize as needed)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create a donut plot
    fig, ax = plt.subplots(figsize=(3,3))
    wedges, texts, autotexts = ax.pie(
        proportions, 
        labels=('State 1','State 2','State 3'), 
        colors=colors[:len(unique)], 
        autopct='%1.1f%%',
        startangle=90, 
        wedgeprops=dict(width=0.4)

    )
    ax.set_title('M1')
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_trial_inferred_spks(binned_spike_data,inferred_spikes,inferred_latent_dynamics,
                             trial_time = 250,
                             filename = "TrialAveraged_Spike_state.pdf"):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # Assuming spike_data is your 2D array with shape (time_total, neurons)
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

    
    _,spike_trial_average = make_trials(binned_spike_data,trial_time)
    _,inferred_spike_average = make_trials(inferred_spikes,trial_time)
    _,latent_average = make_trials(inferred_latent_dynamics,trial_time)
    # Create publication-quality figure with better dimensions
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)


    # Plot binned spike counts
    im1 = axs[0].imshow(np.transpose(spike_trial_average), 
                    aspect='auto', 
                    cmap='plasma',  # Scientific colormap that's colorblind-friendly
                    interpolation='none',
                    vmin = 0, vmax = 8)  # Preserves exact values
    axs[0].set_title("Actual Trial-averaged Spike", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("Neurons", fontsize=12)

    # Add gridlines to help identify neuron positions
    axs[0].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # Custom colorbar with proper positioning
    cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Spike Count", fontsize=12)
    cbar1.ax.tick_params(labelsize=10)

    # Plot smoothed spikes with different colormap for visual distinction
    im2 = axs[1].imshow(np.transpose(inferred_spike_average), 
                    aspect='auto', 
                    cmap='plasma',  # Different colormap to distinguish from raw data
                    interpolation='none', vmin = 0, vmax = 8)
    axs[1].set_title("Inferred Trial-averaged Spike", fontsize=14, fontweight='bold')
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
        ax.set_yticks(np.arange(0, np.transpose(binned_spike_data).shape[0], 5))
        ax.set_yticklabels([f"{i}" for i in range(0, np.transpose(binned_spike_data).shape[0], 5)])

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()
