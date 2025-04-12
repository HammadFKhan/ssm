import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

def plot_spikes_pca(binned_spike_counts,pca,latent_dynamics,filename=None):

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    from scipy.ndimage import gaussian_filter1d
    latent_dynamics_smoothed = np.zeros_like(latent_dynamics, dtype=float)
    for n in range(latent_dynamics.shape[1]):
        latent_dynamics_smoothed[:, n] = gaussian_filter1d(latent_dynamics[:, n], 1)
    latent_dynamics = latent_dynamics_smoothed  
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
    # Create figure with adjusted layout
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 6, height_ratios=[3, 1], 
                        width_ratios=[7, 0.2, 0.6, 0.6, 0.6, 0.6],  # Adjusted width ratios
                        wspace=0.15, hspace=0.25)  # Increased spacing between rows

    # Main spike data heatmap
    ax_spikes = plt.subplot(gs[0, 0])
    im1 = ax_spikes.imshow(np.transpose(binned_spike_counts[:, sort_idx]), 
                        aspect='auto', 
                        cmap='viridis',
                        interpolation='none')
    ax_spikes.set_title("Spiking Activity", fontsize=14, fontweight='bold')
    ax_spikes.set_ylabel("Neurons (sorted by PC weights)", fontsize=12)
    ax_spikes.set_xticklabels([])

    # Colorbar - directly attached to spike plot
    ax_cbar = plt.subplot(gs[0, 1])
    cbar = plt.colorbar(im1, cax=ax_cbar)
    cbar.set_label("Spike Count", fontsize=10)

    # Create three separate PC weight plots (aligned with variance explained plot)
    pc_colors = [('darkred', 'indianred'), ('navy', 'royalblue'), ('darkgreen', 'mediumseagreen')]
    ax_weights = []

    for i in range(3):
        ax = plt.subplot(gs[0, i+3])  # Align with columns of variance explained plot
        ax_weights.append(ax)
        
        weights = pc_weights[i, sort_idx]
        pos_mask = weights > 0
        neg_mask = weights < 0
        
        if np.any(pos_mask):
            ax.barh(np.arange(len(sort_idx))[pos_mask], 
                weights[pos_mask], 
                height=0.8, color=pc_colors[i][1])
        
        if np.any(neg_mask):
            ax.barh(np.arange(len(sort_idx))[neg_mask], 
                weights[neg_mask], 
                height=0.8, color=pc_colors[i][0])
        
        ax.set_title(f"PC{i+1}", fontsize=12)
        ax.set_ylim(ax_spikes.get_ylim())
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        
        max_abs_weight = max(abs(pc_weights[:3, sort_idx].min()), pc_weights[:3, sort_idx].max())
        ax.set_xlim(-max_abs_weight*1, max_abs_weight*1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Latent dynamics trajectories - make full width
    ax_latent = plt.subplot(gs[1, 0:2])  # Span across first two columns for width
    for i in range(min(3, latent_dynamics.shape[1])):
        ax_latent.plot(latent_dynamics[:, i], label=f'PC{i+1}', color=pc_colors[i][1])
    ax_latent.set_title("Latent Dynamics", fontsize=14, fontweight='bold')
    ax_latent.set_xlabel("Time Bins", fontsize=12)
    ax_latent.set_ylabel("PC Score", fontsize=12)
    ax_latent.legend(loc='upper right', bbox_to_anchor=(1.1, 1), borderaxespad=0)
    ax_latent.grid(True, alpha=0.3)
    ax_latent.spines['top'].set_visible(False)
    ax_latent.spines['right'].set_visible(False)

    # Variance explained panel aligned with PC histograms
    ax_var = plt.subplot(gs[1, 3:])
    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)

    ax_var.bar(range(1, len(explained_var)+1), explained_var, color='gray', alpha=0.7)

    ax_var2 = ax_var.twinx()
    ax_var2.plot(range(1, len(cumulative_var)+1), cumulative_var, 'o-', color='black', 
                linewidth=2, markersize=4)
    ax_var2.set_ylim([0, 90])
    ax_var2.tick_params(axis='y', which='both', right=False, labelright=False)

    ax_var.set_xlabel('PC', fontsize=12)
    ax_var.set_ylabel('Variance Explained (%)', fontsize=12)
    ax_var.set_title('Variance Explained', fontsize=14, fontweight='bold')
    ax_var.set_box_aspect(1)  # Make the plot square
    ax_var.set_ylim([0, 90])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
        print(f"Figure saved as {filename}")
    else:
        print('Figure was not saved')
        plt.show()

def plot_state_transitions(transition_matrix, latent_dynamics, true_states, min_max_bounds=(-8, 8), plot_limit=20):
    """
    Plots neural dynamics and state transitions.
    
    Parameters:
    -----------
    transition_matrix : numpy.ndarray
        State transition probability matrix
    latent_dynamics : numpy.ndarray
        Latent dynamics array of shape (n_timepoints, 3)
    true_states : numpy.ndarray
        Array of discrete states (integers)
    min_max_bounds : tuple
        Minimum and maximum bounds for the 3D plot
    plot_limit : int
        Number of time points to show in the state transition plot
    """
    # Check inputs
    assert latent_dynamics.shape[0] == true_states.shape[0], "Latent dynamics and states must have same number of time points"
    assert latent_dynamics.shape[1] >= 3, "Latent dynamics must have at least 3 dimensions"
    
    num_timepoints = latent_dynamics.shape[0]
    
    # Automatically determine axis bounds based on latent_dynamics
    min_x = np.min(latent_dynamics[:, 0])
    max_x = np.max(latent_dynamics[:, 0])
    min_y = np.min(latent_dynamics[:, 1])
    max_y = np.max(latent_dynamics[:, 1])
    min_z = np.min(latent_dynamics[:, 2])
    max_z = np.max(latent_dynamics[:, 2])

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Panel A: Transition matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(transition_matrix, 
                ax=ax1, 
                cmap='rocket',  # Use a visually appealing colormap
                annot=True,
                fmt='.2f',
                square=True,
                vmin=0, vmax=1,
                cbar_kws={'label': 'Transition probability'})
    ax1.set_title('A. State Transition Matrix', fontweight='bold')
    ax1.set_xlabel('To State')
    ax1.set_ylabel('From State')

    # Panel B: 3D and 2D projection plot combined
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')

    # 3D plot with lines connecting points
    for i in range(1, num_timepoints):
        ax2.plot3D(latent_dynamics[i-1:i+1, 0], 
                  latent_dynamics[i-1:i+1, 1], 
                  latent_dynamics[i-1:i+1, 2], 
                  'k-', alpha=0.3, linewidth=0.5)

    # Color points by state
    colors = plt.cm.viridis(true_states / max(true_states))
    scatter = ax2.scatter(latent_dynamics[:, 0], 
                         latent_dynamics[:, 1], 
                         latent_dynamics[:, 2], 
                         c=colors, s=30, alpha=0.8)

    # Bottom XY projection (PC1-PC2)
    for i in range(1, num_timepoints):
        z_vals = np.full_like(latent_dynamics[i-1:i+1, 0], min_z)
        ax2.plot(latent_dynamics[i-1:i+1, 0], 
                 latent_dynamics[i-1:i+1, 1],
                 z_vals, 'k-', alpha=0.2, linewidth=0.5)

    # Back YZ projection (PC2-PC3)
    for i in range(1, num_timepoints):
        x_vals = np.full_like(latent_dynamics[i-1:i+1, 1], min_x)
        ax2.plot(x_vals,
                 latent_dynamics[i-1:i+1, 1],
                 latent_dynamics[i-1:i+1, 2], 'k-', alpha=0.2, linewidth=0.5)

    # Side XZ projection (PC1-PC3)
    for i in range(1, num_timepoints):
        y_vals = np.full_like(latent_dynamics[i-1:i+1, 0], max_y)
        ax2.plot(latent_dynamics[i-1:i+1, 0],
                 y_vals,
                 latent_dynamics[i-1:i+1, 2], 'k-', alpha=0.2, linewidth=0.5)

    # Automatically set axis limits based on data range
    ax2.set_xlim(min_x - 0.5, max_x + 0.5)  # Add some padding for better visualization
    ax2.set_ylim(min_y - 0.5, max_y + 0.5)
    ax2.set_zlim(min_z - 0.5, max_z + 0.5)

    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    ax2.set_title('B. Neural Trajectories in Latent Space', fontweight='bold')

    # Panel C: Time-varying discrete states
    ax3 = fig.add_subplot(gs[1, 0])
    plot_time = min(plot_limit, num_timepoints)
    ax3.step(np.arange(plot_time), true_states[:plot_time]+1, where='post', color='black', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('State')
    ax3.set_title('C. Time-Varying Discrete States', fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlim([0, plot_limit])

    # Panel D: Histograms for each state
    ax4 = fig.add_subplot(gs[1, 1])

    # Define the color pairs for each state
    pc_colors = [('darkred', 'indianred'), ('navy', 'royalblue'), ('darkgreen', 'mediumseagreen')]

    # Separate data by state
    state_data = []
    for s in range(3):
        # Get data for this state
        state_mask = (true_states == s)
        state_data.append(latent_dynamics[state_mask])

    # Plot histograms for each PC component and each state
    pc_names = ['PC 1', 'PC 2', 'PC 3']
    for pc_idx in range(3):
        for state_idx in range(3):
            if len(state_data[state_idx]) > 0:  # Check if we have data for this state
                # Get primary and secondary colors for this state
                primary_color, secondary_color = pc_colors[state_idx]
                
                # Calculate histogram values
                hist, bins = np.histogram(state_data[state_idx][:, pc_idx], bins=15, density=True)
                center = (bins[:-1] + bins[1:]) / 2
                
                # Offset the histograms slightly for each PC
                offset = (pc_idx - 1) * 0.8
                
                # Plot with transparency for better visibility
                ax4.bar(center + offset, hist, width=0.5, alpha=0.6, 
                        color=secondary_color, edgecolor=primary_color, 
                        label=f'{pc_names[pc_idx]}, State {state_idx}' if state_idx == 0 else None)

    # Add legend and labels
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles[:3], pc_names, loc='upper right')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.set_title('D. State Activity Distributions', fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Add state color legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='indianred', alpha=0.6, edgecolor='darkred', label='State 0'),
        Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.6, edgecolor='navy', label='State 1'),
        Rectangle((0, 0), 1, 1, color='mediumseagreen', alpha=0.6, edgecolor='darkgreen', label='State 2')
    ]
    ax4.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()

def plot_combined_model_comparison(results_pc, results_state, n_pcs, figsize=(15, 10), 
                                  palette='colorblind', style='whitegrid'):
    """
    Create a comprehensive visualization combining PC-based and state-based model comparisons.
    
    Parameters:
    -----------
    results_pc : dict
        Dictionary containing model results by PC dimension
    results_state : dict
        Dictionary containing model results by number of states
    n_pcs : int
        Number of PCs for normalization
    figsize : tuple
        Figure size (width, height)
    palette : str
        Seaborn color palette name
    style : str
        Seaborn style ('whitegrid', 'darkgrid', etc.)
    
    Returns:
    --------
    fig : matplotlib Figure
        The figure object for further customization if needed
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set Seaborn style
    sns.set(style=style)
    colors = sns.color_palette(palette, 3)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Top row: PC-based plots
    # Plot 1: Log-likelihood by PC
    ax1 = axes[0, 0]
    x_pc = results_pc['num_pcs']
    
    sns.lineplot(x=x_pc, y=[x/y for x, y in zip(results_pc['hsmm_lls'], results_pc['num_pcs'])], 
                marker='o', linewidth=2.5, markersize=8, color=colors[0], label='HSMM', ax=ax1)
    sns.lineplot(x=x_pc, y=[x/y for x, y in zip(results_pc['hmm_lls'], results_pc['num_pcs'])], 
                marker='o', linewidth=2.5, markersize=8, color=colors[1], label='HMM', ax=ax1)
    sns.lineplot(x=x_pc, y=[x/y for x, y in zip(results_pc['rslds_lls'], results_pc['num_pcs'])], 
                marker='o', linewidth=2.5, markersize=8, color=colors[2], label='rSLDS', ax=ax1)
    
    ax1.set_xlabel('Number of PCs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Log-Likelihood / PC dimension', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fontsize=10, framealpha=0.9)
    ax1.set_title('Model Fit vs. Number of PCs', fontsize=14, fontweight='bold')
    
    # Plot 2: State prediction by PC
    ax2 = axes[0, 1]
    sns.lineplot(x=x_pc, y=results_pc['hsmm_ari'], 
                marker='o', linewidth=2.5, markersize=8, color=colors[0], label='HSMM', ax=ax2)
    sns.lineplot(x=x_pc, y=results_pc['hmm_ari'], 
                marker='o', linewidth=2.5, markersize=8, color=colors[1], label='HMM', ax=ax2)
    sns.lineplot(x=x_pc, y=results_pc['rslds_ari'], 
                marker='o', linewidth=2.5, markersize=8, color=colors[2], label='rSLDS', ax=ax2)
    
    ax2.set_xlabel('Number of PCs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Adjusted Rand Index', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, fontsize=10, framealpha=0.9)
    ax2.set_title('State Prediction vs. Number of PCs', fontsize=14, fontweight='bold')
    
    # Bottom row: State-based plots
    # Plot 3: Log-likelihood by state (with dual y-axis)
    ax3 = axes[1, 0]
    x_state = results_state['num_states']
    
    # Primary axis for HSMM and HMM
    sns.lineplot(x=x_state, y=[x/n_pcs for x in results_state['hsmm_lls']], 
                marker='o', linewidth=2.5, markersize=8, color=colors[0], label='HSMM', ax=ax3)
    sns.lineplot(x=x_state, y=[x/n_pcs for x in results_state['hmm_lls']], 
                marker='o', linewidth=2.5, markersize=8, color=colors[1], label='HMM', ax=ax3)
    
    ax3.set_xlabel('Number of States', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Log-Likelihood / PC dimension\n(HSMM, HMM)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.grid(True, alpha=0.3)
    
    # Secondary axis for rSLDS
    ax3b = ax3.twinx()
    sns.lineplot(x=x_state, y=[x/n_pcs for x in results_state['rslds_lls']], 
                marker='o', linewidth=2.5, markersize=8, color=colors[2], label='rSLDS', ax=ax3b)
    ax3b.set_ylabel('Log-Likelihood / PC dimension\n(rSLDS)', fontsize=12, fontweight='bold', color=colors[2])
    ax3b.tick_params(axis='y', labelcolor=colors[2])
    
    # Adjust the y-limits for rSLDS to focus on its variation
    rslds_values = [x/n_pcs for x in results_state['rslds_lls']]
    min_rslds = min(rslds_values)
    max_rslds = max(rslds_values)
    margin = (max_rslds - min_rslds) * 0.1  # Add 10% margin
    ax3b.set_ylim(min_rslds - margin, max_rslds + margin)
    
    # Add legends for both axes
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=10, framealpha=0.9)
    
    ax3.set_title('Model Fit vs. Number of States', fontsize=14, fontweight='bold')
    
    # Plot 4: State prediction by state
    ax4 = axes[1, 1]
    sns.lineplot(x=x_state, y=results_state['hsmm_ari'], 
                marker='o', linewidth=2.5, markersize=8, color=colors[0], label='HSMM', ax=ax4)
    sns.lineplot(x=x_state, y=results_state['hmm_ari'], 
                marker='o', linewidth=2.5, markersize=8, color=colors[1], label='HMM', ax=ax4)
    sns.lineplot(x=x_state, y=results_state['rslds_ari'], 
                marker='o', linewidth=2.5, markersize=8, color=colors[2], label='rSLDS', ax=ax4)
    
    ax4.set_xlabel('Number of States', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Adjusted Rand Index', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(frameon=True, fontsize=10, framealpha=0.9)
    ax4.set_title('State Prediction vs. Number of States', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    sns.despine(fig=fig, offset=5)  # Remove top and right spines for cleaner look
    
    # Add overall title
    fig.suptitle('Model Performance: Effect of PC Dimension and Number of States', 
                fontsize=16, fontweight='bold', y=1.02)
    
    return fig


def plot_hyperparam_model_performance(latent_dynamics, true_states, states, Hmm_states, rslds_states,
                                      results_state, results, pca, hsmm_em_lls, hmm_em_lls, q_lem_elbos,
                                      n_states=None,filename=None):
    """
    Plot hyperparameterized model performance including latent dynamics, true/inferred states,
    model performance vs. number of states/PCs, variance explained, and training convergence.

    Parameters:
    -----------
    latent_dynamics : np.ndarray
        Latent dynamics (e.g., PCA-reduced data) of shape (time_bins, n_components).
    true_states : np.ndarray
        True discrete states of shape (time_bins,).
    states : np.ndarray
        HSMM inferred discrete states of shape (time_bins,).
    Hmm_states : np.ndarray
        HMM inferred discrete states of shape (time_bins,).
    rslds_states : np.ndarray
        rSLDS inferred discrete states of shape (time_bins,).
    results_state : dict
        Results dictionary for state-dependent analysis (e.g., ARI vs. number of states).
    results : dict
        Results dictionary for PC-dependent analysis.
    pca : sklearn.decomposition.PCA
        PCA object containing explained variance information.
    hsmm_em_lls : list
        Log-likelihoods from HSMM during training.
    hmm_em_lls : list
        Log-likelihoods from HMM during training.
    q_lem_elbos : list
        ELBO values from rSLDS during training.
    n_states : int
        Number of discrete states.

    Returns:
    --------
    None. Displays the figure.
    """


    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import numpy as np
    from matplotlib.colors import ListedColormap
    import matplotlib
    # Set seaborn style for better aesthetics
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if n_states is None:
        n_states=3

    # Create a custom colormap for states that's more distinguishable
    colors = sns.color_palette("husl", n_states)
    state_cmap = ListedColormap(colors)

    # Create figure with adjusted grid layout - increased top margin for title
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1.2, 1, 2], width_ratios=[3, 1, 1, 1], 
                        hspace=0.4, wspace=0.4, top=0.95)  # Added top margin

    # Panel A: Adjust top section to make room for title
    gs_states = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0:2, 0], 
                                            height_ratios=[1.2, 1, 1, 1, 1], 
                                            hspace=0.05)

    # Title for Panel A - now in its own dedicated space

    # First, plot latent dynamics
    ax_dynamics = fig.add_subplot(gs_states[0])
    for i in range(min(3, latent_dynamics.shape[1])):
        ax_dynamics.plot(latent_dynamics[:, i], label=f'PC{i+1}', 
                        linewidth=1.5, alpha=1, 
                        color=sns.color_palette("Set2")[i])
    ax_dynamics.set_ylabel("Latent\nDynamics", fontsize=12, fontweight='bold')
    ax_dynamics.set_title("Hyperparamatized Model Performance", loc="left", fontsize=14, fontweight='bold', 
                x=0.28, y=1.05, transform=ax_dynamics.transAxes)
    ax_dynamics.legend(loc="upper right", frameon=True, fontsize=8, ncol=3)
    ax_dynamics.set_xticklabels([])
    ax_dynamics.spines['top'].set_visible(False)
    ax_dynamics.spines['right'].set_visible(False)
    ax_dynamics.grid(alpha=0.3)
    xmin, xmax = ax_dynamics.get_xlim()
    xmin = 0
    zoomFactor = 2 #how much we want to zoom in the plots for visualization
    # True states
    ax_true = fig.add_subplot(gs_states[1], sharex=ax_dynamics)
    im1 = ax_true.imshow(true_states[None, :], aspect="auto", cmap='Paired', 
                        vmin=0, vmax=n_states+5, interpolation='none',
                        extent=[0, len(true_states), -0.5, 0.5])
    ax_true.set_ylabel("True\nStates", fontsize=12, fontweight='bold')
    ax_true.set_yticks([])
    ax_true.set_xlim(xmin, xmax/zoomFactor)
    plt.setp(ax_true.get_xticklabels(), visible=False)

    # HSMM states
    ax_hsmm = fig.add_subplot(gs_states[2], sharex=ax_dynamics)
    im2 = ax_hsmm.imshow(states[None, :], aspect="auto", cmap='Paired', 
                        vmin=0, vmax=n_states+5, interpolation='none',
                        extent=[0, len(states), -0.5, 0.5])
    ax_hsmm.set_ylabel("HSMM\nStates", fontsize=12, fontweight='bold')
    ax_hsmm.set_yticks([])
    ax_hsmm.set_xlim(xmin, xmax/zoomFactor)
    plt.setp(ax_hsmm.get_xticklabels(), visible=False)

    # HMM states
    ax_hmm = fig.add_subplot(gs_states[3], sharex=ax_dynamics)
    im3 = ax_hmm.imshow(Hmm_states[None, :], aspect="auto", cmap='Paired', 
                        vmin=0, vmax=n_states+5, interpolation='none',
                        extent=[0, len(Hmm_states), -0.5, 0.5])
    ax_hmm.set_ylabel("HMM\nStates", fontsize=12, fontweight='bold')
    ax_hmm.set_yticks([])
    ax_hmm.set_xlim(xmin, xmax/zoomFactor)


    # rSLDS states
    ax_rslds = fig.add_subplot(gs_states[4], sharex=ax_dynamics)
    im4 = ax_rslds.imshow(rslds_states[None, :], aspect="auto", cmap='Paired', 
                        vmin=0, vmax=n_states+5, interpolation='none',
                        extent=[0, len(rslds_states), -0.5, 0.5])
    ax_rslds.set_ylabel("rSLDS\nStates", fontsize=12, fontweight='bold')
    ax_rslds.set_yticks([])
    ax_rslds.set_xlabel("Time Bins", fontsize=12, fontweight='bold')
    ax_rslds.set_xlim(xmin, xmax/zoomFactor)
    # Explicitly set x-axis ticks and labels
    num_ticks = 3  # Number of ticks to display
    tick_positions = np.linspace(0, len(rslds_states)/zoomFactor, num=num_ticks, endpoint=True)
    tick_labels = [f"{int(pos)}" for pos in tick_positions]
    ax_rslds.set_xticks(tick_positions)
    ax_rslds.set_xticklabels(tick_labels)

    # Make x-axis tick labels visible
    plt.setp(ax_rslds.get_xticklabels(), visible=True)
    print(xmin)
    print(xmax)

    # Panel B: Model performance vs number of states
    ax_B = fig.add_subplot(gs[0, 1])
    x_state = results_state['num_states']
    normalized_ari_rslds = [ari/max(results_state['rslds_ari']) for ari in results_state['rslds_ari']]
    normalized_ari_hmm = [ari/max(results_state['hmm_ari']) for ari in results_state['hmm_ari']]
    normalized_ari_hsmm = [ari/max(results_state['hsmm_ari']) for ari in results_state['hsmm_ari']]
    ax_B.plot(x_state, normalized_ari_rslds, 'o-', color=sns.color_palette("Set1")[2], linewidth=2.5,alpha=0.5)
    ax_B.plot(x_state, normalized_ari_hmm, 'o-', color=sns.color_palette("Set1")[1], linewidth=2.5,alpha=0.5)
    ax_B.plot(x_state, normalized_ari_hsmm, 'o-', color=sns.color_palette("Set1")[0], linewidth=2.5,alpha=0.5)
    ax_B.set_xlabel("# of states", fontsize=12, fontweight='bold')
    ax_B.set_ylabel("model performance\nELBO ratio", fontsize=12, fontweight='bold')
    ax_B.spines['top'].set_visible(False)
    ax_B.spines['right'].set_visible(False)
    ax_B.grid(alpha=0.3)

    # Panel C: Model performance vs number of PC dimensions
    ax_C = fig.add_subplot(gs[0, 2])
    x_pc = results['num_pcs']
    normalized_ll_rslds = [ll/max(results['rslds_lls']) for ll in results['rslds_lls']]
    normalized_ll_hmm = [ll/max(results['hmm_lls']) for ll in results['hmm_lls']]
    normalized_ll_hsmm = [ll/max(results['hsmm_lls']) for ll in results['hsmm_lls']]
    ax_C.plot(x_pc, normalized_ll_rslds, 'o-', color=sns.color_palette("Set1")[2], linewidth=2.5,alpha=0.5)
    ax_C.plot(x_pc, normalized_ll_hmm, 'o-', color=sns.color_palette("Set1")[1], linewidth=2.5,alpha=0.5)
    ax_C.plot(x_pc, normalized_ll_hsmm, 'o-', color=sns.color_palette("Set1")[0], linewidth=2.5,alpha=0.5)
    ax_C.set_title("Hyperparamatized Model Performance", loc="left", fontsize=14, fontweight='bold', 
                x=-0.15, y=1.05, transform=ax_C.transAxes)
    ax_C.set_xlabel("# of dimensions", fontsize=12, fontweight='bold')
    ax_C.set_ylabel("model performance\nELBO ratio", fontsize=12, fontweight='bold')
    ax_C.spines['top'].set_visible(False)
    ax_C.spines['right'].set_visible(False)
    ax_C.grid(alpha=0.3)

    # Panel D: Variance explained by PC dimension
    ax_D = fig.add_subplot(gs[0, 3])
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    ax_D.plot(range(1, len(explained_var)+1), cumulative_var, 'o-', color='black', linewidth=2)
    ax_D.set_xlabel("# of dimensions", fontsize=12, fontweight='bold')
    ax_D.set_ylabel("variance explained", fontsize=12, fontweight='bold')
    ax_D.spines['top'].set_visible(False)
    ax_D.spines['right'].set_visible(False)
    ax_D.grid(alpha=0.3)

    # Panel E: Training convergence for all models
    ax_E = fig.add_subplot(gs[1, 1:])
    # Normalize log-likelihoods for better comparison
    max_hsmm = max(hsmm_em_lls)
    max_hmm = max(hmm_em_lls)
    max_rslds = max(q_lem_elbos[1:])

    norm_hsmm = [ll/abs(max_hsmm)*1.025 for ll in hsmm_em_lls]
    norm_hmm = [ll/abs(max_hmm)*1.025 for ll in hmm_em_lls]
    norm_rslds = [ll/abs(max_rslds) for ll in q_lem_elbos[1:]]

    # Plot with custom styling
    ax_E.plot(norm_hsmm, ls='-', label="HSMM", color=sns.color_palette("Set1")[0], linewidth=2)
    ax_E.plot(norm_hmm, ls='-', label="HMM", color=sns.color_palette("Set1")[1], linewidth=2)
    ax_E.plot(norm_rslds, ls='-', label="rSLDS (Laplace-EM)", color=sns.color_palette("Set1")[2], linewidth=2)
    ax_E.set_title("Model Convergance", loc="left", fontsize=14, fontweight='bold', 
                x= 0.4, y=1.02, transform=ax_E.transAxes)
    ax_E.set_xlabel("Iterations", fontsize=14, fontweight='bold')
    ax_E.set_ylabel("normalized log-likelihood", fontsize=0.8, fontweight='bold')
    ax_E.legend(loc="lower right", frameon=True)
    ax_E.spines['top'].set_visible(False)
    ax_E.spines['right'].set_visible(False)
    ax_E.grid(alpha=0.3)
    ax_E.set_xlim(0,25)

    # Use tight_layout but with adjusted parameters
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room at top for title
    if filename:
        plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
        print(f"Figure saved as {filename}")
    else:
        print('Figure was not saved')
        plt.show()