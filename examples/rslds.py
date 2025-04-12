"""
Recurrent Switching Linear Dynamical System (rSLDS)
===================================================
"""

import os
import pickle

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(12345)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import ssm
from ssm.util import random_rotation


# Helper functions for plotting results
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


def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=30, nypts=30,
    alpha=0.8, ax=None, figsize=(3, 3)):

    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    log_Ps = model.transitions.log_transition_matrices(
        xy, np.zeros((nxpts * nypts, 0)), np.ones_like(xy, dtype=bool), None)
    z = np.argmax(log_Ps[:, 0, :], axis=-1)
    z = np.concatenate([[z[0]], z])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax

###############################################################################
# Setting our model parameters
# ----------------------------
#
# Let's first set a few global parameters for our model.

T = 10000
K = 4
D_obs = 10
D_latent = 2


###############################################################################
# Let's now stimulate the NASCAR data
def make_nascar_model():
    As = [random_rotation(D_latent, np.pi/24.),
      random_rotation(D_latent, np.pi/48.)]

    # Set the center points for each system
    centers = [np.array([+2.0, 0.]),
           np.array([-2.0, 0.])]
    bs = [-(A - np.eye(D_latent)).dot(center) for A, center in zip(As, centers)]

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([+0.1, 0.]))

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([-0.25, 0.]))

    # Construct multinomial regression to divvy up the space
    w1, b1 = np.array([+1.0, 0.0]), np.array([-2.0])   # x + b > 0 -> x > -b
    w2, b2 = np.array([-1.0, 0.0]), np.array([-2.0])   # -x + b > 0 -> x < b
    w3, b3 = np.array([0.0, +1.0]), np.array([0.0])    # y > 0
    w4, b4 = np.array([0.0, -1.0]), np.array([0.0])    # y < 0
    Rs = np.row_stack((100*w1, 100*w2, 10*w3,10*w4))
    r = np.concatenate((100*b1, 100*b2, 10*b3, 10*b4))

    true_rslds = ssm.SLDS(D_obs, K, D_latent,
                      transitions="recurrent_only",
                      dynamics="diagonal_gaussian",
                      emissions="gaussian_orthog",
                      single_subspace=True)
    true_rslds.dynamics.mu_init = np.tile(np.array([[0, 1]]), (K, 1))
    true_rslds.dynamics.sigmasq_init = 1e-4 * np.ones((K, D_latent))
    true_rslds.dynamics.As = np.array(As)
    true_rslds.dynamics.bs = np.array(bs)
    true_rslds.dynamics.sigmasq = 1e-4 * np.ones((K, D_latent))

    true_rslds.transitions.Rs = Rs
    true_rslds.transitions.r = r

    true_rslds.emissions.inv_etas = np.log(1e-2) * np.ones((1, D_obs))
    return true_rslds

###############################################################################
# We can sample from the model
true_rslds = make_nascar_model()
z, x, y = true_rslds.sample(T=T)

###############################################################################
# Let's fit a robust rSLDS with its default initialization
rslds_svi = ssm.SLDS(D_obs, K, D_latent,
             transitions="recurrent_only",
             dynamics="diagonal_gaussian",
             emissions="gaussian_orthog",
             single_subspace=True)

###############################################################################
# Initialize the model with the observed data.  It is important
# to call this before constructing the variational posterior since
# the posterior constructor initialization looks at the rSLDS parameters.
rslds_svi.initialize(y)

###############################################################################
# Fit with stochastic variational inference
q_elbos_svi, q_svi = rslds_svi.fit(y, method="bbvi",
                               variational_posterior="tridiag",
                               initialize=False, num_iters=1000)
xhat_svi = q_svi.mean[0]
zhat_svi = rslds_svi.most_likely_states(xhat_svi, y)

###############################################################################
# Fit with Laplace EM
rslds_lem = ssm.SLDS(D_obs, K, D_latent,
             transitions="recurrent_only",
             dynamics="diagonal_gaussian",
             emissions="gaussian_orthog",
             single_subspace=True)
rslds_lem.initialize(y)
q_elbos_lem, q_lem = rslds_lem.fit(y, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               initialize=False, num_iters=100, alpha=0.0)
xhat_lem = q_lem.mean_continuous_states[0]
zhat_lem = rslds_lem.most_likely_states(xhat_lem, y)

###############################################################################
# Plot some results
plt.figure()
plt.plot(q_elbos_svi, label="SVI")
plt.plot(q_elbos_lem[1:], label="Laplace-EM")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.tight_layout()

plt.figure(figsize=[10,4])
ax1 = plt.subplot(131)
plot_trajectory(z, x, ax=ax1)
plt.title("True")
ax2 = plt.subplot(132)
plot_trajectory(zhat_svi, xhat_svi, ax=ax2)
plt.title("Inferred, SVI")
ax3 = plt.subplot(133)
plot_trajectory(zhat_lem, xhat_lem, ax=ax3)
plt.title("Inferred, Laplace-EM")
plt.tight_layout()

plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(xhat_lem).max(axis=0) + 1
plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(xhat_lem[:,0], xhat_lem[:,1], '-k', lw=1)

plt.title("Most Likely Dynamics, Laplace-EM")

plt.show()
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
    margin = 0.001  # 50% margin
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
    rslds_lem,                  # model
    xhat_lem,               # latent states from posterior
    y,        # original data
    q_posterior=q_lem,      # posterior object
)
plt.title("rSLDS Dynamics with Neural Trajectories")
plt.show()
# %%
# Plot the dynamics vector field 

from ssm.plots import plot_dynamics_2d
q = plot_dynamics_2d(A, 
                     bias_vector=b,
                     mins=states.min(axis=0),
                     maxs=states.max(axis=0),
                     color=colors[0])

plt.plot(states[:,0], states[:,1], '-k', lw=3)
#%%
sim_states, sim_latent, sim_observations = slds.sample(10000)
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(sim_latent).max(axis=0) + 1
plot_most_likely_dynamics(slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.plot(sim_latent[:,0], sim_latent[:,1], '-k', lw=1)

plt.title("Most Likely Dynamics, Laplace-EM")
#%%
plt.figure()
plt.plot(y)
plt.show()