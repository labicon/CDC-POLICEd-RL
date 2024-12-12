# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 09:31:03 2024

@author: Jean-Bapsite Bouvier

Comparison of the trained POLICEd and baseline models on the
high relative degree Inverted Pendulum of Gymnasium.
Creates the phase portrait figure seen in the paper.
"""

import torch
import numpy as np
from utils import load, nice_plot
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#%% Extra function

def add_arrow(line, start_ind, direction='right', size=15, arrow_len=0.06):
    """
    add an arrow to a line.

    line:       Line2D object
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    
    Example use:
        t = np.linspace(-2, 2, 100)
        y = np.sin(t)
        line = plt.plot(t, y)[0]
        add_arrow(line)
    """
    
    color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    length = (xdata[end_ind] - xdata[start_ind])**2 + (ydata[end_ind] - ydata[start_ind])**2
    ratio = arrow_len/np.sqrt(length)
    x_end = xdata[end_ind]*ratio + xdata[start_ind]*(1-ratio)
    y_end = ydata[end_ind]*ratio + ydata[start_ind]*(1-ratio)

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(x_end, y_end),
        arrowprops=dict(arrowstyle="wedge", color=color), # arrowstyle="->" style
        size=size
    )
    
    
#%% Load the trained models

POLICEd_args, POLICEd_state_norm, POLICEd_env, POLICEd_agent = load("saved/POLICEd")
base_args, base_state_norm, base_env, base_agent = load("saved/baseline")


#%% Compare their rollouts

y_dot_max = base_args.max_state[3]
initial_states = np.array([[0.2, 0.1, 0., 0.6*y_dot_max],
                           [0.2, 0.1, 0., 0.96*y_dot_max],
                           [0.2, 0.1, 0., 1.2*y_dot_max]])
                           
initial_states[:, 1] *= 0.95

def traj_rollouts(args, env, agent, state_norm, initial_states):
    N = initial_states.shape[0]
    Trajs = []
    
    for i in range(N):
        
        Trajectory = np.zeros((env.max_episode_steps+1, env.state_size))
        Trajectory[0] = env.reset_to(initial_states[i])
        for t in range(env.max_episode_steps):
            with torch.no_grad():
                action = agent.evaluate(Trajectory[t], state_norm)
            Trajectory[t+1], _, done, _ = env.step(action)
            if done: break
            theta = Trajectory[t+1, 1]
            theta_dot = Trajectory[t+1, 3]
            if theta < 0.02 and theta_dot < 0: break
            if theta > env.buffer_max[1]: break
        
        Trajs.append(Trajectory[:t+2])
        
    return Trajs

base_Trajs = traj_rollouts(base_args, base_env, base_agent, base_state_norm, initial_states)
POLICEd_Trajs = traj_rollouts(POLICEd_args, POLICEd_env, POLICEd_agent, POLICEd_state_norm, initial_states)

fig, ax = nice_plot() 
plt.title("test plot to get the small fonts out")
plt.show()


#%% PLotting

fig, ax = nice_plot()

buffer_color = (144./255, 238./255, 144./255)
start_color = (0.2, 0.2, 0.2)
constraint_color = (1., 0., 0.)
POLICEd_color = '#1f77b4'
base_color = '#ff7f0e'

plt.title("Phase portrait")
buffer_2D_vertices = [[base_env.buffer_min[1], base_env.buffer_min[3]],
                      [base_env.buffer_max[1], base_env.buffer_min[3]],
                      [base_env.buffer_min[1], base_env.buffer_max[3]]]
buffer = plt.Polygon(xy=buffer_2D_vertices, color=buffer_color)
ax.add_patch(buffer)
plt.xlabel("Pole angle $\\theta$ (rad)")
plt.ylabel("Pole speed $\dot \\theta$ (rad/s)")

max_theta_dot = 0.
min_theta_dot = 1.
N = initial_states.shape[0]
for i in range(N):
    # Baseline
    traj = base_Trajs[i][:, [1,3]] # un modified traj
    line = plt.plot(traj[:, 0], traj[:, 1], color=base_color, linestyle=':', linewidth=3)[0]
    if i == 0:
        add_arrow(line, start_ind=1, size=30, arrow_len=0.14)
    else:
        add_arrow(line, start_ind=1, size=30)
    for t in range(traj.shape[0]):
        theta_dot = traj[t,1]
        if theta_dot < min_theta_dot:
            min_theta_dot = theta_dot
        elif theta_dot > max_theta_dot:
            max_theta_dot = theta_dot
            
    # POLICEd
    traj = POLICEd_Trajs[i][:, [1,3]] # un modified traj
    line = plt.plot(traj[:, 0], traj[:, 1], color=POLICEd_color, linewidth=3)[0]
    add_arrow(line, start_ind=2, size=30, arrow_len=0.14)
    plt.scatter(traj[0, 0], traj[0, 1], s=30, color=start_color, zorder=3)
    for t in range(traj.shape[0]):
        theta_dot = traj[t,1]
        if theta_dot < min_theta_dot:
            min_theta_dot = theta_dot
        elif theta_dot > max_theta_dot:
            max_theta_dot = theta_dot
            
plt.plot(np.array([base_env.buffer_max[1], base_env.buffer_max[1]]), np.array([min_theta_dot, max_theta_dot]), color=constraint_color, linestyle='--', linewidth=4)  

start_marker = mlines.Line2D([], [], color=start_color, marker='o', linestyle='None', markersize=5, label='initial states')
buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='^', linestyle='None', markersize=20, label='buffer $\mathcal{B}$')
constraint_marker = mlines.Line2D([], [], color=constraint_color, linewidth='4', linestyle='--', markersize=10, label='constraint line')
base_marker = mlines.Line2D([], [], color=base_color, linewidth='4', markersize=10, linestyle=':', label='baseline')
POLICEd_marker = mlines.Line2D([], [], color=POLICEd_color, linewidth='4', markersize=10, label='POLICEd')

ax.set_yticks([0, 1])
ax.set_xticks([0, 0.1, 0.2])
plt.xlim([0., 0.21])
plt.ylim([-0.25, max_theta_dot])

plt.legend(handles=[constraint_marker, buffer_marker, start_marker, base_marker, POLICEd_marker],
           frameon=False,
           borderpad=.0, labelspacing=0.3, handletextpad=0.5, handlelength=1.4,
           loc='center left')

# plt.savefig("figures/pendulum_multi_phase.svg", bbox_inches='tight', format="svg")
# plt.savefig("figures/pendulum_multi_phase.pdf", bbox_inches='tight', format="pdf", dpi=1200)
plt.show()






