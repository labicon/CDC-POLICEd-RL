# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:24:01 2024

@author: Jean-Bapsite Bouvier

Comparison of the trained POLICEd and baseline models on the
high relative degree Shuttle Landing.
Creates the phase portrait of the shuttle landing for the baseline and the POLICEd policies.

 x_state (used for dynamics is an internal state)
 | Num | Name  | Observation        |  Min  |  Max | Unit |
 | --- | ----- | -------------------| ----- | ---- | -----|
 | 0   |   v   | velocity           |   0   |  Inf | m/s  |
 | 1   | gamma | flight path angle  | -pi/2 | pi/2 | rad  |
 | 2   |  h    | altitude           |   0   |  Inf | m/s  |
 
 s_state (the external state used for the buffer composed of the output iterated derivatives)
 | Num | Name | Observation                        |  Min  |  Max | Unit|
 | --- | ----- | ----------------------------------| ----- | ---- |-----|
 | 0   |   y   | output = -h                       |  -Inf |  Inf |  m  |
 | 1   |   y'  | output derivative = -v sin(gamma) |  -Inf |  Inf | m/s |
 | 2   | gamma | flight path angle                 | -pi/2 | pi/2 | rad |
"""

import torch
import numpy as np
from utils import load, nice_plot
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# wide = True # plot the whole trajectories
wide = False # plot zoomed in trajectories showing the buffer and baseline diverging

#%% Extra function

def add_arrow(line, position=None, direction='right', size=15, arrow_len = 7):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
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

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
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

POLICEd_args, POLICEd_state_norm, POLICEd_env, POLICEd_agent = load("saved/POLICEd_128")
base_args, base_state_norm, env, base_agent = load("saved/baseline")


#%% Compare their rollouts

######### x state :           v [ft/s]  gamma [rad]  h [ft]


# initial_x_states = np.array([[env.v_0_min, env.gamma_0_min, env.h_0_min],
#                               # [env.v_0_min, env.gamma_0_min, env.h_0_max], # h_0_min = h_0_max
#                               [env.v_0_min, env.gamma_0_max, env.h_0_min],
#                               # [env.v_0_min, env.gamma_0_max, env.h_0_max],
#                               [env.v_0_max, env.gamma_0_min, env.h_0_min],
#                               # [env.v_0_max, env.gamma_0_min, env.h_0_max],
#                               [env.v_0_max, env.gamma_0_max, env.h_0_min]])
#                               # [env.v_0_max, env.gamma_0_max, env.h_0_max]])

if wide:                            
    initial_x_states = np.array([[370., -10*np.pi/180, 500.],
                                 [400., -20*np.pi/180, 500.],
                                 [450., -30*np.pi/180, 500.],
                                 [500., -40*np.pi/180, 500.]])
else:
    ### Crashing the baseline but not POLICEd
    initial_x_states = np.array([[330.,  -23*np.pi/180, 500.],
                                  [350.,  -30*np.pi/180, 500.],
                                  [400.,  -45*np.pi/180, 500.],
                                  [450.,  -55*np.pi/180, 500.],
                                  [500.,  -57.5*np.pi/180, 500.]])


def multi_rollouts(args, env, agent, state_norm, initial_x_states, title=""):
    """Rollouts a trajectory from the policy"""
    
    N = initial_x_states.shape[0]
    s_Trajs = []
    x_Trajs = []
    Actions = []
     
    for i in range(N):
        env.x_reset_to(initial_x_states[i])
        
        s_Trajectory = np.zeros((env.max_episode_steps+1, env.state_size))
        x_Trajectory = np.zeros((env.max_episode_steps+1, env.state_size))
        s_Trajectory[0] = env.s_state
        x_Trajectory[0] = initial_x_states[i]
        actions = np.zeros((env.max_episode_steps, env.action_size))
        
        for t in range(env.max_episode_steps):
            with torch.no_grad():
                action = agent.evaluate(env.s_state, state_norm)
            s_state, reward, done, _ = env.step(action)
            s_Trajectory[t+1] = s_state
            x_Trajectory[t+1] = env.x_state
            actions[t] = action
            if done: break
        if not env.in_landing_box():
            v0, gamma0, h0 = initial_x_states[i]
            h_dot = s_state[1]
            print(title + f" rollout from v = {v0:.0f} ft/s, gamma = {gamma0*180/np.pi:.1f} deg, h = {h0:.0f} ft crashes with vertical velocity {h_dot:.0f} ft/s")
        
        s_Trajs.append(s_Trajectory[:t+2])
        x_Trajs.append(x_Trajectory[:t+2])
        Actions.append(actions[:t+1])
        
    return s_Trajs, x_Trajs, Actions

POLICEd_s_Trajs, POLICEd_x_Trajs, POLICEd_Actions = multi_rollouts(POLICEd_args, POLICEd_env, POLICEd_agent, POLICEd_state_norm, initial_x_states, title="POLICEd")
               
base_s_Trajs, base_x_Trajs, base_Actions = multi_rollouts(base_args, env, base_agent, base_state_norm, initial_x_states, title="Baseline")




   
fig, ax = nice_plot() 
plt.title("test plot")
plt.show()


#%% Phase portrait plotting

fig, ax = nice_plot()

buffer_color = (144./255, 238./255, 144./255)
start_color = (0.2, 0.2, 0.2)
landing_color = (255./255, 192./255, 203./255)
POLICEd_color = '#1f77b4'
base_color = '#ff7f0e'

plt.title("Phase portrait")
buffer_2D_vertices = [[-env.y_min, -env.s2_min],
                      [-env.y_max, -env.s2_min],
                      [-env.y_min, -env.y_dot_max]]
buffer = plt.Polygon(xy=buffer_2D_vertices, color=buffer_color)
ax.add_patch(buffer)

landing_zone_vertices = [[-env.y_min, -env.s2_min],
                         [-env.y_max, -env.s2_min],
                         [-env.y_max, 0],
                         [-env.y_min, 0.]]
landing_zone = plt.Polygon(xy=landing_zone_vertices, color=landing_color)
ax.add_patch(landing_zone)
plt.xlabel("Altitude $h$ (ft)")
plt.ylabel("Vertical speed $\dot h$ (ft/s)")

##### Trajectories
max_h_dot = 0.
min_h_dot = -300.
N = len(base_s_Trajs)
for i in range(N):
    ### Baseline 
    line = plt.plot(-base_s_Trajs[i][:, 0], -base_s_Trajs[i][:, 1], color=base_color, linestyle=':', linewidth=3)[0]
    if wide:
        add_arrow(line, position=200, size=30, arrow_len = 25)
    else:
        add_arrow(line, position=40, size=30) # for zoomed in plot
        ### Interpolate last state at h=0
        x1 = -base_s_Trajs[i][-1, 0]
        x2 = -base_s_Trajs[i][-2, 0]
        y1 = -base_s_Trajs[i][-1, 1]
        y2 = -base_s_Trajs[i][-2, 1]
        r = x2/(x2 - x1)
        y = r*y1 + (1-r)*y2
        plt.scatter(0, y, s=100, color="red", marker="X", zorder=3)
    # plt.scatter(-base_s_Trajs[i][0, 0], -base_s_Trajs[i][0, 1], s=30, color=start_color, zorder=3)
    
    for t in range(base_s_Trajs[i].shape[0]):
        h_dot = -base_s_Trajs[i][t, 1]
        if h_dot < min_h_dot:
            min_h_dot = h_dot
        elif h_dot > max_h_dot:
            max_h_dot = h_dot
            
    ### POLICEd
    line = plt.plot(-POLICEd_s_Trajs[i][:, 0], -POLICEd_s_Trajs[i][:, 1], color=POLICEd_color, linewidth=3)[0]
    if wide:
        add_arrow(line, position=400, size=30, arrow_len = 25)
    else:
        add_arrow(line, position=61, size=30) # for zoomed in plot
    
    
    for t in range(POLICEd_s_Trajs[i].shape[0]):
        h_dot = -POLICEd_s_Trajs[i][t,1]
        if h_dot < min_h_dot:
            min_h_dot = h_dot
        elif h_dot > max_h_dot:
            max_h_dot = h_dot
            
###### plt.plot(np.array([env.buffer_max[1], env.buffer_max[1]]), np.array([min_h_dot, max_h_dot]), color=constraint_color, linestyle='--', linewidth=4)  
crash_marker = mlines.Line2D([], [], color="red", marker='X', linestyle='None', markersize=10, label='crash')
buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='^', linestyle='None', markersize=20, label='buffer $\mathcal{B}$')
landing_marker = mlines.Line2D([], [], color=landing_color, marker='s', linestyle='None', markersize=15, label='target')
base_marker = mlines.Line2D([], [], color=base_color, linewidth='4', markersize=15, linestyle=':', label='baseline')
POLICEd_marker = mlines.Line2D([], [], color=POLICEd_color, linewidth='4', markersize=15, label='POLICEd')

if wide:
    ### Full view
    ax.set_yticks([0, -100, -200])
    ax.set_xticks([0, 50, 500])
    plt.xlim([-0.1, env.h_0_max*1.01])
    plt.ylim([min_h_dot*1.01, max_h_dot])
    plt.legend(handles=[landing_marker, buffer_marker, base_marker, POLICEd_marker],
                frameon=False,
                borderpad=.0, labelspacing=0.3, handletextpad=0.5, handlelength=1.4,
                loc='lower left')
    # plt.savefig("figures/shuttle_wide.eps", bbox_inches='tight', format="eps", dpi=1200)


else:
    ### Zoomed in
    ax.set_yticks([0, -100, -200])
    ax.set_xticks([0, 50, 100])
    plt.xlim([-1.5, 100])
    plt.ylim([-150, 0])
    plt.legend(handles=[buffer_marker, landing_marker, base_marker, crash_marker, POLICEd_marker],
                frameon=False,
                borderpad=.0,      # fractional whitespace inside the legend border, in font-size units
                labelspacing=0.3,  # vertical space between the legend entries, in font-size units
                handletextpad=0.4, # pad between the legend handle and text, in font-size units
                handlelength=1.2,  # length of the legend handles, in font-size units
                loc='upper right')
    
    # plt.savefig("figures/shuttle_multi_phase.eps", bbox_inches='tight', format="eps", dpi=1200)
    # plt.savefig("figures/shuttle_phase_base_POLICEd.svg", bbox_inches='tight')

plt.show()




    
    







#%% Trajectories plotting
    
# N = initial_x_states.shape[0]
N = 3


fig, ax = nice_plot()
for i in range(N):
    time = env.dt * np.arange(POLICEd_x_Trajs[i].shape[0])
    plt.plot(time, POLICEd_x_Trajs[i][:, 0], color=POLICEd_color, linewidth=3)
    
    time = env.dt * np.arange(base_x_Trajs[i].shape[0])
    plt.plot(time, base_x_Trajs[i][:, 0], color=base_color, linestyle=':', linewidth=3)
    
plt.ylabel("velocity $v$ (ft/s)")
plt.xlabel("time (s)")
plt.legend(handles=[base_marker, POLICEd_marker], frameon=False, borderpad=.0,
            labelspacing=0.3, handletextpad=0.4, handlelength=1.2, loc='upper right')

# plt.savefig("figures/shuttle_v.eps", bbox_inches='tight', format="eps", dpi=1200)
plt.savefig("figures/shuttle_v.svg", bbox_inches='tight')

plt.show()

    

fig, ax = nice_plot()
for i in range(N):
    time = env.dt * np.arange(POLICEd_x_Trajs[i].shape[0])
    plt.plot(time, POLICEd_x_Trajs[i][:, 1]*180/np.pi, color=POLICEd_color, linewidth=3)
    
    time = env.dt * np.arange(base_x_Trajs[i].shape[0])
    plt.plot(time, base_x_Trajs[i][:, 1]*180/np.pi, color=base_color, linestyle=':', linewidth=3)    
    
plt.ylabel("flight path angle $\gamma$ (deg)")
plt.xlabel("time (s)")
plt.legend(handles=[base_marker, POLICEd_marker], frameon=False, borderpad=.0,
            labelspacing=0.3, handletextpad=0.4, handlelength=1.2, loc='lower right')
# plt.savefig("figures/shuttle_gamma.eps", bbox_inches='tight', format="eps", dpi=1200)
plt.savefig("figures/shuttle_gamma.svg", bbox_inches='tight')
plt.show()
    


fig, ax = nice_plot()
for i in range(N):
    time = env.dt * np.arange(POLICEd_x_Trajs[i].shape[0])
    plt.plot(time, POLICEd_x_Trajs[i][:, 2], color=POLICEd_color, linewidth=3)
    
    time = env.dt * np.arange(base_x_Trajs[i].shape[0])
    plt.plot(time, base_x_Trajs[i][:, 2], color=base_color, linestyle=':', linewidth=3)
    
plt.ylabel("altitude $h$ (ft)")
plt.xlabel("time (s)")
plt.legend(handles=[base_marker, POLICEd_marker], frameon=False, borderpad=.0,
            labelspacing=0.3, handletextpad=0.4, handlelength=1.2, loc='upper right')
# plt.savefig("figures/shuttle_h.eps", bbox_inches='tight', format="eps", dpi=1200)
# plt.savefig("figures/shuttle_h.svg", bbox_inches='tight')
plt.show()




fig, ax = nice_plot()
for i in range(N):
    time = env.dt * np.arange(POLICEd_Actions[i].shape[0])
    Alpha = env.action_to_alpha(POLICEd_Actions[i])
    plt.plot(time, Alpha, color=POLICEd_color, linewidth=3)
    
    time = env.dt * np.arange(base_Actions[i].shape[0])
    Alpha = env.action_to_alpha(base_Actions[i])
    plt.plot(time, Alpha, color=base_color, linestyle=':', linewidth=3)
    
plt.ylabel("angle of attack $\\alpha$ (deg)")
plt.xlabel("time (s)")
plt.legend(handles=[base_marker, POLICEd_marker], frameon=False, borderpad=.0,
            labelspacing=0.3, handletextpad=0.4, handlelength=1.2, loc='lower right')
# plt.savefig("figures/shuttle_alpha.eps", bbox_inches='tight', format="eps", dpi=1200)
# plt.savefig("figures/shuttle_alpha.svg", bbox_inches='tight')
plt.show()





