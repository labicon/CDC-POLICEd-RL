# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:52:25 2023

@author: Jean-Baptiste Bouvier

Function utils for the PPO POLICEd applied on the Space Shuttle environmnent.
"""

import warnings
warnings.simplefilter("ignore", SyntaxWarning)


import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#%% Training utils

class ReplayBuffer:
    """Store experiences. The sampling from this buffer is handled by the update function
    of PPO, where each experience is sampled as part of mini-batches."""
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s[:self.count], dtype=torch.float)
        a = torch.tensor(self.a[:self.count], dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob[:self.count], dtype=torch.float)
        r = torch.tensor(self.r[:self.count], dtype=torch.float)
        s_ = torch.tensor(self.s_[:self.count], dtype=torch.float)
        dw = torch.tensor(self.dw[:self.count], dtype=torch.float)
        done = torch.tensor(self.done[:self.count], dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


def evaluate_policy(args, env, agent, state_norm):
    """Evaluates the policy, returns the average reward and whether the
    reward threshold has been met"""
    
    good_policy = True
    if env.initial_range:
        initial_x_states = np.array([[env.v_0_min, env.gamma_0_min, env.h_0_min],
                                     [env.v_0_min, env.gamma_0_min, env.h_0_max],
                                     [env.v_0_min, env.gamma_0_max, env.h_0_min],
                                     [env.v_0_min, env.gamma_0_max, env.h_0_max],
                                     [env.v_0_max, env.gamma_0_min, env.h_0_min],
                                     [env.v_0_max, env.gamma_0_min, env.h_0_max],
                                     [env.v_0_max, env.gamma_0_max, env.h_0_min],
                                     [env.v_0_max, env.gamma_0_max, env.h_0_max]])
    else:
        initial_x_states = np.array([env.x_0_min*0.99, env.x_0_min*1.01])
        
    N = initial_x_states.shape[0]
    worst_reward = 10000.
    for i in range(N): 
        s = env.x_reset_to(initial_x_states[i])
        done = False
        while not done:
            with torch.no_grad():
                action = agent.evaluate(s, state_norm)  # We use the deterministic policy during the evaluating
            s, _, done, _ = env.step(action)
        reward = env.final_reward()
        if worst_reward > reward:
            worst_reward = reward
        good_policy *= env.in_landing_box() # needs it to be true for each rollout
        
    return worst_reward, good_policy
    
    

def nice_plot():
    """Makes the plot nice"""
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    
    return fig, ax



def phase_portrait_y(env, s_Traj, title=""):
    """Plots the phase portrait (y, y') """
        
    fig, ax = nice_plot()
    
    # Colors
    buffer_color = (0., 1., 0., 0.5)
    start_color = (0.2, 0.2, 0.2)
    traj_color = '#1f77b4'
    
    plt.title("Phase portrait " + title)
    # Buffer 
    buffer_2D_vertices = [[env.y_min, env.s2_min],
                          [env.y_max, env.s2_min],
                          [env.y_min, env.y_dot_max]]
    buffer = plt.Polygon(xy=buffer_2D_vertices, color=buffer_color)
    ax.add_patch(buffer)
    
    # Trajectory
    plt.plot(s_Traj[:, 0], s_Traj[:, 1], color=traj_color, linewidth=3)
    plt.scatter(s_Traj[0, 0], s_Traj[0, 1], s=30, color=start_color, zorder=3)
    
    plt.xlabel("Output $y$")
    plt.ylabel("Output speed $\dot y$")
    
    start_marker = mlines.Line2D([], [], color=start_color, marker='o', linestyle='None', markersize=10, label='start')
    traj_marker   = mlines.Line2D([], [], color=traj_color, linewidth='3', markersize=10, label='trajectory')
    buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='^', linestyle='None', markersize=20, label='affine buffer $\mathcal{B}$')
    plt.legend(handles=[start_marker, traj_marker, buffer_marker], frameon=False,
                borderpad=.0, loc="lower left",
                labelspacing=0.3, handletextpad=0.5, handlelength=1.4)
    plt.show()
    
    
def phase_portrait_h(env, s_Traj, title=""):
    """Plots the phase portrait (h, h dot) """
        
    fig, ax = nice_plot()
    
    # Colors
    buffer_color = (0., 1., 0., 0.5)
    start_color = (0.2, 0.2, 0.2)
    traj_color = '#1f77b4'
    
    plt.title("Phase portrait " + title)
    # Buffer 
    buffer_2D_vertices = [[-env.y_min, -env.s2_min],
                          [-env.y_max, -env.s2_min],
                          [-env.y_min, -env.y_dot_max]]
    buffer = plt.Polygon(xy=buffer_2D_vertices, color=buffer_color)
    ax.add_patch(buffer)
    
    # Trajectory
    plt.plot(-s_Traj[:, 0], -s_Traj[:, 1], color=traj_color, linewidth=3)
    plt.scatter(-s_Traj[0, 0], -s_Traj[0, 1], s=30, color=start_color, zorder=3)
    
    plt.xlabel("Altitude $h$ (ft)")
    plt.ylabel("Vertical speed $\dot h$ (ft/s)")
    
    start_marker = mlines.Line2D([], [], color=start_color, marker='o', linestyle='None', markersize=10, label='start')
    traj_marker   = mlines.Line2D([], [], color=traj_color, linewidth='3', markersize=10, label='trajectory')
    buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='^', linestyle='None', markersize=20, label='affine buffer $\mathcal{B}$')
    plt.legend(handles=[start_marker, traj_marker, buffer_marker], frameon=False,
                borderpad=.0, loc="upper right",
                labelspacing=0.3, handletextpad=0.5, handlelength=1.4)
    plt.show()


def plot_traj(env, x_Traj, Actions=None, title=""):
    """Plots the trajectory of the space shuttle.
    x_state = [v gamma h]   action = [alpha] """
        
    N = x_Traj.shape[0] # number of steps of the trajectory before terminating
    time = env.dt * np.arange(N)
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, x_Traj[:, 0], s=10)
    plt.ylabel("Velocity (ft/s)")
    plt.xlabel("time (s)")
    plt.show()
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, x_Traj[:, 1]*180/np.pi, s=10)
    plt.plot([time[0], time[-1]], [0., 0.], c="orange")
    plt.ylabel("Flight path angle (deg)")
    plt.xlabel("time (s)")
    plt.show()
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, x_Traj[:, 2], s=10)
    plt.ylabel("altitude h (ft)")
    plt.xlabel("time (s)")
    plt.show()
    
    if Actions is not None:
        Alpha = env.action_to_alpha(Actions)
        fig, ax = nice_plot()
        plt.title(title)
        plt.scatter(time[:-1], Alpha, s=10)
        plt.ylabel("angle of attack alpha (deg)")
        plt.xlabel("time (s)")
        plt.show()


 
def training(args, env, agent, replay_buffer, env_eval, state_norm=None, reward_scaling=None, data=None):
    """Training of the PPO agent to achieve high reward"""
    
    evaluate_num = 0  # Record the number of evaluations
    if hasattr(args, 'total_steps'):
        total_steps = args.total_steps
    else:
        total_steps = 0  # Record the total steps during the training
    stable = False # Inverse pendulum not stabilized
    trained = False # Stable and buffer reached full size
    episode = 0
    buffer_vertices = args.buffer_vertices
    num_vertices = buffer_vertices.shape[0]
    
    while (not trained):
        episode += 1
        
        if episode % 5 == 0: # reset in buffer regularly            
            coefs = np.random.rand(num_vertices)
            coefs /= coefs.sum()
            s = env.s_reset_to(coefs @ buffer_vertices)
        else:
            s = env.reset()
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_reward = 0
        done = False
        while not done:
            a, a_logprob = agent.choose_action(s, state_norm)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)
            episode_reward += r

            if args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be True, we need to distinguish them;
            # dw means dead or win, there is no next state s'
            # but when reaching the max_episode_steps, there is a next state s' actually.
            if done and env.episode_step != env.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size, then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps, state_norm)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, stable = evaluate_policy(args, env_eval, agent, state_norm)
                print(f"evaluation {evaluate_num} \t reward: {evaluate_reward:.1f}")
                data.plot()
                rollout(env_eval, agent, state_norm)
            
            if stable and args.POLICEd and args.enlarging_buffer and agent.actor.iter == 0:
                print("\nStarting to enlarge the buffer")
                agent.actor.enlarge_buffer = True
            
            trained = total_steps > args.max_train_steps or stable
            if args.POLICEd and args.enlarging_buffer:
                trained = trained and agent.actor.iter > agent.actor.max_iter
           
        r = respect_ratio(args, env_eval, agent, state_norm)
        data.add(episode_reward, r)
        
    args.total_steps = total_steps
    return stable






def modified_training(args, env, agent, replay_buffer, env_eval, state_norm=None, reward_scaling=None, data=None):
    """Modified training of the PPO agent to achieve high reward for the Shuttle
    Reward is only given when the whole trajectory is complete, so the Replay Buffer
    needs to store entire trajectories
    """

    evaluate_num = 0  # Record the number of evaluations
    if hasattr(args, 'total_steps'):
        total_steps = args.total_steps
    else:
        total_steps = 0  # Record the total steps during the training
    stable = False # Inverse pendulum not stabilized
    trained = False # Stable and buffer reached full size
    episode = 0
    buffer_vertices = args.buffer_vertices
    num_vertices = buffer_vertices.shape[0]
    
    while (not trained):
        episode += 1
        
        if episode % args.freq_reset_in_buffer == 0: # reset in buffer regularly            
            coefs = np.random.rand(num_vertices)
            coefs /= coefs.sum()
            s = env.s_reset_to(coefs @ buffer_vertices)
        else:
            s = env.reset()
        
        done = False
        while not done:
            a, a_logprob = agent.choose_action(s, state_norm)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)
            if done and env.episode_step != env.max_episode_steps: dw = True
            else: dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

        c = replay_buffer.count-1 # index of the final step of the episode in replay_buffer        
        replay_buffer.r[c] += env.final_reward()
        data.add(replay_buffer.r[c], 0) # don't care about repulsion respect
        
        
        # When another full episode would not fit in the buffer, update
        if replay_buffer.count + env.max_episode_steps > args.batch_size:
            agent.update(replay_buffer, total_steps, state_norm)
            replay_buffer.count = 0

        # Evaluate the policy every 'evaluate_freq' steps
        if episode % (args.evaluate_freq//env.max_episode_steps) == 0:
            evaluate_num += 1
            evaluate_reward, stable = evaluate_policy(args, env_eval, agent, state_norm)
            print(f"evaluation {evaluate_num} \t reward: {evaluate_reward:.1f}")
            rollout(env_eval, agent, state_norm, plot=True, title=f"Eval {evaluate_num}")
            #data.plot()
        
        if stable and args.POLICEd and args.enlarging_buffer and agent.actor.iter == 0:
            print("\nStarting to enlarge the buffer")
            agent.actor.enlarge_buffer = True
        
        trained = total_steps > args.max_train_steps or stable
        if args.POLICEd and args.enlarging_buffer:
            trained = trained and agent.actor.iter >= agent.actor.max_iter        
        
    args.total_steps = total_steps
    return stable



#%% Utils to verify constraint respect


def respect_ratio(args, env, agent, state_norm):
    """Empirical constraint verification
    Returns a percentage of constraint respect over a griding of the buffer."""
    nb_respect = 0
    
    nb_steps = 5
    buffer_max = env.buffer_vertices.max(axis=0)
    step = (buffer_max - env.buffer_min)/nb_steps
    s1_min, s2_min, s3_min = env.buffer_min
    s1_max, s2_max, s3_max = buffer_max
    s1_range = np.arange(start = s1_min, stop = s1_max + step[0]/2, step = step[0])
    s2_range = np.arange(start = s2_min, stop = s2_max + step[1]/2, step = step[1])
    s3_range = np.arange(start = s3_min, stop = s3_max + step[2]/2, step = step[2])
    
    nb_points = 0
    
    for s1 in s1_range:
        for s2 in s2_range:
            for s3 in s3_range:
                s = np.array([s1, s2, s3])
                
                if env.in_buffer(s_state = s):
                    env.s_reset_to(s)
                    with torch.no_grad():
                        action = agent.evaluate(s, state_norm)
                    _, _, _, repulsion_respect = env.step(action)
                    nb_points += 1
                    if repulsion_respect:
                        nb_respect += 1
           
    return nb_respect/nb_points




class data:
    """class to store the rewards over time and the percentage of constraint respect
    during training to plot it"""
    def __init__(self):
        self.len = 0
        self.reward_list = []
        self.respect_list = []
        
    def add(self, reward, respect):
        self.len += 1
        self.reward_list.append(reward)
        self.respect_list.append(respect)
        
    def add_respect(self, respect):
        self.respect_list.append(respect)
        
    def plot(self):
        iterations = np.arange(self.len)
        nice_plot()
        plt.title("Rewards")
        plt.plot(iterations, np.array(self.reward_list))
        plt.xlabel("Episodes")
        plt.show()
        

        
        


def rollout(env, agent, state_norm, initial_s_state=None, plot=False, title=""):
    """Rollouts a trajectory from the policy"""
    
    if initial_s_state is None:
        s_state = env.reset()
    else:
        s_state = env.s_reset_to(initial_s_state)
     
    s_Trajectory = np.zeros((env.max_episode_steps+1, env.state_size))
    x_Trajectory = np.zeros((env.max_episode_steps+1, env.state_size))
    s_Trajectory[0] = s_state
    x_Trajectory[0] = env.x_state
    Actions = np.zeros((env.max_episode_steps, env.action_size))
    
    for t in range(env.max_episode_steps):
        with torch.no_grad():
            action = agent.evaluate(s_state, state_norm)
        s_state, reward, done, _ = env.step(action)
        s_Trajectory[t+1] = s_state
        x_Trajectory[t+1] = env.x_state
        Actions[t] = action
        if done: break
    
    if plot:
        print(f"Vertical velocity at landing: {-s_state[1]:.1f} ft/s")
        plot_traj(env, x_Trajectory[:t+2], Actions[:t+1], title)
        phase_portrait_h(env, s_Trajectory[:t+2], title)
        
    return s_Trajectory[:t+2], x_Trajectory[:t+2], Actions[:t+1]

    
   
def multi_phase_portraits_h(env, s_Trajs):
    """Plots the phase portrait (h, h dot) """
        
    fig, ax = nice_plot()
    
    # Colors
    buffer_color = (0., 1., 0., 0.5)
    start_color = (0.2, 0.2, 0.2)
    traj_color = '#1f77b4'
    
    plt.title("Phase portrait")
    # Buffer 
    buffer_2D_vertices = [[-env.y_min, -env.s2_min],
                          [-env.y_max, -env.s2_min],
                          [-env.y_min, -env.y_dot_max]]
    buffer = plt.Polygon(xy=buffer_2D_vertices, color=buffer_color)
    ax.add_patch(buffer)
    
    # Trajectories
    N = len(s_Trajs)
    for i in range(N):
        plt.plot(-s_Trajs[i][:, 0], -s_Trajs[i][:, 1], color=traj_color, linewidth=3)
        plt.scatter(-s_Trajs[i][0, 0], -s_Trajs[i][0, 1], s=30, color=start_color, zorder=3)
    
    plt.xlabel("Altitude $h$ (ft)")
    plt.ylabel("Vertical speed $\dot h$ (ft/s)")
    
    start_marker = mlines.Line2D([], [], color=start_color, marker='o', linestyle='None', markersize=10, label='start')
    traj_marker   = mlines.Line2D([], [], color=traj_color, linewidth='3', markersize=10, label='trajectories')
    buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='^', linestyle='None', markersize=20, label='affine buffer $\mathcal{B}$')
    plt.legend(handles=[start_marker, traj_marker, buffer_marker], frameon=False,
                borderpad=.0, loc="upper right",
                labelspacing=0.3, handletextpad=0.5, handlelength=1.4)
    plt.show()


def multi_plot_trajs(env, x_Trajs, Actions=None):
    """Plots the trajectories of the space shuttle.
    x_state = [v gamma h]   action = [alpha] """
        
    N = len(x_Trajs)
    
    fig, ax = nice_plot()
    for i in range(N):
        time = env.dt * np.arange(x_Trajs[i].shape[0])
        plt.scatter(time, x_Trajs[i][:, 0], s=10)
    plt.ylabel("Velocity (ft/s)")
    plt.xlabel("time (s)")
    plt.show()
        
    fig, ax = nice_plot()
    for i in range(N):
        time = env.dt * np.arange(x_Trajs[i].shape[0])
        plt.scatter(time, x_Trajs[i][:, 1]*180/np.pi, s=10)
        plt.plot([time[0], time[-1]], [0., 0.], c="orange")
    plt.ylabel("Flight path angle (deg)")
    plt.xlabel("time (s)")
    plt.show()
        
    fig, ax = nice_plot()
    for i in range(N):
        time = env.dt * np.arange(x_Trajs[i].shape[0])
        plt.scatter(time, x_Trajs[i][:, 2], s=10)
    plt.ylabel("altitude h (ft)")
    plt.xlabel("time (s)")
    plt.show()
    
    if Actions is not None:
        fig, ax = nice_plot()
        for i in range(N):
            time = env.dt * np.arange(Actions[i].shape[0])
            Alpha = env.action_to_alpha(Actions[i])
            plt.scatter(time, Alpha, s=10)
        plt.ylabel("angle of attack alpha (deg)")
        plt.xlabel("time (s)")
        plt.show()






#%% Utils to save and load the whole framework


import json
import argparse
from PPO import PPO
from SpaceShuttle import ShuttleEnv
from normalization import Normalization


 

def save(filename, args, state_norm, agent):
    """Saving the arguments, state_norm and controller"""
    agent.save(filename)
    args_2 = copy.deepcopy(args)
    
    ### Convert arrays into lists  as arrays cannot be stored in json
    args_2.buffer_vertices = args.buffer_vertices.tolist()
    args_2.state_norm_n = state_norm.running_ms.n
    args_2.state_norm_mean = state_norm.running_ms.mean.tolist()
    args_2.state_norm_S = state_norm.running_ms.S.tolist()
    args_2.state_norm_std = state_norm.running_ms.std.tolist()
    
    with open(filename + '_args.txt', 'w') as f:
        json.dump(args_2.__dict__, f, indent=2)



def load(filename):
    """Loads the arguments, environment, state_norm and controller"""
    
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    args = parser.parse_args()
    with open(filename + '_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    
    ### Convert lists back to arrays  as arrays cannot be stored in json
    args.buffer_vertices = np.array(args.buffer_vertices)

    ### state normalization
    state_norm = Normalization(shape=args.state_dim)  
    state_norm.running_ms.n = args.state_norm_n
    state_norm.running_ms.mean = np.array(args.state_norm_mean)
    state_norm.running_ms.S = np.array(args.state_norm_S)
    state_norm.running_ms.std = np.array(args.state_norm_std)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = ShuttleEnv(args)
    
    agent = PPO(args)
    agent.load(filename)
    if args.POLICEd and args.enlarging_buffer: # sets buffer to fully enlarged
        agent.actor.iter = args.max_iter_enlargment+1
        agent.actor.vertices = agent.actor.buffer_vertices
        
    return args, state_norm, env, agent

