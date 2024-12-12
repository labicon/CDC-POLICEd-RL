# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:52:25 2023

@author: Jean-Baptiste Bouvier

Function utils for the PPO POLICEd applied on the Inverted Pendulum environmnent.
"""

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#%% Utils to verify that the POLICEd network is affine in the buffer region

def activation_pattern(net, x):
    """Returns the activation pattern tensor of a NN net."""
    AP = torch.zeros((net.nb_layers-1, net.width))
    
    for layer_id in range(net.nb_layers-1):
        W, b = net.layer_W_b(layer_id)
        x = x @ W.T + b
        AP[layer_id,:] = (x > 0).to(int)
        x = torch.relu(x)
    return AP

def local_affine_map(net, x):
    """Returns tensors C and d such that the NN net
    is equal to C*x + d in the activation area of x."""
    prod = torch.eye((net.layer_0.in_features + 1)).to(device) # initial dim of weight + bias
    # weight + bias activation
    weight_AP = activation_pattern(net, x)
    bias_AP = torch.ones((net.nb_layers-1,1))
    AP = torch.cat((weight_AP, bias_AP), dim=1).to(device)
    
    for layer_id in range(net.nb_layers-1):
        W, b = net.layer_W_b(layer_id)
        row = torch.cat((torch.zeros((1,W.shape[1])), torch.tensor([[1.]])), dim=1).to(device)
        W_tilde = torch.cat( (torch.cat((W, b.unsqueeze(dim=1)), dim=1), row), dim=0)
        prod = torch.diag_embed(AP[layer_id]) @ W_tilde @ prod
        
    W, b = net.layer_W_b(net.nb_layers-1)
    W_tilde = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
    prod = W_tilde @ prod
    return prod[:,:-1], prod[:,-1]


class NetworkCopy(torch.nn.Module):
    """Creates a copy of the ConstrainedPolicyNetwork but without the extra-bias
    computation, which is simply included in the layer biases."""
    def __init__(self, policy, state_norm):
        super().__init__()
        self.nb_layers = policy.nb_layers
        self.width = policy.width
        self.state_norm = state_norm
        self.buffer_vertices = self.state_norm(policy.buffer_vertices.clone(), update=False)
        self.num_vertices = self.buffer_vertices.shape[0]
        self.setup(policy)
        
    def setup(self, policy):
        # Copy all layers from policy
        x = self.buffer_vertices
        for i in range(self.nb_layers):
            layer = getattr(policy, f"layer_{i}")
            W = layer.weight
            b = layer.bias
            setattr(self, f"layer_{i}", torch.nn.Linear(W.shape[1], W.shape[0]) )
            copied_layer = getattr(self, f"layer_{i}")
            copied_layer.weight.data = copy.deepcopy(W)
            copied_layer.bias.data = copy.deepcopy(b)
            
            # Add the extra-bias from policy
            h = x @ W.T + b
            
            agreement = h > 0
            invalid_ones = agreement.all(0).logical_not_().logical_and_(agreement.any(0))
            sign = agreement[:, invalid_ones].float().sum(0).sub_(self.num_vertices / 2 + 1e-3).sign_()
            extra_bias = (h[:, invalid_ones] * sign - 1e-4).amin(0).clamp(max=0) * sign
            copied_layer.bias.data[invalid_ones] -= extra_bias
            h[:, invalid_ones] -= extra_bias
            x = torch.relu(h)
        
    def forward(self, x):
        x = self.state_norm(x, update=False)
        for i in range(self.nb_layers-1):
            x = getattr(self, f"layer_{i}")(x)
            x = torch.relu(x)
        x = getattr(self, f"layer_{self.nb_layers-1}")(x)
        return x
    
    def layer_W_b(self, layer_id):
        """Returns weight and bias of given layer."""
        layer = getattr(self, f"layer_{layer_id}")
        return layer.weight.data, layer.bias.data
    


def same_activation(policy):
    """Verifies whether the whole CONVEX constraint area has the same activation pattern.
    Returns a boolean and a vertex where activation differs."""
    
    nb_vertices = policy.buffer_vertices.shape[0] # number of vertices of the constraint area
    ### Verify that all vertices of the constraint area have the same activation pattern
    prev_AP = activation_pattern(policy, policy.buffer_vertices[0])
    for vertex_id in range(1,nb_vertices):
        vertex = policy.buffer_vertices[vertex_id].unsqueeze(dim=0)
        # Vertices in copied network already normalized
        AP = activation_pattern(policy, vertex)
        if not (AP == prev_AP).all(): 
            print("The constraint region does not share the same activation pattern.")
            return False, vertex
        
    print("The constraint region shares the same activation pattern.")
    return True, vertex
     





#%% Utils for POLICEd RL theory: calculate minimal buffer radius and linearization constant

def buffer_vertices(args):
    """Returns the buffer vertices"""
    
    ### Unpack buffer bounds
    x_min, theta_min, x_dot_min, theta_dot_min = args.min_state
    x_max, theta_max, x_dot_max, theta_dot_max = args.max_state
    

    ### Buffer vertices calculation
    v = np.zeros((12, 4))
    i = 0
    for x in [x_min, x_max]:
        for theta in [theta_min, theta_max]:
            for x_dot in [x_dot_min, x_dot_max]:
                for theta_dot in [theta_dot_min, theta_dot_max]:
                    if theta == theta_max and theta_dot == theta_dot_max:
                        continue
                    v[i] = np.array([x, theta, x_dot, theta_dot])
                    i += 1        
    return v




### Grid search
def epsilon(args, env):
    """Calculates the epsilon difference for the dynamics linearization.
    A lot of specifics to the pendulum dynamics and constraint.
    """
    
    nb_steps = 10
    step = (args.max_state - args.min_state)/nb_steps
    x_min, theta_min, x_dot_min, theta_dot_min = args.min_state
    x_max, theta_max, x_dot_max, _             = args.max_state
    x_range      = np.arange(start = x_min,         stop = x_max         + step[0]/2, step = step[0])
    theta_range  = np.arange(start = theta_min,     stop = theta_max     + step[1]/2, step = step[1])
    x_dot_range  = np.arange(start = x_dot_min,     stop = x_dot_max     + step[2]/2, step = step[2])
    action_range = np.arange(start = -env.action_max, stop = env.action_max + 0.01, step = 2*env.action_max/nb_steps)
    N = (nb_steps+1)**5
    
    ### Storing datapoints to perform linear regression on
    States = np.zeros((N, env.state_size))
    Next_States = np.zeros((N, env.state_size))
    Actions = np.zeros((N, env.action_size))
    
    i = 0
    for x in x_range:
        for theta in theta_range:
            theta_dot_max = env.offset + theta*env.slope
            step[3] = (theta_dot_max - theta_dot_min)/nb_steps
            if step[3] == 0.0:
                theta_dot_range = [theta_dot_max]
            else:
                theta_dot_range = np.arange(start = theta_dot_min, stop = theta_dot_max + step[3]/2, step = step[3])
            for x_dot in x_dot_range:
                for theta_dot in theta_dot_range:
                    state = np.array([x, theta, x_dot, theta_dot])
                    for action in action_range:
                        
                        States[i] = state
                        Actions[i] = action
                        env.reset_to(States[i])
                        Next_States[i], _, _, _ = env.step(Actions[i])
                        i += 1
                        
    #### Least square fit
    Ones = np.ones((N, 1))
    A = np.concatenate((States, Actions, Ones), axis=1)
    Theta_ddot = (Next_States[:,-1] - States[:,-1])/env.dt
    x, _, _, _ = np.linalg.lstsq(A, Theta_ddot, rcond=None)
    
    ### Calculation of epsilon
    CA = np.array([[x[0], x[1], x[2], x[3]]])
    CB = x[4]
    Cc = x[5]
    affine_pred = States @ CA.T + Actions * CB.T + Cc
    eps = abs(Theta_ddot - affine_pred.squeeze()).max()
    
    return eps






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
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


def evaluate_policy(args, env, agent, state_norm):
    """Evaluates the policy, returns the average reward and whether the
    reward threshold has been met"""
    times = 1#3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(s, state_norm)  # We use the deterministic policy during the evaluating
            s, r, done, _ = env.step(action)
            episode_reward += r
        evaluate_reward += episode_reward

    average_reward = evaluate_reward / times
    return average_reward, average_reward >= env.reward_threshold


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


def phase_portrait_theta(env, Traj):
    """Plots the phase portrait of theta in the trajectory Traj"""
        
    fig, ax = nice_plot()
    
    # Colors
    buffer_color = (0., 1., 0., 0.5)
    start_color = (0.2, 0.2, 0.2)
    end_color = (1., 0., 0.)
    
    plt.title("Phase portrait")
    # Buffer 
    buffer_2D_vertices = [[env.buffer_min[1], env.buffer_min[3]],
                          [env.buffer_max[1], env.buffer_min[3]],
                          [env.buffer_min[1], env.buffer_max[3]]]
    buffer = plt.Polygon(xy=buffer_2D_vertices, color=buffer_color)
    ax.add_patch(buffer)
    
    # Trajectory
    # plt.scatter(Traj[1:-1, 1], Traj[1:-1, 3], s=30)
    # plt.plot(Traj[:, 1], Traj[:, 3], linewidth=3)
    smoothed = Traj[:, [1,3]] # smoothing does NOT work
    plt.plot(smoothed[:, 0], smoothed[:, 1], linewidth=3)
    
    plt.scatter(Traj[ 0, 1], Traj[ 0, 3], s=60, color=start_color, zorder=3)
    plt.scatter(Traj[-1, 1], Traj[-1, 3], s=60, color=end_color, zorder=3)
    
    plt.xlabel("Pole angle $\\theta$ (rad)")
    plt.ylabel("Pole speed $\dot \\theta$ (rad/s)")
    
    start_marker = mlines.Line2D([], [], color=start_color, marker='o', linestyle='None', markersize=10, label='start')
    end_marker   = mlines.Line2D([], [], color=end_color,   marker='o', linestyle='None', markersize=10, label='end')
    buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='^', linestyle='None', markersize=10, label='affine buffer $\mathcal{B}$')
    plt.legend(handles=[start_marker, end_marker, buffer_marker], frameon=False,
               borderpad=.0,
               labelspacing=0.3, handletextpad=0.5, handlelength=1.4)
    plt.show()
    


def plot_traj(env, Traj, N_step=None, states_to_plot=[0]):
    """Plots the trajectory of the states of the inverted pendulum.
    states_to_plot contains the indices of the states to plot:
        [0: x, 1: theta, 2: x dot, 3: theta dot]    """
    if N_step is None:
        N_step = env.max_episode_steps
        
    N = Traj.shape[0] # number of steps of the trajectory before terminating
    
    titles = ["Cartpole position (m)", "Pole angle (rad)", "Cartpole speed (m/s)", "Pole speed (rad/s)"]
    variables = ["x", "theta", "x dot", "theta dot"]

    for state_id in states_to_plot:
        
        fig, ax = nice_plot()
        plt.title(titles[state_id])
        plt.scatter(np.arange(N), Traj[:, state_id], s=10, label=variables[state_id])
        
        if env.max_state[state_id] < 1e5: # don't display infinite limit
            plt.plot(np.array([0, N_step]),  np.ones(2)*env.max_state[state_id], c='red')
            plt.plot(np.array([0, N_step]), -np.ones(2)*env.max_state[state_id], c='red', label="limit")
        
        plt.legend()
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
            s = env.reset_to(coefs @ buffer_vertices)
        else:
            s = env.reset()
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s, state_norm)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)

            if args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be True, we need to distinguish them;
            # dw means dead or win, there is no next state s'
            # but when reaching the max_episode_steps, there is a next state s' actually.
            if done and episode_steps != env.max_episode_steps:
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
            
            if stable and args.POLICEd and args.enlarging_buffer and agent.actor.iter == 0:
                print("\nStarting to enlarge the buffer")
                agent.actor.enlarge_buffer = True
            
            trained = total_steps > args.max_train_steps or stable
            if args.POLICEd and args.enlarging_buffer:
                trained = trained and agent.actor.iter > agent.actor.max_iter
           
        r = respect_ratio(args, env_eval, agent, state_norm)
        data.add(episode_steps, r)
        
    args.total_steps = total_steps
    return stable






#%% Utils to verify and train for constraint respect

def repulsion_verification(args, env, agent, state_norm, display=True):
    """Verifies whether the repulsion condition holds by checking each vertices of the buffer.
    Returns a Boolean of respect and if violation returns the id of the violating vertex"""
    
    nb_vertices = args.buffer_vertices.shape[0] # number of vertices of the constraint area
    
    ### Verify that the constraint holds at every vertex of the CONVEX constraint area
    for vertex_id in range(nb_vertices):
        vertex = args.buffer_vertices[vertex_id]
        env.reset_to(vertex)
            
        with torch.no_grad():
            action = agent.evaluate(vertex, state_norm)
        _, _, _, repulsion_respect = env.step(action)
        if not repulsion_respect:
            if display:
                print(f"Constraint is not satisfied on vertex {vertex_id}")
            return False, vertex_id
    if display:
        print("Constraint is satisfied everywhere")
    return True, vertex_id


def constraint_training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling=None, data=None):
    """Constraint training of the PPO agent."""
    
    respect = False
    buffer_vertices = args.buffer_vertices
    num_vertices = buffer_vertices.shape[0]
    id_bad_vertex = 0
    if hasattr(args, 'total_steps'):
        total_steps = args.total_steps
    else:
        total_steps = 0
    repeats = 0
    while not respect and repeats < 4:
        repeats += 1
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        
        ### Reset on each vertex of the buffer and take one step
        for vertex_id in range(id_bad_vertex, num_vertices):
            total_steps += 1
            s = env.reset_to(buffer_vertices[vertex_id])
            
            a, a_logprob = agent.choose_action(s, state_norm)  # Action and the corresponding log probability
            s_, r, done, repulsion_respect = env.step(a)
            
            if not repulsion_respect: # vertices where constraint is not respected
                value, tol = env.repulsion(s)
                
                r = 5*(tol - value) -1.
                if args.use_reward_scaling:
                    r = reward_scaling(r)
                # print(f"Vertex {vertex_id},\t v = {value:.3f} > tol = {args.tol:.3f}")
                dw = done # Since there is only one step died/win is equal to done
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                if replay_buffer.count == args.batch_size:
                    replay_buffer.count = 0
        
        if replay_buffer.count > 1: # don't update with empty buffer
            agent.update(replay_buffer, total_steps, state_norm)
        respect, id_bad_vertex = repulsion_verification(args, env_eval, agent, state_norm, display=True)
    
    evaluate_reward, stable = evaluate_policy(args, env_eval, agent, state_norm)
    r = respect_ratio(args, env_eval, agent, state_norm)
    data.add(evaluate_reward, r)
    print(f"Reward after constraint training: {evaluate_reward:.2f}")
    args.total_steps = total_steps
    return stable, respect



def respect_ratio(args, env, agent, state_norm):
    """Empirical constraint verification
    Returns a percentage of constraint respect over a griding of the buffer."""
    nb_respect = 0
    
    nb_steps = 4
    step = (args.max_state - args.min_state)/nb_steps
    x_min, theta_min, x_dot_min, theta_dot_min = args.min_state
    x_max, theta_max, x_dot_max, _             = args.max_state
    x_range         = np.arange(start = x_min,         stop = x_max         + step[0]/2, step = step[0])
    theta_range     = np.arange(start = theta_min,     stop = theta_max     + step[1]/2, step = step[1])
    x_dot_range     = np.arange(start = x_dot_min,     stop = x_dot_max     + step[2]/2, step = step[2])
    nb_points = 0
    
    for x in x_range:
        for theta in theta_range:
            theta_dot_max = env.offset + theta*env.slope
            step[3] = (theta_dot_max - theta_dot_min)/nb_steps
            if step[3] == 0.0:
                theta_dot_range = [theta_dot_max]
            else:
                theta_dot_range = np.arange(start = theta_dot_min, stop = theta_dot_max + step[3]/2, step = step[3])
            for x_dot in x_dot_range:
                for theta_dot in theta_dot_range:
                    state = np.array([x, theta, x_dot, theta_dot])
                    env.reset_to(state)
                        
                    with torch.no_grad():
                        action = agent.evaluate(state, state_norm)
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
        nice_plot()
        
        iterations = np.arange(self.len)
        plt.title("Rewards")
        plt.plot(iterations, np.array(self.reward_list))
        plt.xlabel("Episodes")
        plt.show()
        
        iterations = np.arange(len(self.respect_list))
        plt.title("Respect during training")
        plt.plot(iterations, np.array(self.respect_list))
        plt.xlabel("Episodes")
        plt.show()
        
        


def rollout(args, env, agent, state_norm, initial_state=None, render=False):
    """Rollouts a trajectory from the policy"""
    
    if initial_state is None:
        state = env.reset()
    else:
        state = env.reset_to(initial_state)
     
    episode_reward = 0
    Trajectory = np.zeros((env.max_episode_steps+1, env.state_size))
    Trajectory[0] = state
    Actions = np.zeros((env.max_episode_steps, env.action_size))
    
    for t in range(env.max_episode_steps):
        with torch.no_grad():
            action = agent.evaluate(state, state_norm)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        Trajectory[t+1] = state
        Actions[t] = action
        
        if render: env.render()
        if done: break
    
    # print(f"Average reward: {episode_reward/env.max_episode_steps:.3f}")
    plot_traj(env, Trajectory[:t+2], env.max_episode_steps)
    # print(f"Min max actions on visual traj:  {Actions.min().item():.2f},  {Actions.max().item():.2f}")
    phase_portrait_theta(env, Trajectory[:t+2])
    
    
    
def rollout_comparisons(args, env, agent, state_norm, initial_states):
    """Compares the rollout trajectories from several initial states"""
    
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
    
   
    fig, ax = nice_plot()
    
    buffer_color = (0., 1., 0., 0.5)
    start_color = (0.2, 0.2, 0.2)
    constraint_color = (1., 0., 0., 0.5)
    traj_color = '#1f77b4'
    
    plt.title("Phase portrait")
    buffer_2D_vertices = [[env.buffer_min[1], env.buffer_min[3]],
                          [env.buffer_max[1], env.buffer_min[3]],
                          [env.buffer_min[1], env.buffer_max[3]]]
    buffer = plt.Polygon(xy=buffer_2D_vertices, color=buffer_color)
    ax.add_patch(buffer)
    plt.xlabel("Pole angle $\\theta$ (rad)")
    plt.ylabel("Pole speed $\dot \\theta$ (rad/s)")
    
    max_theta_dot = 0.
    min_theta_dot = 1.
    for i in range(N):
        traj = Trajs[i][:, [1,3]] # un modified traj
        plt.plot(traj[:, 0], traj[:, 1], color=traj_color, linewidth=3)
        plt.scatter(traj[0, 0], traj[0, 1], s=30, color=start_color, zorder=3)
        for t in range(traj.shape[0]):
            theta_dot = traj[t,1]
            if theta_dot < min_theta_dot:
                min_theta_dot = theta_dot
            elif theta_dot > max_theta_dot:
                max_theta_dot = theta_dot
                
    plt.plot(np.array([env.buffer_max[1], env.buffer_max[1]]), np.array([min_theta_dot, max_theta_dot]), color=constraint_color, linewidth=4)  
    
    start_marker = mlines.Line2D([], [], color=start_color, marker='o', linestyle='None', markersize=10, label='start')
    buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='^', linestyle='None', markersize=10, label='affine buffer $\mathcal{B}$')
    constraint_marker = mlines.Line2D([], [], color=constraint_color, linewidth='4', markersize=10, label='constraint line')
    
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([0., 0.1, 0.2])
    plt.xlim([-0.05, 0.21])
    plt.ylim([-1.1*max_theta_dot, 1.1*max_theta_dot])
    
    plt.legend(handles=[start_marker, buffer_marker, constraint_marker], frameon=False,
               borderpad=.0, labelspacing=0.3, handletextpad=0.5, handlelength=1.4,
               loc='upper left')
    plt.show()
    

#%% Utils to save and load the whole framework


import json
import argparse
from PPO import PPO
from safe_invPendulum import Safe_InvertedPendulumEnv
from normalization import Normalization


 

def save(filename, args, state_norm, agent):
    """Saving the arguments, state_norm and controller"""
    agent.save(filename)
    args_2 = copy.deepcopy(args)
    
    ### Convert arrays into lists  as arrays cannot be stored in json
    args_2.constraint_C = args.constraint_C.tolist()
    args_2.min_state = args.min_state.tolist()
    args_2.max_state = args.max_state.tolist()
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
    args.constraint_C = np.array(args.constraint_C)
    args.min_state = np.array(args.min_state)
    args.max_state = np.array(args.max_state)
    args.buffer_vertices = np.array(args.buffer_vertices)

    ### state normalization
    state_norm = Normalization(shape=args.state_dim)  
    state_norm.running_ms.n = args.state_norm_n
    state_norm.running_ms.mean = np.array(args.state_norm_mean)
    state_norm.running_ms.S = np.array(args.state_norm_S)
    state_norm.running_ms.std = np.array(args.state_norm_std)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = Safe_InvertedPendulumEnv(args, dense_reward=False)
    
    agent = PPO(args)
    agent.load(filename)
    if args.POLICEd and args.enlarging_buffer: # sets buffer to fully enlarged
        agent.actor.iter = args.max_iter_enlargment
        agent.actor.vertices = agent.actor.buffer_vertices
        
    return args, state_norm, env, agent




#%%  Utils for training_plots 

def save_data(filename, rewards, respects):
    """Saving the multi-run data of rewards and respects"""
    
    with open(filename + '_rewards.txt', 'w') as f:
        json.dump(rewards, f, indent=2)
        
    with open(filename + '_respects.txt', 'w') as f:
        json.dump(respects, f, indent=2)
        
        

def load_data(filename):
    """Loads the multi-run data of rewards and respects"""
    
    rewards = [[]]
    with open(filename + '_rewards.txt', 'r') as f:
        rewards = json.load(f)
    
    respects = [[]]
    with open(filename + '_respects.txt', 'r') as f:
        respects = json.load(f)
        
    return rewards, respects






#%% Make video

import warnings
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def recording(video_name, agent, state_norm, initial_state=None, N_step=200):
    """Records a video of the inverted pendulum"""
    
    with warnings.catch_warnings(action="ignore"):
        env = gym.make('InvertedPendulum-v4', render_mode="rgb_array")
        path = "C:\\Users\\jeanb\\Videos\\Inverted Pendulum"
        recording_env = RecordVideo(env, video_folder=path, name_prefix=video_name)
        
        if initial_state is None:
            state, _ = recording_env.reset()
        else:
            qpos = initial_state[:2] # get desired positions (x, theta)
            qvel = initial_state[2:] # get desired velocities (x_dot, theta_dot)
            recording_env.reset()
            # Go through all the wrappers to set the initial state in Mujoco
            recording_env.env.env.env.env.set_state(qpos, qvel)
            state = recording_env.env.env.env.env._get_obs()
        
        for t in range(N_step):
            with torch.no_grad():
                action = agent.evaluate(state, state_norm)
            state, _, done, _, _ = recording_env.step(action)
            recording_env.render()
            if done: break
         
        recording_env.close()



def compare_trajs(env, b_Traj, P_Traj, title=""):
    """Plots together a baseline and a POLICEd trajectories."""

    N_step = env.max_episode_steps
    N_b = b_Traj.shape[0] # number of steps of the trajectory before terminating
    N_P = P_Traj.shape[0]
    
    titles = ["Cartpole position (m)", "Pole angle (rad)"]
    variables = ["x", "theta"]

    for state_id in range(2):
        
        plt.title(title + " " + titles[state_id])
        plt.scatter(np.arange(N_b), b_Traj[:, state_id], s=10, label=variables[state_id] + " baseline")
        plt.scatter(np.arange(N_P), P_Traj[:, state_id], s=10, label=variables[state_id] + " POLICEd")
        
        if env.max_state[state_id] < 1e5: # don't display infinite limit
            plt.plot(np.array([0, N_step]),  np.ones(2)*env.max_state[state_id], c='red')
            plt.plot(np.array([0, N_step]), -np.ones(2)*env.max_state[state_id], c='red', label="limit")
        
        plt.legend()
        plt.show()


def propagate(args, env, agent, state_norm, initial_x_state=None, traj_len=None):
    """Propagate a trajectory from the initial state with the agent."""
    
    if initial_x_state is None:
        state = env.reset()
    else:
        state = env.reset_to(initial_x_state)
     
    if traj_len is None:
        traj_len = env.max_episode_steps
     
    episode_reward = 0
    Trajectory = np.zeros((env.max_episode_steps, env.state_size))
    Trajectory[0] = state
    
    for t in range(1, traj_len):
        with torch.no_grad():
            action = agent.evaluate(state, state_norm)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        Trajectory[t] = state
        if done: break
    
    return Trajectory[:t+1], episode_reward
