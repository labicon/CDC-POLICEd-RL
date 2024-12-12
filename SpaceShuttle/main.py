# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 9:15:08 2024

@author: Jean-Baptiste Bouvier

Space Shuttle Landing main with PPO for high relative degree constraint
Goal: learn to land the shuttle softly

Environment has 2 equivalent states:
x_state = [v gamma h]
s_state = [y  y' gamma] iterated output derivatives
           
PPO agent works with the s_state because the buffer is in s-representation

Space Shuttle system taken from
"Optimal Online Path Planning for Approach and Landing Guidance"
by Ali Heydari and S. N. Balakrishnan

PPO implementation from
https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous

"""

import copy
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PPO import PPO
from SpaceShuttle import ShuttleEnv
from utils import ReplayBuffer, data, modified_training, load, save, rollout, nice_plot
from normalization import Normalization, RewardScaling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Hyperparameters

loading = True
# warm_starting = False
# loading = False
warm_starting = True

parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
parser.add_argument("--max_train_steps", type=int, default=int(5e6), help="Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=1e5, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Gaussian") # or Beta")
parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2: state normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4: reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--seed", type=int, default=10, help="Common seed for all environments")

### POLICEd additions
parser.add_argument("--POLICEd", type=bool, default=True, help= "Whether the layers are POLICEd or linear")
parser.add_argument("--enlarging_buffer", type=bool, default=True, help= "Whether the buffer starts from a volume 0 and increases once policy has converged")
parser.add_argument("--max_iter_enlargment", type=int, default=int(1e4), help="Number of enlarging iterations for the buffer")

parser.add_argument("--y_min",     type=float, default=-50., help="start of the buffer in y")
parser.add_argument("--y_max",     type=float, default=0., help="end of the buffer in y, constraint line")
parser.add_argument("--y_dot_max", type=float, default=100., help="maximal output velocity allowed in buffer")
parser.add_argument("--s2_min",    type=float, default=6., help="minimal output velocity in the buffer")
### Gamma must be negative to have negative vertical velocity since v > 0
args = parser.parse_args()
args.freq_reset_in_buffer = 1000
args.nb_buffer_partitions = 1

# args.POLICEd = False

if args.POLICEd:
    filename = "saved/POLICEd_far"
    # filename = "saved/POLICEd_128"
    # filename = "saved/POLICEd_Buffer2"
else:
    filename = "saved/base"
    
### Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

### Environment
env = ShuttleEnv(args)
env_eval = ShuttleEnv(args)
print(f"epsilon = {env.eps:.1f}")

### Add new constants to the arguments
args.buffer_vertices = env.buffer_vertices
args.state_dim = env.state_size
args.action_dim = env.action_size
args.max_action = env.action_max

training_data = data()

assert args.use_reward_scaling
reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
reward_scaling.running_ms.n = 10000
reward_scaling.running_ms.mean = np.array([-100])


#%% Loading pretrained


if warm_starting:
    warm_start_filename = "saved/POLICEd_128" #"saved/base"
    loaded_args, loaded_state_norm, loaded_env, loaded_agent = load(warm_start_filename)
    _, _, _ = rollout(env, loaded_agent, loaded_state_norm, plot=True, title="Loaded")
    
    agent = PPO(args) # new agent to be warmstarted at the loaded model
    # Initialize the state normalization as in the loaded model
    state_norm = Normalization(shape=args.state_dim)
    state_norm = copy.deepcopy(loaded_state_norm)
    agent.load(warm_start_filename)
    _, _, _ = rollout(env, agent, state_norm, plot=True, title="Pretrained")

elif loading:
    _, state_norm, _, agent = load(filename)
    
    agent.actor.buffer_vertices = torch.tensor(env.buffer_vertices, dtype=torch.float32)
    agent.actor.buffer_center = torch.ones((agent.actor.num_vertices, 1)) @ agent.actor.buffer_vertices.mean(axis=0, keepdims=True)
    
    if agent.actor.iter >= agent.actor.max_iter: # buffer reached full size
        agent.actor.vertices = agent.actor.buffer_vertices.clone()
    elif agent.actor.iter == 0:
        agent.actor.vertices = agent.actor.buffer_center.clone()
    
    _, _, _ = rollout(env, agent, state_norm, plot=True, title="Loaded")
  
else: # cold start
    print("Cold start")
    agent = PPO(args) # new agent
    # Initialize the state normalization as in the loaded model
    state_norm = Normalization(shape=args.state_dim)
    s = env.s(env.x_0_min)
    s = state_norm(torch.tensor(s).unsqueeze(0), update=True)

    # Pretraining of the actor based on the loaded
    # Supervised learning of the POLICEd network based on the baseline
       
    import random 
    def partition (list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]
     
    warm_start_filename = "saved/base"
    loaded_args, loaded_state_norm, loaded_env, loaded_agent = load(warm_start_filename)
    
    opt = torch.optim.Adam(agent.actor.parameters(), lr=1e-5)
    state_norm = copy.deepcopy(loaded_state_norm)
    nb_test_trajs = 1000
    nb_repeats = 10
    mini_batch_len = 32
    Losses = []
    
    for _ in range(nb_test_trajs):
        
        s_Trajectory, x_Trajectory, Actions = rollout(env, loaded_agent, loaded_state_norm, plot=False)
        
        actions = torch.tensor(Actions)
        s_Traj = torch.tensor(s_Trajectory)
        N = Actions.shape[0] # number of training data points
        nb_mini_batches = N//mini_batch_len
        
        for repeats in range(nb_repeats):
            # creates random mini batches to split the trajectory
            batches = partition(list(range(N)), nb_mini_batches) 
            
            for id_batch in range(nb_mini_batches):
                batch = batches[id_batch]
                pred = agent.actor(s_Traj[batch], state_norm)
                loss = ((pred - actions[batch])**2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                Losses.append(loss.item())
    
    fig, ax = nice_plot()
    plt.title("Pretraining loss")
    plt.plot(np.arange(len(Losses)), Losses)   
    plt.show()
    
    
    _, _, _ = rollout(env, agent, state_norm, plot=True, title="Pretrained")

    
#%%
replay_buffer = ReplayBuffer(args)
trained = modified_training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling, training_data)


#%% Training with whole trajectories and reward given afterward for the whole trajectory

trained = False
replay_buffer = ReplayBuffer(args)

while not trained:
    
    trained = modified_training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling, training_data)
    if args.POLICEd:
        stable, respect = constraint_training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling, training_data)
        trained = stable and respect # POLICEd is only trained when achieves reward (stable) and respect repulsion condition


#%% Saving trained agent

# save("saved/POLICEd_128", args, state_norm, agent)
# save(filename, args, state_norm, agent)
# save("saved/baseline", args, state_norm, agent)






