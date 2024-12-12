# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 9:15:08 2024

@author: Jean-Baptiste Bouvier

Inverted Pendulum main with POLICEd PPO for high relative degree constraint
Goal: learn to stabilize the pole and guarantee that it respects the constraint

PPO implementation from
https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous

"""

import torch
import argparse
import numpy as np

from PPO import PPO
from safe_invPendulum import Safe_InvertedPendulumEnv
from utils import ReplayBuffer, buffer_vertices, epsilon, plot_traj, data
from utils import repulsion_verification, training, constraint_training, rollout, save
from normalization import Normalization, RewardScaling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Hyperparameters


parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Gaussian") # or Beta")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=32, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
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

parser.add_argument("--constraint_C", type=np.array, default=np.array([0., 1., 0., 0.]), help="C matrix for the constraint  C @ state < d")
parser.add_argument("--constraint_d", type=float, default=0.2, help="d matrix for the constraint  C @ state < d")
parser.add_argument("--min_state", type=np.array, default=np.array([-0.9, 0.1, -1., 0.]), help="min value for the states in the buffer")
parser.add_argument("--max_state", type=np.array, default=np.array([ 0.9, 0.2,  1., 1.]), help="max value for the states in the buffer")

args = parser.parse_args()

args.enlarging_buffer = True #False
args.POLICEd = True # False

if args.POLICEd:
    filename = "saved/POLICEd"
else:
    filename = "saved/baseline"

### Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

### Environment
env = Safe_InvertedPendulumEnv(args, dense_reward=True)
# We need a second environment to evaluate the policy
env_eval = Safe_InvertedPendulumEnv(args, dense_reward=False)  

### Constraint and buffer
buffer_vertices = buffer_vertices(args)
args.buffer_vertices = buffer_vertices
args.eps = epsilon(args, env_eval)
print(f"Nonlinear approximation measure epsilon = {args.eps:.3f}")


### Add new constants to the environments
env.eps = args.eps
env_eval.eps = args.eps

### Add new constants to the arguments
args.state_dim = env.state_size
args.action_dim = env.action_size
args.max_action = env.action_max

agent = PPO(args)
replay_buffer = ReplayBuffer(args)

### state normalization
state_norm = Normalization(shape=args.state_dim)  
training_data = data()

assert args.use_reward_scaling
reward_scaling = RewardScaling(shape=1, gamma=args.gamma)



#%% Training
trained = False
stable = False

while (not trained) and (training_data.len < 5000):
    if not stable:
        stable = training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling, training_data)

    if args.POLICEd:
        stable, respect = constraint_training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling, training_data)
        trained = stable and respect
    else:
        trained = stable

# Saving the trained model
# save(filename, args, state_norm, agent)


#%% Visualisation 

env = Safe_InvertedPendulumEnv(args, dense_reward=False, render_mode="human")  
rollout(args, env, agent, state_norm, initial_state=None, render=True)
env.close()


  
#%% POLICEd verifications

env.render_mode = None

from utils import NetworkCopy, same_activation, local_affine_map

def norm(s):
    return torch.linalg.vector_norm(s).item()


if args.POLICEd:
    copied = NetworkCopy(agent.actor, state_norm)    
    same_activation(copied)
    C, d = local_affine_map(copied, copied.buffer_vertices[0]) # already normalized vertices
    eps = 1e-3
    
    ### Verification that the affine map is the same everywhere
    for vertex_id in range(buffer_vertices.shape[0]):
        s = torch.tensor(buffer_vertices[vertex_id]).float().unsqueeze(dim=0)
        normalized_s = state_norm(s, update=False) # normalized state
        with torch.no_grad():
            copy = copied(s)
            tru = agent.actor(s, state_norm, update=False)
        assert norm(copy - tru)/norm(tru) < eps or norm(copy - tru) < eps**2, "Network copy does not match policy" 
        ### Normalization is messing up the POLICEd verification
        calc = normalized_s @ C.T + d
        assert norm(calc - tru)/norm(tru) < eps or norm(calc - tru) < eps**2, "Affine map does not match policy"
    print("The original policy, the copy and the affine map all match on the buffer")
    

#%% Testing constraint_verification

repulsion_verification(args, env_eval, agent, state_norm, display=True)

#%%
from utils import repulsion_verification, training, constraint_training, rollout

y_dot_max = args.max_state[3]
initial_state = np.array([0., 0.1, 0., 0.9*y_dot_max])
rollout(args, env, agent, state_norm, initial_state)

initial_state = np.array([0., 0.1, 0., 0.99*y_dot_max])
rollout(args, env, agent, state_norm, initial_state)


#%%
from utils import rollout_comparisons

initial_states = np.array([[0., 0.1, 0., 0.3*y_dot_max],
                           [0., 0.1, 0., 0.6*y_dot_max],
                           [0., 0.1, 0., 0.9*y_dot_max],
                           [0., 0.1, 0., 1.1*y_dot_max],
                           [0., 0.1, 0., 1.2*y_dot_max],
                           [0., 0.1, 0., 1.3*y_dot_max]])

rollout_comparisons(args, env, agent, state_norm, initial_states)

#%%
from utils import recording

recording("test2", agent, state_norm, initial_state=np.array([0., 0.19, 0., 0.]), N_step=200)

env.close()