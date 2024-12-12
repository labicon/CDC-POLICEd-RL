# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:25:36 2023

@author: Jean-Baptiste Bouvier

Wrapper for the Mujoco Inverted Pendulum Environment
Add safety features with the constraint on the pole and repulsive buffer.
"""

import copy
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec


class Safe_InvertedPendulumEnv():
    """
    Wrapper to make a Safe version of the Mujoco Inverted Pendulum

    | Num | Action                    | Min | Max | Joint | Unit      |
    |-----|---------------------------|-----|-----|-------|-----------|
    | 0   | Force applied on the cart | -3  | 3   | slide | Force (N) |


    | Num | Observation                                   | Min  | Max | Joint | Unit                      |
    | --- | --------------------------------------------- | ---- | --- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge | anglular velocity (rad/s) |

    if dense_reward == False: reward = +1 when pole is in admissible range, otherwise 0 and done
    if dense_reward == True:  penalty on theta, x, inadmissible actions
    """

    def __init__(self, args, dense_reward=True, render_mode=None):
        
        self.max_episode_steps = 1000 # but use 200 for faster iterations
        self.reward_threshold = 0.95*self.max_episode_steps
        
        # Specify the specs to remove all env wrappers to access the Mujoco env directly
        self.spec = EnvSpec(id = 'InvertedPendulum-v4',   # The string used to create the environment with :meth:`gymnasium.make`
                      entry_point='gymnasium.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv', # A string for the environment location
                      reward_threshold=None, # The reward threshold for completing the environment.
                      nondeterministic=False,      # If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
                      max_episode_steps=None, # None to remove wrapper
                      order_enforce=False,    # False to remove wrapper
                      autoreset=False,             # If to automatically reset the environment on episode end
                      disable_env_checker=True,    # If to disable the environment checker wrapper in :meth:`gymnasium.make`, by default False (runs the environment checker)
                      kwargs={'render_mode': render_mode}, # Additional keyword arguments passed to the environment during initialisation
                      additional_wrappers=(),      #  A tuple of additional wrappers applied to the environment (WrapperSpec)
                      vector_entry_point=None)     # The location of the vectorized environment to create from
               
        self.env = gym.make(self.spec)
        self._seed = args.seed
        self.env.action_space.seed(self._seed)
        
        self.env.reset(seed = args.seed)
        self.state_size = 4
        self.action_size = 1
        
        self.dt = 0.02 # timestep from the XML data sheet of the inverted pendulum
        self.reset_bound = 0.01 # uniform noise on the reset
        self.theta_threshold = 0.2 # [radians] max pole angle before termination
       
        self.action_max = 1. # 3 original, but reduced to 1 to have smaller buffer radius
        self.max_state = np.array([1., 0.2, np.inf, np.inf])
        
        self.dense_reward = dense_reward
        self.C = args.constraint_C # C matrix of the constraint
        self.d = args.constraint_d # d matrix of the constraint: C*s < d
        self.eps = 0.1  # constraint tolerance
        self.buffer_min = args.min_state
        self.buffer_max = args.max_state
        self.beta = self.buffer_max[3]/(self.buffer_max[1] - self.buffer_min[1]) # y_dot_max/(y_max - y_min)
        ### slope and offset for buffer upper limit of    theta_dot = f(theta)
        self.slope = (self.buffer_min[3] - self.buffer_max[3])/(self.buffer_max[1] - self.buffer_min[1])
        self.offset = self.buffer_min[3] - self.slope*self.buffer_max[1]
    
    def repulsion(self, previous_state):
        """Calculates the actuated state and its desired maximal value for repulsion"""
        theta_dot = self.state[3]
        theta_ddot = (theta_dot - previous_state[3])/self.dt
        tol = -2*self.eps - theta_dot * self.beta
        return theta_ddot, tol

    def repulsion_respect(self, previous_state):
        """Verifies whether the repulsion condition holds."""
        theta_ddot, tol = self.repulsion(previous_state)
        return theta_ddot <= tol

    def step(self, action):
        self.episode_step += 1
        previous_state = self.state
        self.state, _, done, _, _ = self.env.step(action.clip(-self.action_max, self.action_max))
        if self.episode_step >= self.max_episode_steps:
            done = True
        theta = self.state[1]
        
        if self.dense_reward:
            reward = 1 - 4*abs(theta)
            reward -= self.state[0]**2 # keep x close to 0
            reward -= 10*np.maximum(abs(action) - self.action_max, 0.) # keep action in admissible range                
        else:
            reward = float(abs(theta) <= self.theta_threshold)
        
        # constraint_respect = self.C @ self.state <= self.d
        if self.in_buffer(previous_state): #self.state):
            theta_ddot, tol = self.repulsion(previous_state)
            repulsion_respect = theta_ddot <= tol
            if self.dense_reward:
                reward += 5*min(tol - theta_ddot, 0.)
        else:
            repulsion_respect = False
         
        return copy.deepcopy(self.state), reward, done, repulsion_respect


    def reset(self):
        self.state, _ = self.env.reset(seed = self._seed)
        self.episode_step = 0
        return copy.deepcopy(self.state)

    def reset_to(self, state):
        """Resets the state of the environment to specified state"""
        qpos = state[:2] # get desired positions (x, theta)
        qvel = state[2:] # get desired velocities (x_dot, theta_dot)
        self.env.reset(seed = self._seed)
        self.env.set_state(qpos, qvel) # Mujoco method to set state
        ### Sanity check
        self.state = self.env._get_obs()
        assert np.linalg.norm(self.state - state) < 1e-5
        self.episode_step = 0
        return copy.deepcopy(self.state)

    def in_buffer(self, state):
        """Verifies whether the state is in the buffer."""
        # Buffer is hyperrectangle in x, theta, x_dot, and is `triangular' in theta_dot
        if (state[:3] >= self.buffer_min[:3]).all() and (state[:3] <= self.buffer_max[:3]).all():
            theta = state[1]
            theta_dot = state[3]
            return theta_dot >= 0 and theta_dot <= self.slope*theta + self.offset
        return False

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
        

