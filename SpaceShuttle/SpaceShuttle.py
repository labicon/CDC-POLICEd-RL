# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 10:25:36 2024

@author: Jean-Baptiste Bouvier

Space Shuttle system taken from
"Optimal Online Path Planning for Approach and Landing Guidance"
by Ali Heydari and S. N. Balakrishnan
"""

import copy
import torch
import numpy as np


class ShuttleEnv():
    """
    Space Shuttle environment 
    1 action 3 states available in 2 state representations (s_ and x_)
    Relative degree 2

    | Num | Name  | Action          |  Min  |  Max | Unit|
    |-----|-------|-----------------|-------|------|-----|
    |  0  |   u   | control         |  -1   |   1  |     |
    
    alpha = 30 * ( u + 0.8 )
    |  0  | alpha | angle of attack |  0  |  50  | deg |

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

    def __init__(self, args):
        
        self.max_episode_steps = 500
        self.reward_threshold = -args.s2_min # [ft/s] for vertical velocity at landing
        self.relative_degree = 2
        self.state_size = 3
        self.action_size = 1
        
        self.dt = 0.1         # [s] timestep
        self.action_max = 1   # []  (control input)
        
        self.CD0 = 0.0975    # [] zero-lift drag coefficient
        self.CL0 = 2.3       # [] zero-angle-of-attack lift coefficient
        self.Ki = 0.1819     # [] lift-induced drag coefficient parameter

        ###### Imperial system
        self.Sm = 0.912     # [ft^2/slug] shuttle aero area / mass
        self.H = 8.5e3*3.28 # [ft] altitude scale for the atmospheric pressure
        self.g = 32.174     # [ft/s^2] Earth’s gravitational acceleration
        self.rho_0 = 0.0027 # [slugs/ft^3] sea-level air density
        
        # Transformation between input u in [-1, 1] and angle of attack alpha
        self.u_to_alpha_slope = 30.
        self.u_to_alpha_offset = 0.8
        
        self.initial_range = False # True
        
        if self.initial_range: # Initial state belongs to a range
            self.v_0_min = 370   # [ft/s]
            self.v_0_max = 400 # [ft/s] 
            self.gamma_0_min = -10*np.pi/180 # [rad] 
            self.gamma_0_max = -30*np.pi/180 # [rad]
            self.h_0_min = 500 # [ft]
            self.h_0_max = 500 # [ft]
            self.x_0_min = np.array([self.v_0_min, self.gamma_0_min, self.h_0_min])
            self.x_0_max = np.array([self.v_0_max, self.gamma_0_max, self.h_0_max])
            
        else: # Initial state around a reference state
            self.v_0_min = 300   # [ft/s] initial velocity
            self.gamma_0 = -30*np.pi/180 # [rad] initial flight path angle
            self.h_0 = 10000     # [ft] initial altitude
            self.x_0_min = np.array([self.v_0_min, self.gamma_0, self.h_0])
            self.x_reset_bound = 0.001 # uniform noise on the reset
        
        
        ### Buffer parameters
        self.y_min = args.y_min
        self.y_max = args.y_max
        self.y_dot_max = args.y_dot_max
        self.s2_min = args.s2_min
        # self.s3_min = args.s3_min   # gamma [rad]
        # self.s3_max = args.s3_max   # gamma [rad]
        self.beta = (self.y_dot_max - self.s2_min)/(self.y_max - self.y_min)
        self.beta_offset = self.y_dot_max + self.beta * self.y_min
        
        ### Sanity checks
        assert abs(self.y_dot_max - (self.beta_offset - self.beta * self.y_min)) < 1e-8
        assert abs(self.s2_min -    (self.beta_offset - self.beta * self.y_max)) < 1e-8
        
        self.nb_buffer_partitions = args.nb_buffer_partitions
        self._buffer_vertices()
        # For handmade buffer this does not work
        self.s3_min = self.buffer_vertices[:, 2].min() # gamma [rad]
        self.s3_max = self.buffer_vertices[:, 2].max() # gamma [rad]
        self.buffer_min = np.array([self.y_min, self.s2_min, self.s3_min])
        
        # self._epsilon()  # [] affine approximation linear
        self._epsilon_partition()
        if self.nb_buffer_partitions == 1:
            self.all_buffer_vertices = self.buffer_vertices
        else:
            self.all_buffer_vertices = np.concatenate((self.buffer_vertices, self.extra_vertices), axis=0)
        
    def action_to_alpha(self, action):
        """Calculates angle of attack alpha from action  u in [-1, 1]"""
        u = action.clip(-self.action_max, self.action_max)
        alpha = self.u_to_alpha_slope*(u + self.u_to_alpha_offset)
        return alpha
    
    
    def alpha_to_action(self, alpha):
        """Calculates action u in [-1, 1] from angle of attack alpha"""
        return alpha/self.u_to_alpha_slope - self.u_to_alpha_offset
        
    
    def s(self, x_state):
        """Transforms the x_state into the s_state"""
        v, gamma, h = x_state
        s_state = np.array([-h, -v*np.sin(gamma), gamma])
        return s_state
        
    
    def x(self, s_state):
        """Transforms the s_state into the x_state"""
        y, y_dot, gamma = s_state
        x_state = np.array([-y_dot/np.sin(gamma), gamma, -y])
        return x_state


    def out_of_bounds(self):
        """Verifies whether the x_state is out of bounds"""
        v, gamma, h = self.x_state
        return abs(gamma) > np.pi/2 or abs(v) > 1000
        


    def step(self, action):
        """step for given action.
        Extra input: if alpha_in = True, then action = alpha [deg]"""
        self.episode_step += 1
        previous_s_state = self.s_state
        alpha_deg = self.action_to_alpha(action)[0]
        
        v, gamma, h = self.x_state
        
        rho = self.rho_0 * np.exp(-h/self.H) # air density
        alpha_rad = alpha_deg * np.pi/180
        CL = self.CL0 * np.sin(alpha_rad)**2 * np.cos(alpha_rad) # lift coefficient
        CD = self.CD0 + self.Ki * CL**2      # drag coefficient
        
        dv = -0.5 * rho * v**2 * self.Sm * CD - self.g * np.sin(gamma)
        dgamma = 0.5 * rho * v * self.Sm * CL - self.g * np.cos(gamma)/v
        dh = v * np.sin(gamma)
        self.x_state += self.dt * np.array([dv, dgamma, dh])
        self.s_state = self.s(self.x_state)
    
        v, gamma, h = self.x_state # updated state
        reward = -10*np.maximum(abs(action) - self.action_max, 0.)[0] # keep action in admissible range                
        out = self.out_of_bounds()
        reward -= out*50000 # penalize if state grows out of bounds
        reward -= 0.2*abs(alpha_deg - self.previous_alpha_deg) # penalize chattering
        
        if self.in_buffer(s_state = previous_s_state):
            repulsion_respect = self.repulsion_respect(previous_s_state)
        else:
            repulsion_respect = False # True
        
        done = self.s_state[0] > self.y_max or out or self.in_landing_box()
        if self.episode_step >= self.max_episode_steps:
            done = True
        
        self.previous_alpha_deg = alpha_deg
        return copy.deepcopy(self.s_state), reward, done, repulsion_respect
    
    
    # if small altitude and small vertical velocity, we are good
    def in_landing_box(self):
        """Verifies whether the state is in the landing box, i.e., below the buffer"""
        return self.s_state[0] > self.y_min and self.s_state[1] < self.s2_min

    def final_reward(self):
        """Calculates the reward at the last time step.
        Penalize altitude h = s[0] and penalize final vertical velocity h_dot = s[1] 
         """
        return -abs(self.s_state[0]) - abs(self.s_state[1])


    def reset(self):
        """Resets the state of the environment around a base x_state"""
        if self.initial_range:
            self.x_state = self.x_0_min + (self.x_0_max - self.x_0_min) * np.random.rand(self.state_size)
        else:
            self.x_state = self.x_0_min*(1 + self.x_reset_bound*np.random.rand(self.state_size))
        
        self.previous_alpha_deg = 0 # [deg] value of previous alpha to discourage chattering
        self.episode_step = 0
        self.s_state = self.s(self.x_state)
        return copy.deepcopy(self.s_state)


    def x_reset_to(self, x_state):
        """Resets the state of the environment to a specified x_state"""
        self.x_state = copy.deepcopy(x_state)
        self.s_state = self.s(self.x_state)
        self.previous_alpha_deg = 0 # [deg] value of previous alpha to discourage chattering
        self.episode_step = 0
        return copy.deepcopy(self.s_state)


    def s_reset_to(self, s_state):
        """Resets the state of the environment to a specified s_state"""
        self.s_state = copy.deepcopy(s_state)
        self.x_state = self.x(s_state)
        self.previous_alpha_deg = 0 # [deg] value of previous alpha to discourage chattering
        self.episode_step = 0
        return s_state
        
       
    def repulsion(self, previous_s_state):
        """Calculates the actuated state and its desired maximal value for repulsion"""
        s_r = self.s_state[self.relative_degree-1]
        y_r = (s_r - previous_s_state[self.relative_degree-1])/self.dt
        tol = -2*self.eps - s_r * self.beta
        return y_r, tol

    def repulsion_respect(self, previous_s_state):
        """Verifies whether the repulsion condition holds."""
        y_r, tol = self.repulsion(previous_s_state)
        return y_r <= tol
    
    
    def compute_buffer_max(self, s_state):
        """Calculates the vector of maximal values of the buffer at a given s_state"""
        self.buffer_max = np.array([self.y_max,
                                    self.beta_offset - self.beta * s_state[0],
                                    self.s3_max])
        return self.buffer_max
      
    
    def in_buffer(self, x_state=None, s_state=None):
        """Verifies whether the x_state is in the buffer."""
        if x_state is not None:
            s_state = self.s(x_state)
        
        ### Doesn't work with the handmade buffer, it makes gamma interval
        ### larger than it actually is, so extra states will be declared in-buffer
        ### even if they are not in the handmade buffer.
        if (s_state >= self.buffer_min).all():
            b_max = self.compute_buffer_max(s_state)
            return (s_state <= b_max).all()
            
        return False
    
    
    def update_epsilon(self, agent, state_norm):
        """Updates epsilon using the policy 'agent' instead of using any admissible control input.
        Needs state_norm to call the agent policy"""
        self._epsilon_partition(agent, state_norm)
        return self.eps
        
  
    def _buffer_vertices(self):
        """Calculates the buffer vertices to make them feasible flight condidtions.
        The buffer cannot encompass the whole range of velocities since low speeds make the shuttle
        crash no matter what. The shuttle has to be physically capable of landing
        from all the vertices of the buffer, which requires some fine-tuning vertices."""
        
        self.buffer_vertices = np.zeros((6, 3))
        
        ###################### Vertex 1  #################################
        h = -self.y_max      #  0ft/s
        h_dot = -self.s2_min # -6ft/s
        v = 250. # ft/s
        gamma = np.arcsin(h_dot/v)
        self.buffer_vertices[0] = self.s(np.array([v, gamma, h]))
        
        ###################### Vertex 2  #################################
        h = -self.y_max      #  0ft/s
        h_dot = -self.s2_min # -6ft/s
        v = 300. # ft/s
        gamma = np.arcsin(h_dot/v)
        self.buffer_vertices[1] = self.s(np.array([v, gamma, h]))
        
        ###################### Vertex 3  #################################
        h = -self.y_min      # 50ft/s
        h_dot = -self.s2_min # -6ft/s
        v = 250. # ft/s
        gamma = np.arcsin(h_dot/v)
        self.buffer_vertices[2] = self.s(np.array([v, gamma, h]))
        
        ###################### Vertex 4  #################################
        h = -self.y_min      # 50ft/s
        h_dot = -self.s2_min # -6ft/s
        v = 300. # ft/s
        gamma = np.arcsin(h_dot/v)
        self.buffer_vertices[3] = self.s(np.array([v, gamma, h]))
       
        ###################### Vertex 5  #################################
        h = -self.y_min         #   50ft/s
        h_dot = -self.y_dot_max # -100ft/s
        v = 450. # ft/s
        gamma = np.arcsin(h_dot/v)
        self.buffer_vertices[4] = self.s(np.array([v, gamma, h]))
        
        ###################### Vertex 6  #################################
        h = -self.y_min         #   50ft/s
        h_dot = -self.y_dot_max # -100ft/s
        v = 500. # ft/s
        gamma = np.arcsin(h_dot/v)
        self.buffer_vertices[5] = self.s(np.array([v, gamma, h]))
       
       
        
    def _epsilon_partition(self, agent=None, state_norm=None):
       """Since the Shuttle dynamics are highly nonlinear, the approximation measure
       epsilon takes a value too large for the POLICEd condition.
       Here we partition the buffer into smaller polytopic regions with each their own
       approximation measure epsilon.
       If agent is provided, instead of using any admissible control inputs, we only use
       the provided policy to calculate epsilon. It should reduce epsilon.
       We need state_norm to call the agent policy.
       Buffer is split along s_2. 
       Returns the maximal epsilon and all extra epsilons"""
       
       ### epsilon increases faster at high y_dot, so make partitions smaller
       s2_pts = [self.s2_min, self.y_dot_max]
       ratio = 0.7
       for _ in range(self.nb_buffer_partitions-1):
           s2_pts.insert(-1, s2_pts[-1]*ratio + (1-ratio)*s2_pts[-2])
       s2_pts = np.array(s2_pts)
       
       self.eps = 0
       for i in range(self.nb_buffer_partitions): # iterates over the sub-buffers
           s2_min = s2_pts[i]
           s2_max = s2_pts[i+1]
           s1_max = (s2_min - self.beta_offset)/(-self.beta)
           eps = self._multi_epsilon(s1_max, s2_min, s2_max, agent, state_norm)
           if eps > self.eps:
               self.eps = eps
    
       extra_V = []
       for i in range(1, self.nb_buffer_partitions):
           extra_V.append(np.array([self.y_min, s2_pts[i], self.s3_min]))
           extra_V.append(np.array([self.y_min, s2_pts[i], self.s3_max]))
           s1 = (s2_pts[i] - self.beta_offset)/(-self.beta)
           extra_V.append(np.array([s1, s2_pts[i], self.s3_min]))
           extra_V.append(np.array([s1, s2_pts[i], self.s3_max]))
           
       self.extra_vertices = np.array(extra_V)
         
       
    def _multi_epsilon(self, s1_max, s2_min, s2_max, agent=None, state_norm=None):
        """Calculates the approximation measure epsilon for a given polytope.
        If agent is provided, instead of using any admissible control inputs, we only use
        the provided policy to calculate epsilon. It should reduce epsilon.
        We need state_norm to call the agent policy."""
        
        nb_steps = 10
        
        ### Range of states s1
        s1_min = self.y_min
        s1_step = (s1_max - s1_min)/nb_steps
        s1_range = np.arange(start = s1_min,
                             stop = s1_max + s1_step/2,
                             step = s1_step)
        ### Range of states s2
        s2_step = (s2_max - s2_min)/nb_steps
        s2_range = np.arange(start = s2_min,
                             stop = s2_max + s2_step/2,
                             step = s2_step)
        ### Range of states s3
        s3_step = (self.s3_max - self.s3_min)/nb_steps
        s3_range = np.arange(start = self.s3_min,
                             stop = self.s3_max + s3_step/2,
                             step = s3_step)
        
        action_step = 2*self.action_max/nb_steps
        action_range = np.arange(start = -self.action_max,
                                 stop = self.action_max + action_step/2,
                                 step = action_step)
        
        ### Storing datapoints to perform linear regression on
        N = (nb_steps+1)**(self.state_size + self.action_size)
        States = np.zeros((N, self.state_size))
        Next_States = np.zeros((N, self.state_size))
        Actions = np.zeros((N, self.action_size))
        
        i = 0
        for s1 in s1_range:
            for s2 in s2_range:
                if s2 > self.beta_offset - self.beta * s1: break # go to next s1
                
                for s3 in s3_range:
                    s = np.array([s1, s2, s3])
                    if agent is not None:
                        with torch.no_grad():
                            action_range = agent.evaluate(s, state_norm)
                    
                    for action in action_range:
                        self.s_reset_to(s)
                        next_s, _, done, _ = self.step(np.array([action]))
                        if not done: # ignore crash states
                            States[i] = s
                            Actions[i] = action
                            Next_States[i] = next_s
                            i += 1
                            
        #### Least square fit
        Ones = np.ones((i, 1))
        A = np.concatenate((States[:i], Actions[:i], Ones), axis=1)
        y_r = (Next_States[:i, self.relative_degree-1] - States[:i, self.relative_degree-1])/self.dt
        x, _, _, _ = np.linalg.lstsq(A, y_r, rcond=None)
        
        ### Calculation of epsilon
        CA = np.array([[x[0], x[1], x[2]]])
        CB = x[3]
        Cc = x[4]
        affine_pred = States[:i] @ CA.T + Actions[:i] * CB.T + Cc
        eps = abs(y_r - affine_pred.squeeze()).max()
        return eps
       
   
        
      
  