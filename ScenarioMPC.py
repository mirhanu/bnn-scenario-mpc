# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:59:01 2025

@author: PP56CJ
"""

import casadi as ca
import numpy as np
import torch
from BNNTrain import *
from CartPole import DynamicSystem, CartPole 


class ScenarioMPC:
    """
    Scenario Model Predictive Control (Scenario MPC).

    This implementation solves an optimization problem using multiple sampled trajectories 
    from a given dynamics sampler.

    Args:
        N (int): Prediction horizon (number of future steps).
        n (int): Number of state variables.
        m (int): Number of control inputs.
        S (int): Number of scenario samples.
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control cost matrix.
        state_limits (tuple[np.ndarray, np.ndarray]): Lower and upper bounds for each state variable.
        input_limits (tuple[np.ndarray, np.ndarray]): Lower and upper bounds for each control input.

    Methods:
        solve(x0, dynamics_sampler):
            Solves the optimization problem and returns the optimal control action.
    """

    def __init__(self, N, n, m, S, Q, R, state_limits, input_limits):
        self.N = N
        self.Q = Q
        self.R = R
        self.n = n
        self.m = m
        self.S = S
        self.state_limits = state_limits
        self.input_limits = input_limits

    def solve(self, x0, dynamics_sampler):
        """
        Solve the Scenario MPC problem using dynamics sampler.
        
        Args:
            x0 (np.ndarray): Initial state of the system, shape (state_dim,).
        
        Returns:
            np.ndarray: Optimal control input for the first time step.
        """

        # CasADi optimization variables
        opti = ca.Opti()
        U = opti.variable(self.N, self.m)  # Control sequence
    
        # Define scenario state trajectories
        X_scenario = [opti.variable(self.N + 1, self.n) for _ in range(self.S)]
    
        # Initial state constraint
        for i in range(self.S):
            opti.subject_to(X_scenario[i][0, :].T == x0.flatten())
    
        cost = 0
    
        for t in range(self.N):
            u_t = U[t, :]
    
            # Sample multiple next states using the sampler
            next_states = dynamics_sampler(X_scenario[0][t, :], u_t)
            
            # Compute average cost over scenarios
            for i in range(self.S):
                cost += ca.mtimes([(X_scenario[i][t, :] - next_states[i, :]), self.Q, (X_scenario[i][t, :] - next_states[i, :]).T])
    
            cost += ca.mtimes([U[t, :].T, self.R, U[t, :]])
    
            # Enforce dynamics separately for each scenario
            for i in range(self.S):
                opti.subject_to(X_scenario[i][t + 1, :] == next_states[i, :])
    
            # State constraints
            if self.state_limits:
                x_min, x_max = self.state_limits
                for i in range(self.S):
                    opti.subject_to(X_scenario[i][t, :].T >= x_min)
                    opti.subject_to(X_scenario[i][t, :].T <= x_max)
            # if self.input_limits:
            #     u_min, u_max = self.input_limits
            #     opti.subject_to(U[t, :] >= u_min)
            #     opti.subject_to(U[t, :] <= u_max)
    
        # Define cost function
        opti.minimize(cost / self.S)
    
        # Solver settings
        opti.solver('ipopt')
    
        # Solve the problem
        sol = opti.solve()
    
        return sol.value(U[0, :])  # Return first control input

def main(train_bnn=False):
    """
    Main function to load posterior samples and run Scenario MPC on the CartPole system.
    
    Args:
        train_bnn (bool): If True, trains the BNN from scratch before running MPC.
    """
    #Number of Posterior samples
    num_samples=50
    
    #Time step of the system
    dt=0.05
    
    #File to save and load posteriors
    posterior_file="posterior_samples.pth"
    
    # Initialize the CartPole system and BNN
    system = CartPole(dt=dt)
    bnnModel = BNN(in_dim=system.n + system.m, out_dim=system.n)
    
    # Train BNN if required
    if train_bnn:
        bnnModel.learn_dynamics(self, dynamical_system, num_samples=num_samples, save_path=posterior_file)

    # Load MCMC posterior samples
    bnnModel.load_posterior_samples(posterior_file)

    # Create MPC instance
    mpc = ScenarioMPC(N=10, n=system.n, m=system.m, S=num_samples, 
              Q=np.eye(system.n), R=np.eye(system.m), 
              state_limits=(np.array([-100, -200, -100, -300]), np.array([100, 200, 100, 3000])), 
              input_limits=(np.array([-200, -100]), np.array([200, 100])))

    # Run animation with MPC control rule
    print("Running MPC animation...")
    system.animate(T=0.1, control_law=lambda state, t: mpc.solve(state,bnnModel.forward_casadi), add_noise=True)

if __name__ == "__main__":
    main(train_bnn=False)  # Change to True to retrain BNN