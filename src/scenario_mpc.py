# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:59:01 2025

@author: Mirhan Urkmez
"""

import casadi as ca
import numpy as np
import torch


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
                cost += ca.mtimes([(X_scenario[i][t, :] ), self.Q, (X_scenario[i][t, :]).T])
    
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

