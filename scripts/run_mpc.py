# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 14:02:15 2025

@author: Mirhan Urkmez
"""

import numpy as np
from src.bnn import BNN
from src.scenario_mpc import ScenarioMPC
from src.cartpole import CartPole

def main(train_bnn=False):
    """
    Main function to load posterior samples and run Scenario MPC on the CartPole system.
    
    Args:
        train_bnn (bool): If True, trains the BNN from scratch before running MPC.
    """
    #Number of Posterior samples
    num_samples=200
    
    #Time step of the system
    dt=0.05
    
    #File to save and load posteriors
    posterior_file="models/posterior_samples.pth"
    
    # Initialize the CartPole system and BNN
    system = CartPole(dt=dt)
    bnn_model = BNN(in_dim=system.n + system.m, out_dim=system.n)
    
    # Train BNN if required
    if train_bnn:
        # Define state bounds for data generation (each state variable has its own range)
        state_min = np.array([-10, -5, -8, -50])  # Lower limits for state variables
        state_max = np.array([10, 5, 8, 50])      # Upper limits for state variables

        # Define control input bounds (each control variable has its own range)
        control_min = np.array([-20])  # Lower limit for control input
        control_max = np.array([20])   # Upper limit for control input
    
        # Train the BNN on simulated system data with specified bounds
        bnn_model.learn_dynamics(
            dynamical_system=system,   # The system to generate training data from
            num_samples=num_samples,   # Number of posterior samples for MCMC
            state_bounds=(state_min, state_max),  # Set individual bounds for each state
            control_bounds=(control_min, control_max),  # Set individual bounds for control inputs
            save_path=posterior_file   # Save trained posterior samples to file
        )


    # Load MCMC posterior samples
    bnn_model.load_posterior_samples(posterior_file)

    # Create MPC instance
    mpc = ScenarioMPC(N=5, n=system.n, m=system.m, S=num_samples, 
              Q=np.eye(system.n), R=np.eye(system.m), 
              state_limits=(np.array([-1000, -2000, -1000, -3000]), np.array([1000, 2000, 1000, 3000])), 
              input_limits=(np.array([-2000, -1000]), np.array([2000, 1000])))

    # Run animation with MPC control rule
    print("Running MPC animation...")
    states, controls = system.animate(T=1.0, control_law=lambda state, t: mpc.solve(state,bnn_model.forward_casadi), add_noise=True)
    return states, controls
if __name__ == "__main__":
    states, controls = main(train_bnn=True)  # Change to True to retrain BNN