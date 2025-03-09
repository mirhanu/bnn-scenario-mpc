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
    num_samples=50
    
    #Time step of the system
    dt=0.05
    
    #File to save and load posteriors
    posterior_file="models/posterior_samples.pth"
    
    # Initialize the CartPole system and BNN
    system = CartPole(dt=dt)
    bnn_model = BNN(in_dim=system.n + system.m, out_dim=system.n)
    
    # Train BNN if required
    if train_bnn:
        bnn_model.learn_dynamics(dynamical_system=system, num_samples=num_samples, save_path=posterior_file)

    # Load MCMC posterior samples
    bnn_model.load_posterior_samples(posterior_file)

    # Create MPC instance
    mpc = ScenarioMPC(N=10, n=system.n, m=system.m, S=num_samples, 
              Q=np.eye(system.n), R=np.eye(system.m), 
              state_limits=(np.array([-100, -200, -100, -300]), np.array([100, 200, 100, 3000])), 
              input_limits=(np.array([-200, -100]), np.array([200, 100])))

    # Run animation with MPC control rule
    print("Running MPC animation...")
    system.animate(T=0.1, control_law=lambda state, t: mpc.solve(state,bnn_model.forward_casadi), add_noise=True)

if __name__ == "__main__":
    main(train_bnn=False)  # Change to True to retrain BNN