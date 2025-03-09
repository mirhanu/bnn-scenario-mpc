# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 14:02:04 2025

@author: Mirhan Urkmez
"""

import torch
from src.bnn import BNN
from src.cartpole import CartPole
import numpy as np


def main():
    """
    Train the Bayesian Neural Network (BNN) using MCMC on the CartPole system
    and save the posterior samples.
    """
    # Number of Posterior samples
    num_samples = 50
    
    # Time step of the system
    dt = 0.05
    
    # Define state bounds for data generation (each state variable has its own range)
    state_min = np.array([-10, -5, -8, -50])  # Lower limits for state variables
    state_max = np.array([10, 5, 8, 50])      # Upper limits for state variables

    # Define control input bounds (each control variable has its own range)
    control_min = np.array([-20])  # Lower limit for control input
    control_max = np.array([20])   # Upper limit for control input
    
    # File to save posterior samples
    posterior_file = "models/posterior_samples.pth"
    
    # Initialize the CartPole system and BNN
    system = CartPole(dt=dt)
    bnn_model = BNN(in_dim=system.n + system.m, out_dim=system.n)

    # Train BNN using synthetic CartPole data
    print("Training BNN on CartPole dynamics...")


    # Train the BNN on simulated system data with specified bounds
    bnn_model.learn_dynamics(
        dynamical_system=system,   # The system to generate training data from
        num_samples=num_samples,   # Number of posterior samples for MCMC
        state_bounds=(state_min, state_max),  # Set individual bounds for each state
        control_bounds=(control_min, control_max),  # Set individual bounds for control inputs
        save_path=posterior_file   # Save trained posterior samples to file
    )    
    print(f"Training complete. Posterior samples saved to {posterior_file}")


if __name__ == "__main__":
    main()