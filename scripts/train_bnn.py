# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 14:02:04 2025

@author: Mirhan Urkmez
"""

import torch
from src.bnn import BNN
from src.cartpole import CartPole


def main():
    """
    Train the Bayesian Neural Network (BNN) using MCMC on the CartPole system
    and save the posterior samples.
    """
    # Number of Posterior samples
    num_samples = 50
    
    # Time step of the system
    dt = 0.05
    
    # File to save posterior samples
    posterior_file = "models/posterior_samples.pth"
    
    # Initialize the CartPole system and BNN
    system = CartPole(dt=dt)
    bnn_model = BNN(in_dim=system.n + system.m, out_dim=system.n)
    
    # Train BNN using synthetic CartPole data
    print("Training BNN on CartPole dynamics...")
    bnn_model.learn_dynamics(dynamical_system=system, num_samples=num_samples, save_path=posterior_file)
    
    print(f"Training complete. Posterior samples saved to {posterior_file}")


if __name__ == "__main__":
    main()