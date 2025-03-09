# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 19:08:41 2025

@author: Mirhan Urkmez
"""


import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer import MCMC, NUTS, Predictive
from src.cartpole import CartPole 
import numpy as np
import casadi as ca



class BNN(PyroModule):
    """
    Bayesian Neural Network (BNN) using Pyro for probabilistic inference.
    
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hid_dim (int): Number of hidden units per layer.
        n_hid_layers (int): Number of hidden layers.
        prior_scale (float): Scale of the prior distribution for weights and biases.
        posterior_samples (dict, optional): Stored posterior samples from MCMC.
    """
    
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5., posterior_samples=None):
        """Initialize the BNN architecture and define prior distributions."""
        super().__init__()
        
        self.activation = nn.Tanh()  # Non-linearity used in hidden layers
        
        # Define network architecture (input layer -> hidden layers -> output layer)
        self.layer_sizes = [in_dim] + [hid_dim] * n_hid_layers + [out_dim]
        
        # Create layers as PyroModules to enable probabilistic modeling
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[i], self.layer_sizes[i + 1]) 
                      for i in range(len(self.layer_sizes) - 1)]
        self.layers = PyroModule[nn.ModuleList](layer_list)
        
        # Assign prior distributions to weights and biases
        for i, layer in enumerate(self.layers):
            layer.weight = PyroSample(
                dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[i])).expand(
                    [self.layer_sizes[i + 1], self.layer_sizes[i]]).to_event(2)
            )
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[i + 1]]).to_event(1))
        self.posterior_samples = posterior_samples  # Store posterior samples for CasADi-based MPC

    def store_posterior_samples(self, samples):
        """ Store posterior samples after MCMC training. """
        self.posterior_samples = samples
    def train(self, x_train, y_train, num_samples=50, warmup_steps=50, save_path=None):
        """
        Train the BNN using MCMC and store posterior samples.

        Args:
            x_train (torch.Tensor): Input training data.
            y_train (torch.Tensor): Target training data.
            num_samples (int): Number of posterior samples.
            warmup_steps (int): Number of warmup steps.
            save_path (str, optional): Path to save posterior samples.
        """
        # Define NUTS sampler
        nuts_kernel = NUTS(self, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)

        print("Running MCMC training...")
        mcmc.run(x_train, y_train)

        # Store posterior samples
        self.posterior_samples = mcmc.get_samples()

        # Optionally save posterior samples
        if save_path:
            torch.save(self.posterior_samples, save_path)
            print(f"Posterior samples saved to {save_path}")
    def learn_dynamics(self, dynamical_system, num_initial_conditions=10, T=5.0, num_samples=50, save_path=None):
        """
        Train the BNN using synthetic data from a given system.

        Args:
            dynamical_system (object): System instance to generate data.
            num_initial_conditions (int): Number of initial conditions.
            T (float): Simulation time.
            num_samples (int): Number of posterior samples.
            save_path (str, optional): Path to save posterior samples.
        """
        print("Generating training data...")
        
        # Generate synthetic dataset using the provided system
        current_state, next_state, controls = dynamical_system.generate_dataset(num_initial_conditions, T)

        # Flatten the inputs: current_state and controls into a single input vector
        inputs = np.hstack([current_state, controls])  # Shape: (N, state_dim + control_dim)
        targets = next_state  # Shape: (N, state_dim)

        # Convert data to PyTorch tensors
        x_train = torch.from_numpy(inputs).float()
        y_train = torch.from_numpy(targets).float()

        # Train using MCMC
        self.train(x_train, y_train, num_samples=num_samples, save_path=save_path)

    def load_posterior_samples(self, load_path):
        """
        Load stored posterior samples from file.

        Args:
            load_path (str): Path to the posterior samples file.
        """
        self.posterior_samples = torch.load(load_path, weights_only=True)
        print(f"Posterior samples loaded from {load_path}")

    def forward(self, x, y=None):
        """
        Forward pass of the Bayesian Neural Network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
            y (torch.Tensor, optional): Target values for supervised learning (default: None).
        
        Returns:
            torch.Tensor: The mean predictions (mu) of the BNN.
        """
        x = x.view(x.shape[0], -1)  # Ensure input shape is correct
        
        # Pass input through hidden layers with activation function
        x = self.activation(self.layers[0](x))  # First hidden layer
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # Hidden layers
        
        # Final layer (output layer, no activation function)
        mu = self.layers[-1](x).squeeze()  # Predicted mean
        
        # Define noise model for output uncertainty
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1))  # Sample noise scalar
        sigma = sigma.expand(mu.shape)  # Ensure sigma has the same shape as mu
        
        # Apply probabilistic observation model
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
        
        return mu
    def forward_casadi(self, x, u):
        """
        CasADi-compatible forward pass using stored posterior samples.
    
        Args:
            x (ca.MX): Current state (CasADi symbolic variable).
            u (ca.MX): Control input (CasADi symbolic variable).
    
        Returns:
            ca.MX: Matrix of predicted next states (num_samples x state_dim).
        """
        if self.posterior_samples is None:
            raise RuntimeError("Posterior samples are not stored! Run MCMC and store them first.")
    
        num_samples = self.posterior_samples["layers.0.weight"].shape[0]  # Number of posterior samples
    
        # Reshape x and u to column vectors
        x = ca.reshape(x, -1, 1)  
        u = ca.reshape(u, -1, 1)
        x_u = ca.vertcat(x, u)  # Concatenate state and control
    
        sampled_next_states = []
    
        # Loop through all posterior samples
        for sample_idx in range(num_samples):
            x_u_sample = x_u  # Start with the same input
    
            for i in range(len(self.layer_sizes) - 1):
                W_name = f"layers.{i}.weight"
                b_name = f"layers.{i}.bias"
    
                # Get sample-specific weights and biases (different per sample)
                W = self.posterior_samples[W_name][sample_idx].detach().numpy()
                b = self.posterior_samples[b_name][sample_idx].detach().numpy()
    
                # Compute layer output for this sample
                x_u_sample = ca.mtimes(W, x_u_sample) + b
    
                # Apply activation function (except for last layer)
                if i < len(self.layer_sizes) - 2:
                    x_u_sample = ca.tanh(x_u_sample)
    
            sampled_next_states.append(x_u_sample)  # Store next state for this sample
    
        # Stack all sampled next states into a single matrix (num_samples x state_dim)
        return ca.horzcat(*sampled_next_states).T