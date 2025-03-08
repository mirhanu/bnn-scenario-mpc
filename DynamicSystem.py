# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 12:59:23 2025

@author: PP56CJ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial



# Base class for dynamic systems
class DynamicSystem:
    """
    Base class for dynamic systems.

    Args:
        n (int): State dimension.
        m (int): Control dimension.
        dt (float): Time step.
        state (np.ndarray, optional): Initial state.
    """
    
    def __init__(self, n=4, m=1, dt=0.01, state=None):
        """Initialize the dynamic system."""
        self.n = n  # State dimension
        self.m = m  # Control dimension
        self.dt = dt
        self.state = state if state is not None else np.zeros(n)

    
    def dynamics(self, state, u):
        """This method should be overridden in child classes."""
        raise NotImplementedError("The dynamics method must be implemented by subclasses.")
    
    def rk4_step(self, state, u):
        """Performs one Runge-Kutta step."""
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(state + self.dt * k3, u)
        return state + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, control_law=None, T=5.0, add_noise=False, noise_std=0.01):
        """
        Simulate the system over a time horizon.

        Args:
            control_law (function, optional): Function computing control inputs, default is zero control.
            T (float): Total simulation time.
            add_noise (bool): Whether to add Gaussian noise.
            noise_std (float): Standard deviation of noise.

        Returns:
            tuple: (states, controls) as NumPy arrays.
        """
        # Define a default control law that returns a zero array of appropriate size
        if control_law is None:
            control_law = lambda state, t: np.zeros(self.m)
            
        num_steps = int(T / self.dt)
        states = np.zeros((num_steps + 1, self.n))  # Preallocate memory using self.n
        controls = np.zeros((num_steps, self.m))    # Preallocate memory for control inputs
        states[0] = self.state  # Set initial state

        for i in range(num_steps):
            t = i * self.dt  # Current time
            u = control_law(states[i], t)  # Compute control input
            
            # Ensure control has correct dimensions
            if np.isscalar(u):
                u = np.array([u])
                
            controls[i] = u  # Store the control input
            next_state = self.rk4_step(states[i], u)  # Compute next state
            
            # Add optional noise
            if add_noise:
                next_state += np.random.normal(0, noise_std, size=next_state.shape)

            states[i + 1] = next_state  # Store new state
        
        self.state = states[-1]  # Update system state
        return states, controls
    

    def generate_dataset(self, num_initial_conditions=10, T=5.0, add_noise=False, noise_std=0.01,
                     state_bounds=(-1, 1), control_bounds=(-1, 1)):
        """
        Generate a dataset for learning system dynamics.

        Args:
            num_initial_conditions (int): Number of sampled initial conditions.
            T (float): Simulation time per trajectory.
            add_noise (bool): Whether to add Gaussian noise.
            noise_std (float): Standard deviation of noise.
            state_bounds (tuple): Range for sampling initial states.
            control_bounds (tuple): Range for sampling control inputs.

        Returns:
            tuple: (current_state, next_state, controls) as NumPy arrays.
        """
        num_steps = int(T / self.dt)
        dataset_size = num_initial_conditions * num_steps
    
        # Preallocate input-output arrays using self.n and self.m
        current_state = np.zeros((dataset_size, self.n))  # x
        next_state = np.zeros((dataset_size, self.n))  # x⁺
        controls = np.zeros((dataset_size, self.m))  # u
    
        idx = 0
        for _ in range(num_initial_conditions):
            # Sample a random initial state
            initial_state = np.random.uniform(state_bounds[0], state_bounds[1], size=self.n)
            self.state = initial_state  
    
            # Define a random control law for the simulation
            def random_control_law(state, t):
                return np.random.uniform(control_bounds[0], control_bounds[1], size=self.m)  # Random control input
    
            # Simulate the system with the control law
            states, applied_controls = self.simulate(control_law=random_control_law, T=T, add_noise=add_noise, 
                                   noise_std=noise_std)
    
            # Store (x, u) → x⁺
            current_state[idx:idx + num_steps, :] = states[:-1]  # x
            next_state[idx:idx + num_steps, :] = states[1:]  # x⁺
            controls[idx:idx + num_steps, :] = applied_controls  # Use the actual applied controls
    
            # Update the index for the next batch
            idx += num_steps
    
            self.state = states[-1]  # Update system state to last state of current trajectory
    
        return current_state, next_state, controls
