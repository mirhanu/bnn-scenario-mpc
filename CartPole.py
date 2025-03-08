# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 17:07:23 2025

@author: Mirhan Urkmez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from DynamicSystem import *

# CartPole class as a subclass of DynamicSystem
class CartPole(DynamicSystem):
    """
    CartPole system, a subclass of `DynamicSystem`.

    Args:
        mass_pole (float): Mass of the pole.
        M (float): Mass of the cart.
        L (float): Length of the pole.
        d (float): Damping coefficient.
        dt (float): Time step.
        initial_state (np.ndarray, optional): Initial state of the system.
    """
    
    G = 9.81  # Class-level constant for gravity
    MS_PER_SEC = 1000 # Class-level constant for conversion
    
    def __init__(self, mass_pole=1.0, M=5.0, L=2.0, d=1.0, dt=0.01, initial_state=None):
        """Initialize CartPole system parameters."""
        # Set default initial state if none is provided
        default_state = np.array([0, 0, np.pi / 20, 0])  
        state = initial_state if initial_state is not None else default_state
    
        # Call the parent constructor with n=4 (state dimension) and m=1 (control dimension)
        super().__init__(n=4, m=1, dt=dt, state=state)
    
        # Assign system parameters
        self.mass_pole = mass_pole  # Renamed from m to mass_pole
        self.M = M
        self.L = L
        self.d = d
        self.g = self.G
    
    def dynamics(self, state, u):
        """Implement the CartPole specific dynamics."""
        x, x_dot, theta, theta_dot = state
        u = u[0]  # Extract scalar control from array of size m=1
        
        Sy, Cy = np.sin(theta), np.cos(theta)
        D = self.mass_pole * self.L**2 * (self.M + self.mass_pole * (1 - Cy**2))
        
        dx = x_dot
        dvx = (1 / D) * (-self.mass_pole**2 * self.L**2 * self.g * Cy * Sy + self.mass_pole * self.L**2 * (self.mass_pole * self.L * theta_dot**2 * Sy - self.d * x_dot) + self.mass_pole * self.L**2 * u)
        dtheta = theta_dot
        domega = (1 / D) * ((self.M + self.mass_pole) * self.mass_pole * self.g * self.L * Sy - self.mass_pole * self.L * Cy * (self.mass_pole * self.L * theta_dot**2 * Sy - self.d * x_dot) - self.mass_pole * self.L * Cy * u)
        
        return np.array([dx, dvx, dtheta, domega], dtype=np.float64)
    
    def animate(self, T=5.0, control_law=lambda state, t: np.zeros(1), add_noise=False, noise_std=0.01):
        """
        Animate the CartPole system.

        Args:
            T (float): Simulation time.
            control_law (function): Function mapping state and time to control input.
            add_noise (bool): Whether to add noise.
            noise_std (float): Standard deviation of noise.
        """
        states, _ = self.simulate(control_law, T=T, add_noise=add_noise, noise_std=noise_std)  # Now returns both states and controls
        fig, ax = plt.subplots()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
    
        # Cart dimensions
        cart_width = 1.0
        cart_height = 0.5
    
        # Create cart (rectangle)
        cart = plt.Rectangle((-0.5, -0.25), cart_width, cart_height, fc='black', ec='black')
        ax.add_patch(cart)
    
        # Create pendulum (line)
        pole, = ax.plot([], [], 'r-', lw=3)  # Pendulum rod
        joint, = ax.plot([], [], 'bo', markersize=6)  # Pivot joint
    
        def update(frame):
            """Update the cart and pendulum for each frame."""
            x = states[frame, 0]  # Cart position
            theta = states[frame, 2]  # Pendulum angle
            
            # Cart position update
            cart.set_xy((x - cart_width / 2, -cart_height / 2))
    
            # Pendulum position update
            pend_x = x + self.L * np.sin(theta)
            pend_y = self.L * np.cos(theta)
            pole.set_data([x, pend_x], [0, pend_y])
    
            # Joint position update
            joint.set_data([x], [0])
    
            return cart, pole, joint
    
        self.ani = animation.FuncAnimation(
            fig, update, frames=len(states), interval=self.dt * self.MS_PER_SEC, repeat=False
        )    
        plt.show()