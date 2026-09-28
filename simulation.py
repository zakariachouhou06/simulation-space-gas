# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:20:01 2026

@author: oscar
"""


#libs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
np.set_printoptions(threshold=np.inf)

class Simulation:

  def __init__(
      self, n=500, frames=10000, dt=0.01, epsilon=0.1, G=0.001,collision_distance=0.005
  ):
    # Parameters
    self.n = n
    self.frames = frames
    self.dt = dt
    self.epsilon = epsilon
    self.G = G

    # Initialize arrays as instance variables
    self.pos_arr = np.random.rand(self.n, 3)
    self.vel_arr = np.zeros((self.n, 3))
    self.acc_arr = np.empty((self.n, 3))
    self.mass_arr = np.ones(self.n)

    self.animation_pos_matrix = []
    self.collision_distance = collision_distance

  def next_frame(self):
    # Difference matrix between points (using self. for instance variables)
    r_diff = (
        self.pos_arr[np.newaxis, :, :] - self.pos_arr[:, np.newaxis, :]
    )
    v_diff = (
        self.vel_arr[np.newaxis, :, :] - self.vel_arr[:, np.newaxis, :]
    )
    # Collision detection
    dist_sq = np.sum(r_diff**2, axis=-1)
    dist = np.sqrt(dist_sq)
    dist_vel = (np.sum(r_diff * v_diff, -1))/dist
    collision_detection = (dist + dist_vel*self.dt < self.collision_distance)  #Boolean values for collision detection.
    np.fill_diagonal(collision_detection, False)

    # Collision
    dist_safe = dist.copy()
    np.fill_diagonal(dist_safe, 1.0)
    n_hat = r_diff / dist_safe[:,:,np.newaxis]
    v_par = np.sum(v_diff * collision_detection[:,:, np.newaxis] * n_hat, -1)

    self.vel_arr = self.vel_arr + np.sum(v_par[:,:, np.newaxis] * n_hat, 1)# Scratching my head

    dist_smth = dist_sq + self.epsilon**2
    np.fill_diagonal(dist_smth, np.inf)
    inv_dist_cube = dist_smth**-1.5

    term = (
        self.mass_arr[np.newaxis, :, np.newaxis]
        * r_diff
        * inv_dist_cube[:, :, np.newaxis]
    )
    self.acc_arr = self.G * np.sum(term, axis=1)
    self.vel_arr += self.acc_arr * self.dt
    self.pos_arr += self.vel_arr * self.dt
    self.pos_arr = self.pos_arr % 1
    return self.pos_arr
  

    
    
        
        
            
            
            
            
