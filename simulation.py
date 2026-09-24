# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:20:01 2026

@author: oscar
"""

#libs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


#data
frames = 1000
dt = 0.01
n = 500
epsilon = 0.1
G = 0.001 #units m^3 kg^-1 s^-2 #6.6743 *10 ** -11 #units m^3 kg^-1 s^-2


pos_arr = np.random.rand(n, 3)
vel_arr = np.zeros((n, 3))
acc_arr = np.empty((n, 3))
mass_arr = np.ones(n)

animation_pos_matrix = []


#text

#initialization

#update loop 
for frame in range(frames):
    animation_pos_matrix.append(pos_arr)
    #difference matrix between points
    r_diff = pos_arr[np.newaxis, :, :]- pos_arr[:, np.newaxis, :]
    
    dist_sq = np.sum(r_diff**2, axis=-1) + epsilon**2
    np.fill_diagonal(dist_sq, np.inf)
    inv_dist_cube = dist_sq**(-1.5)
    term = mass_arr[np.newaxis, :, np.newaxis] * r_diff * inv_dist_cube[:, :, np.newaxis]
    acc_arr = G * np.sum(term, axis=1)
    vel_arr += acc_arr*dt
    pos_arr += vel_arr*dt
    pos_arr = pos_arr % 1  
    
    
        
        
            
            
            
            
