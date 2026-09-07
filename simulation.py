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
frames = 100
dt = 0.1
n = 100
epsilon = 0.1
G = 6.6743 *10 ** -11 #units m^3 kg^-1 s^-2


pos_matrix = np.random.rand(n, 3)
vel_matrix = np.empty((n, 3))
acc_matrix = np.empty((n, 3))
mass_arr = np.empty(n)

animation_pos_matrix = []


#text

#initialization

#update loop 
for frame in range(frames):
    animation_pos_matrix.append(pos_matrix)
    
    for i in range(n):
        acc_r =  np.array([0,0,0],dtype = float)  #resterende kracht wordt hier onder berekened
        for j in range(n):
            if i == j:
                continue #check if you not take the same point to avoid x/0
            else:
                r_vec = pos_matrix[j]-pos_matrix[i] #vector from particle i to particle j 
                r_norm = np.linalg.norm(r_vec)
                acc_r += (r_vec/r_norm)* G / (r_norm**2 + epsilon**2)
        acc_matrix[i] = acc_r
    vel_matrix += acc_matrix*dt
    pos_matrix += vel_matrix*dt
    pos_matrix = pos_matrix % 1
    
    
        
        
            
            
            
            