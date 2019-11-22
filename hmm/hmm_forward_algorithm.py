#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:04:44 2019

@author: q
"""

#forward algorithm

import numpy as np

'''
python implementation for statistics learning methods(LI Hang) Example 10.2 page 177
States = {1,2,3}
Observation = {red,white} or {0,1}
'''
#state transitin matrix
A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])

#observation probabiity matrix
B = np.array([[.5,.5],[.4,.6],[.7,.3]])

#initial probability distribution
pi = np.array([.2,.4,.4])

#using forward algorithm to calculate the probability of observation sequence (red,white,red), T = 3
#initialization
state_num = 3
T = 3
observation_sequence = [0, 1, 0]
a = np.zeros((T,state_num))
a[:,0] = pi*(B[:,observation_sequence[0]])

for i in range(1,T):
    a[:,i] = np.dot(A.T,a[:,i-1]) * B[:,observation_sequence[i]]

prob = np.sum(a[:,T-1])
print("The probability for observation sequence:",observation_sequence, "is: %f " %(prob))