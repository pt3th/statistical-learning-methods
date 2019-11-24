#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:14:54 2019

@author: q
"""

import numpy as np
'''
python implementation for hidden Markov model Viterbi algorithm
using statistics learning methods(LI Hang) Example 10.3
States = {1,2,3}
Observation = {red,white} or {0,1}
'''
#state transitin matrix
A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])

#observation probabiity matrix
B = np.array([[.5,.5],[.4,.6],[.7,.3]])

#initial probability distribution
pi = np.array([.2,.4,.4])

#initialization
state_num = 3
observation_sequence = [0,1,0]
T = len(observation_sequence)

sigma = np.zeros((state_num,T))
sigma[:,0] = pi * B[:,observation_sequence[0]]

#phi = np.zeros((state_num,T))
#phi[:,0] = 0

path = []

for t in np.arange(2,T+1):
    sigma[:,t-1] = np.max(np.tile(sigma[:,t-2],(state_num,1)).T*A*(np.tile(B[:,observation_sequence[t-1]],(state_num,1))),axis=0)
    path.append(np.argmax(np.tile(sigma[:,t-2],(state_num,1)).T*A*(np.tile(B[:,observation_sequence[t-1]],(state_num,1))),axis=0))

last_node = np.argmax(sigma[:,T-1])
final_prob = np.max(sigma[:,T-1])
final_path = []

#backtracking
final_path.append(last_node+1)
for i in np.arange(len(path)-1,-1,-1):
    final_path.append(path[i][last_node]+1)
    last_node = path[i][last_node]
final_path.reverse()
print("probability of optimum path: ", final_prob)
print("optimum path: ", final_path)
