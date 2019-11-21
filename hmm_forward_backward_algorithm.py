#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:40:40 2019

@author: q
"""

#forward backward algorithm
import numpy as np

'''
python implementation for HMM forward/backward algorithm based on statistics learning methods(LI Hang)
States = {1,2,3}
Observation = {red,white} or {0,1}
'''

def forward(A,B,pi,observation_sequence):
    state_num = A.shape[0]
    T = len(observation_sequence)
    alpha = np.zeros((T,state_num))
    alpha[:,0] = pi*(B[:,observation_sequence[0]])
    for i in range(1,T):
        alpha[:,i] = np.dot(A.T,alpha[:,i-1]) * B[:,observation_sequence[i]]

    prob = np.sum(alpha[:,T-1])
    return prob, alpha

def backward(A,B,pi,observation_sequence):
    state_num = A.shape[0]
    T = len(observation_sequence)
    beta = np.zeros((T,state_num))
    beta[:,T-1] = 1
    for i in np.arange(T-2,-1,-1):
        beta[:,i] = np.dot(A,B[:,observation_sequence[i+1]]*beta[:,i+1])    

    prob = np.sum(np.dot(pi,B[:,observation_sequence[0]]*beta[:,0]))
    return prob,beta

def forward_backward(A,B,alpha,beta,t):
    #equation (10.22)
    #t = 1,2,...,T-1
    #prob =np.dot(alpha[:,t-1],np.dot(A,B[:,observation_sequence[t]]*beta[:,t]))
    
    #simplified
    #t=1,2,...T
    prob = np.dot(alpha[:,t-1],beta[:,t-1])
    return prob
#state transitin matrix
A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])

#observation probabiity matrix
B = np.array([[.5,.5],[.4,.6],[.7,.3]])

#initial probability distribution
pi = np.array([.2,.4,.4])

#using backward algorithm to calculate the probability of observation sequence (red,white,red), T = 3
#initialization
observation_sequence = [0, 1, 0]
prob_forward,alpha = forward(A,B,pi,observation_sequence)
prob_backward,beta = backward(A,B,pi,observation_sequence)

#t = 1, 2, ... , T-1
t = 2
prob_forward_backward = forward_backward(A,B,alpha,beta,t)

print("transition matrix:")
print(A)
print("observation probability matrix:")
print(B)
print("initial probability distribution:")
print(pi)
print("observation sequence: ",observation_sequence)
print("forward algorithm: the probability for observation sequence is: %f " %(prob_forward))
print("backward algorithm: the probability for observation sequence is: %f " %(prob_backward))
print("forward-backward algorithm: the probability for observation sequence is: %f " %(prob_forward_backward))