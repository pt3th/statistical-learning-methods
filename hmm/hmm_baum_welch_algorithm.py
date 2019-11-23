#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:26:53 2019

@author: q
"""

import numpy as np

'''
python implementation for Baum-Welch algorithm of hidden Markov model;
using statistics learning methods(LI Hang) Example 10.2;
in this case, only the observation sequence, the number of states, and the number of output are known 
and we should estimate the transition matrix A, the observation probabiity matrix B, and the initial probability distribution pi
'''

def forward(A,B,pi,observation_sequence):
    state_num = A.shape[0]
    T = len(observation_sequence)
    alpha = np.zeros((state_num,T))
    alpha[:,0] = pi*(B[:,observation_sequence[0]])
    for i in range(1,T):
        alpha[:,i] = np.dot(A.T,alpha[:,i-1]) * B[:,observation_sequence[i]]

    prob = np.sum(alpha[:,T-1])
    return prob, alpha

def backward(A,B,pi,observation_sequence):
    state_num = A.shape[0]
    T = len(observation_sequence)
    beta = np.zeros((state_num,T))
    beta[:,T-1] = 1
    for i in np.arange(T-2,-1,-1):
        beta[:,i] = np.dot(A,B[:,observation_sequence[i+1]]*beta[:,i+1])    

    prob = np.sum(np.dot(pi,B[:,observation_sequence[0]]*beta[:,0]))
    return prob,beta

def forward_backward(A,B,alpha,beta,t):
    #simplified
    #t=1,2,...T
    prob = np.dot(alpha[:,t-1],beta[:,t-1])
    return prob

#assume that the number of states is 3 and there are two different output
state_num = 3;
output_num = 2;

observation_sequence = [0,1,1,0]
T = len(observation_sequence)

#initialization
#A = np.ones((state_num,state_num))/state_num;
#B = np.ones((state_num,output_num))/output_num
#should break symmetry, so initialize A,B as follows
A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])
B = np.array([[.5,.5],[.4,.6],[.7,.3]])
pi = np.ones((state_num))/state_num
A_list = []
B_list = []
pi_list = []
A_list.append(A)
B_list.append(B)
pi_list.append(pi)
pi[0] = pi[0] - 1/(state_num+1)
pi[-1] = pi[-1] + 1/(state_num+1)
iter_num = 100;
t = T-1;
ksi = np.zeros((T-1,state_num,state_num))
epsilon = 0.0001;
for i in range(iter_num):
    _,alpha = forward(A,B,pi,observation_sequence)
    _,beta = backward(A,B,pi,observation_sequence)
    prob = forward_backward(A,B,alpha,beta,t)
    gamma = (alpha*beta)/prob
    for j in range(T-1):
        ksi[j,:,:] = np.tile(alpha[:,j],(state_num,1)).T*A*(np.tile(B[:,observation_sequence[j+1]]*beta[:,j+1],(state_num,1)))
#        for p in range(state_num):
#            for q in range(state_num):
#                ksi[j,p,q] = alpha[p,j]*A[p,q]*B[q,observation_sequence[j+1]]*beta[q,j+1]
    ksi = ksi/prob
    A_prev = A
    B_prev = B
    pi_prev= pi
    A = np.sum(ksi,axis=0)/np.sum(gamma[:,0:T-1],axis=1,keepdims=True)
    for k in range(output_num):
        B[:,k]=np.sum((np.array(observation_sequence) == k)*1*gamma,axis=1)/np.sum(gamma,axis=1)
    pi = gamma[:,0]
    if np.linalg.norm(A-A_prev) + np.linalg.norm(B-B_prev) + np.linalg.norm(pi-pi_prev) < epsilon:
        print("done!")
        break;
print("iteration number: %d"%(i+1))
print("transition matrix:")
print(A)
print("observation probability matrix:")
print(B)
print("initial probability distribution:")
print(pi)