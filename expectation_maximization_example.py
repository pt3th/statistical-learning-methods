#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:40:35 2019

@author: q
"""

'''
python implementation for statistics learning methods(LI Hang) Example 9.1
'''
import numpy as np

#output
y = np.array([1,1,0,1,0,0,1,0,1,1])
pi_list = []
p_list = []
q_list = []
pi_0 = .46
p_0 = .55
q_0 = .67
pi_list.append(pi_0)
p_list.append(p_0)
q_list.append(q_0)

iter_num = 5
epsilon = 0.001
for i in range(iter_num):
    #E-step
    mu = (pi_list[i]*p_list[i]**(y)*(1-p_list[i])**(1-y))/(pi_list[i]*p_list[i]**(y)*(1-p_list[i])**(1-y) + (1-pi_list[i])*q_list[i]**(y)*(1-q_list[i])**(1-y))
    #M-step
    pi = np.mean(mu)
    p = np.dot(mu,y)/np.sum(mu)
    q = np.dot(1-mu,y)/np.sum(1-mu)
    
    pi_list.append(pi)
    p_list.append(p)
    q_list.append(q)
    if np.linalg.norm(np.array([pi_list[i+1]-pi_list[i], p_list[i+1]-p_list[i], q_list[i+1]-q_list[i]])) < epsilon:
        break
print("iteration number: %d"%(i+1))
print("pi: %f\np: %f\nq: %f"%(pi,p,q))
    
    

