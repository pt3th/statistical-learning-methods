#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:05:21 2019

@author: q
"""

'''
python implementation for statistics learning methods(LI Hang) exercise 9.3
'''

import numpy as np

def gaussian_distribution(mu,sigma,y):
    return np.exp(-(y-mu)**2/(2*(sigma**2)))/sigma/((2*np.pi)**0.5)

def response_gamma(alpha,mu,sigma,y):
    k = len(alpha)
    gamma = alpha[0]*gaussian_distribution(mu[0],sigma[0],y)
    for i in range(1,k):
        gamma2 = alpha[i]*gaussian_distribution(mu[i],sigma[i],y) 
        gamma = np.concatenate((gamma,gamma2))
    gamma = gamma.reshape(k,len(y)).T
    gamma = gamma/np.sum(gamma,axis=1,keepdims=True)
    return gamma
#data
y = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75])

#number of different gaussian distributions
k = 3
N = len(y)
#
alpha_list = []
all_mu_list = []
all_sigma_list = []
mu_list = []
sigma_list = []

#initialization
alpha_0 = np.ones((k))/k
alpha_0[0] = alpha_0[0]-1/(k+1)
alpha_0[k-1] = alpha_0[k-1]+1/(k+1)
for i in range(k):
    sigma_0 = np.std(y)+i
    mu_0 = np.mean(y)+i
    mu_list.append(mu_0)
    sigma_list.append(sigma_0)
alpha_list.append((alpha_0))
all_mu_list.append(np.array(mu_list))
all_sigma_list.append(np.array(sigma_list))

#EM algorithm
iter_num = 100
epsilon=0.0001
for i in range(iter_num):
    #E-step
    gamma = response_gamma(alpha_list[i],all_mu_list[i],all_sigma_list[i],y)
    #M-step
    gamma_sum = np.sum(gamma,axis = 0)
    mu = np.dot(y,gamma)/gamma_sum
    all_mu_list.append((mu))
    alpha = gamma_sum/N
    alpha_list.append((alpha))
    sigma_list = []
    for j in range(k):
        sigma = (np.dot(gamma[:,j],(y-mu[j])**2)/gamma_sum[j])**0.5
        sigma_list.append(sigma)
    all_sigma_list.append(np.array(sigma_list))
    #print(alpha_list)
    #print(all_mu_list)
    #print(all_sigma_list)
    parameter_norm = np.linalg.norm(np.array([alpha_list[i+1]-alpha_list[i], all_mu_list[i+1]-all_mu_list[i], all_sigma_list[i+1]-all_sigma_list[i]]))
    #print(parameter_norm)
    if parameter_norm < epsilon:
        print("done! iteration number: ",i+1)
        break

print("alpha: ", alpha_list[-1])
print("mu: ", all_mu_list[-1])
print("sigma: ", all_sigma_list[-1])

    
    
