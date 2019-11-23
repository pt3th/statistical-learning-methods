#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:14:54 2019

@author: q
"""

import numpy as np
'''
python implementation for hidden Markov model Viterbi algorithm
using statistics learning methods(LI Hang) Example 10.2
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
T = 3
observation_sequence = [0, 1, 0]

