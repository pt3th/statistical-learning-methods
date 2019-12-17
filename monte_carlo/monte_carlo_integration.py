#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:59:34 2019

@author: q
"""

'''
python implementation for Monte Carlo Integration 
using statistics learning methods(LI Hang) Example 19.1 and 19.2
'''

import numpy as np
from scipy.integrate import quad

def f1(x):
    return np.exp(-x**2/2)

def f2(x):
    return x

x_array1 = np.random.rand(10000)
f1x = f1(x_array1)
mean1 = np.mean(f1x)
val1,err1 = quad(lambda x:np.exp(-x**2/2),0,1)
print("Result of Example 19.1")
print("Monte Carlo Method: %f Integration Routine: %f"%(mean1,val1))

x_array2 = np.random.randn(10000)
f2x = f2(x_array2)
mean2 = np.mean(f2x)
val2,err2 = quad(lambda x:x*np.exp(-x**2/2)/np.sqrt(2*np.pi),-np.inf,np.inf)
print("Result of Example 19.2")
print("Monte Carlo Method: %f Integration Routine: %f"%(mean2,val2))
    