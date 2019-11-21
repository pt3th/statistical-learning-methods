#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:44:40 2019

@author: q
"""

'''
python implementation for statistics learning methods(LI Hang) Example 8.2 page 151
'''

import numpy as np

def get_cutting_point(x,y,step):
    best_dis = 1000000000
    best_cut_point = 0
    best_c1 = 0
    best_c2 = 0
    for cut_point in np.arange((x[0]+x[1])/2,x[-1],step):
        sub_y1 = [y[i] for (t,i) in zip(x,range(len(x))) if t < cut_point]
        sub_y2 = [y[i] for (t,i) in zip(x,range(len(x))) if t >= cut_point] 
        c1 = np.mean(sub_y1)
        c2 = np.mean(sub_y2)
        dis = np.sum((sub_y1-c1)**2) + np.sum((sub_y2-c2)**2)
        if dis < best_dis:
            best_cut_point = cut_point
            best_c1 = c1
            best_c2 = c2
            best_dis = dis
    return best_cut_point,best_c1,best_c2
        
def regression(x,cutpoint,c1,c2):
    y = (x<cutpoint)*c1
    y = y + (x>cutpoint)*c2
    return y

def regression_2(x,cutpoints,c1s,c2s):
    pred_y = x - x
    for i in range(len(cutpoints)):
        pred_y = pred_y + regression(x,cutpoints[i],c1s[i],c2s[i])
    return pred_y

#training set
x = np.arange(1,11)
y = np.array([5.56,5.70,5.91,6.40,6.80,7.05,8.90,8.70,9.00,9.05])

cutpoints = []
c1s = []
c2s = []
step = 1
max_iternum = 6
error_threshold = 0.2
res = y
for i in range(max_iternum):
    cutpoint,c1,c2 = get_cutting_point(x,res,step)
    cutpoints.append(cutpoint)
    c1s.append(c1)
    c2s.append(c2)
    pred_y = regression_2(x,cutpoints,c1s,c2s)
    reg_error = np.sum((pred_y-y)**2)
    if reg_error < error_threshold:
        print("regression boosting tree found!")
        break
    res = y - pred_y

#res1 = y-y1
#cutpoint2,c21,c22 = get_cutting_point(x,res1,step)
#y2 = regression(x,cutpoint2,c21,c22)
#
#res2 = res1-y2
#cutpoint3,c31,c32 = get_cutting_point(x,res2,step)
#y3 = regression(x,cutpoint3,c31,c32)

print("regression output: ", pred_y)
print("cutting points: ",cutpoints)
print("c1: ",c1s)
print("c2: ",c2s)
