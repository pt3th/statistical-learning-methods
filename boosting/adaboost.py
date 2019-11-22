#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:15:57 2019

@author: q
"""

'''
python implementation for statistics learning methods(LI Hang) Example 8.1 page 140
'''
import numpy as np

#basic classifer
#direc indicates different classification method
#direc = 1, x < threshold => 1; x > threshold => -1
#direc = -1, x < threshold => -1; x > threshold => 1
def basic_cls(x,threshold,direc = 1):
    return -2*(((x>threshold)*1)-0.5)*direc

def cls_error(cls_output,true_label,weight):
    error = np.sum((cls_output!=true_label)*1*weight)
    return error

def calc_alpha(error):
    return np.log((1-error)/error)/2

def update_weight(D_prev,alpha_prev,true_label,cls_output_prev):
    tmp = D_prev*np.exp(true_label * cls_output_prev *(-alpha_prev))
    return tmp/np.sum(tmp)

def get_threshold(D,x,true_label,step):
    best_error = 10
    best_threshold = 999
    best_direc = 3
    for threshold in np.arange((x[0]+x[1])/2,x[-1],step):
        cls_output1 = basic_cls(x,threshold,1)
        cls_output2 = basic_cls(x,threshold,-1)
        current_error1 = cls_error(cls_output1,true_label,D)
        current_error2 = cls_error(cls_output2,true_label,D)
        current_error = np.min([current_error1,current_error2])
        index = np.argmin([current_error1,current_error2])
        current_direc = [1,-1][index]
        if current_error < best_error:
            best_threshold = threshold
            best_error = current_error
            best_direc = current_direc
    return best_threshold,best_direc

#training set
N = 10
x = np.arange(10)
true_label = np.array([1,1,1,-1,-1,-1,1,1,1,-1])

weight_list = []
threshold_list = []
direc_list = []
output_list = []
alpha_list = []
#initialization
W1 = np.ones((N,))/N
pred = np.zeros(N,)
weight_list.append(W1)

M = 5
step = 1
for i in range(M):
    threshold,direc = get_threshold(weight_list[i],x,true_label,step)
    threshold_list.append(threshold)
    direc_list.append(direc)
    output = basic_cls(x,threshold_list[i],direc_list[i])
    output_list.append(output)
    error = cls_error(output_list[i],true_label,weight_list[i])
    alpha = calc_alpha(error)
    alpha_list.append(alpha)
    W =  update_weight(weight_list[i],alpha_list[i],true_label,output_list[i])
    weight_list.append(W)
    for j in range(len(threshold_list)):
        pred += alpha_list[j]*basic_cls(x,threshold_list[j],direc_list[j])
    final_pred = np.sign(pred)
    pred_error = cls_error(final_pred,true_label,W1)
    if pred_error ==0:
        print("perfect classifier found!")
        break

print("alpha: ",alpha_list)
print("threshold: ",threshold_list)
print("direc: ", direc_list)