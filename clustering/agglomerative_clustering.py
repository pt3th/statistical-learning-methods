#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:56:15 2019

@author: q
"""
'''
python implementation for agglomerative/bottom-up clustering 
using statistics learning methods(LI Hang) Example 14.1
'''


import numpy as np
import copy


#use agglomerative clustering and return the clustering result after one iteration  
def agglomerative_clustering(D,class_list):
    l = len(class_list)
    dmin = 9999999999999999
    for i in range(l):
        for j in range(i+1,l):
            class_i = class_list[i]
            class_j = class_list[j]
            dij = distance(D,class_i,class_j)
            if dij < dmin:
                index_i = i
                index_j = j
                dmin = dij
    new_class_list = []
    for i in range(l):
        if i != index_i and i != index_j:
            new_class_list.append(class_list[i])
    cls_i = class_list[index_i]
    cls_j = class_list[index_j]
    if isinstance(cls_i,list):
        if isinstance(cls_j,list):
            new_cls = cls_i + cls_j
        else:
            cls_i.append(cls_j)
            new_cls = cls_i
    else:
        if isinstance(cls_j,list):
            cls_j.append(cls_i)
            new_cls = cls_j
        else:
            new_cls = [cls_i,cls_j]
    new_class_list.append(new_cls)
    return new_class_list
    
#return the minimum distance between class_i and class_j
def distance(D,class_i,class_j):
    dist = []
    if isinstance(class_i,list) == False:
        class_i = [class_i]
    if isinstance(class_j,list) == False:
        class_j = [class_j]
    for i in class_i:
        for j in class_j:
            dist.append(D[i-1,j-1])
    return min(dist)

#given 5 samples, the Euclidean distance between ith sample and jth sample is dij - an element of matrix D
D = np.array([[0, 7, 2, 9, 3], [7, 0, 5, 4, 6], [2, 5, 0, 8, 1], [9, 4, 8, 0, 5], [3, 6, 1, 5, 0]])

#
init_class_num = np.shape(D)[0]
final_class_num = 1

clustering_list = []
init_class = list(np.arange(1,init_class_num+1))
clustering_list.append(init_class)

for i in np.arange(0,init_class_num-final_class_num):
    #deep copy: avoid changing the value of clustering_list
    class_list =copy.deepcopy(clustering_list[i])
    new_class_list = agglomerative_clustering(D,class_list)
    clustering_list.append(new_class_list)
print('clustering process:')
print(clustering_list)