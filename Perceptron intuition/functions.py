# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:06:21 2019

@author: KEARS4
"""

def negative_linear_func(funcRange):
    linearX = []
    linearY = []
    
    for index in range(funcRange + 1):
        linearX.append(index - funcRange/2)
        linearY.append(index *-1 + funcRange/2)
    
    return (linearX, linearY)