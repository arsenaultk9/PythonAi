# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:58:01 2019

@author: KEARS4
"""

import numpy as np

a = np.array([0.75,0.5]) 
b = np.array([0.64,0.75]) 
result = np.dot(a,b)

print('numpy result: ' + str(result))

manual_result = (0.75*0.64 + 0.5*0.75)
print('manual result: ' + str(manual_result))

a = np.array([0.75,0.5]) 
b = np.array(0.75) 
result = np.dot(a,b)

print('numpy result last one dimension: ' + str(result))

x = np.array( ((2,3), 
               (3, 5)) )
y = np.array( ((1,2), 
               (5, -1)) )

result = np.dot(x,y)
print('matrix dot product')
print(str(result))

result_manual =[
            [2*1 + 3*5, 2*2 + 3*-1],
            [3*1 + 5*5, 3*2 + 5*-1]
        ]
print(str(result_manual))

x = np.array([1.0, 1.0])
w = np.array([[0.75, 0.5], 
             [0.5, 0.25]])
y = np.dot(x, w)

print('perceptron dot matrix:')
print(str(y))
