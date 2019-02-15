# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:41:11 2019

@author: KEARS4
"""

import matplotlib.pyplot as plt
import numpy as np

def relu(Z):
    return np.maximum(0,Z)

def negative_linear_func(funcRange):
    linearX = []
    linearY = []
    
    for index in range(funcRange + 1):
        linearX.append(index - funcRange/2)
        linearY.append(index *-1 + funcRange/2)
    
    return (linearX, linearY)

linearLine = functions.negative_linear_func(10);

fx = np.linspace(-5, 5, 15)
fy = np.array(np.linspace(-5, 5, 15))
fy = np.array(list(map(relu, fy)))

np.random.seed(40)
x = np.random.normal(size=5)
y = np.random.normal(size=5)

z = np.array(list(map(relu, x+y+1))) # Intéressant regarde le graph.

plt.title("Initiale data 1x")
plt.plot(x, y, 'ro')
plt.plot(fx, fy)
plt.plot(linearLine[0], linearLine[1])
plt.axis([-5, 5, -5, 5])
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, z, 'bo')

plt.axvline()
plt.axhline()
plt.show()