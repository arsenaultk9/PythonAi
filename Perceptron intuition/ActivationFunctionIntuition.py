# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:22:12 2019

@author: KEARS4
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:51:06 2019

@author: KEARS4
"""

import matplotlib.pyplot as plt
import numpy as np
import functions

x = np.random.normal(size=40)
y = np.random.normal(size=40)

activatedX = []
activatedY = []

notActivatedX = []
notActivatedY = []

for index in range(40):
    if(x[index] + y[index] >= 0):
        activatedX.append(x[index])
        activatedY.append(y[index])
        
    else:
        notActivatedX.append(x[index])
        notActivatedY.append(y[index])
    
linearLine = functions.negative_linear_func(10);

plt.title("Initiale data")
plt.plot(activatedX, activatedY, 'ro', color='blue')
plt.plot(notActivatedX, notActivatedY, 'ro', color='red')
plt.plot(linearLine[0], linearLine[1])
plt.axis([-5, 5, -5, 5])
plt.xlabel('x')
plt.ylabel('y')

plt.axvline()
plt.axhline()
plt.show()