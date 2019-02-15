# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:41:52 2019

@author: KEARS4
"""

import matplotlib.pyplot as plt
import numpy as np

zx = np.linspace(-5, 5, 10)
zy = np.linspace(-5, 5, 10)

x = np.random.normal(size=40)
y = np.random.normal(size=40)

plt.title("Initiale data 1x")
plt.plot(x, y, 'ro')
plt.plot(zx, zy)
plt.axis([-5, 5, -5, 5])
plt.xlabel('x')
plt.ylabel('y')

plt.axvline()
plt.axhline()
plt.show()

plt.title("Initiale data 2x no axis ajustement")
plt.plot(x, y, 'ro')
plt.plot(zx, zy*2)
plt.axis([-5, 5, -5, 5])
plt.xlabel('x')
plt.ylabel('y')

plt.axvline()
plt.axhline()
plt.show()


plt.title("Initiale data 2x with axis ajustement")
plt.plot(x, y, 'ro')
plt.plot(zx, zy*2)
plt.axis([-5, 5, -10, 10])
plt.xlabel('x')
plt.ylabel('y')

plt.axvline()
plt.axhline()
plt.show()

