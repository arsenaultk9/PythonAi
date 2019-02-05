# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:51:06 2019

@author: KEARS4
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(size=40)
y = np.random.normal(size=40)

plt.title("Initiale data")
plt.plot(x, y, 'ro')
plt.axis([-5, 5, -5, 5])
plt.xlabel('x')
plt.ylabel('y')

plt.axvline()
plt.axhline()
plt.show()

plt.title("Multiply x weights")
plt.plot(x * 2, y, 'ro')
plt.axis([-5, 5, -5, 5])
plt.axvline()
plt.axhline()
plt.show()

plt.title("Multiply y weights")
plt.plot(x, y * 2, 'ro')
plt.axis([-5, 5, -5, 5])
plt.axvline()
plt.axhline()
plt.show()

plt.title("Multiply weights")
plt.plot(x * 2, y * 2, 'ro')
plt.axis([-5, 5, -5, 5])
plt.axvline()
plt.axhline()
plt.show()

plt.title("bias")
plt.plot(x + 1, y + 1, 'ro')
plt.axis([-5, 5, -5, 5])
plt.axvline()
plt.axhline()
plt.show()

plt.title("weights + bias")
plt.plot(x * 1.5 + 1, y * 1.5 + 1, 'ro')
plt.axis([-5, 5, -5, 5])
plt.axvline()
plt.axhline()
plt.show()
