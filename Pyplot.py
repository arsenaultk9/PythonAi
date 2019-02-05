# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:00:01 2019

@author: KEARS4
"""

import matplotlib.pyplot as plt
plt.plot([1, 4, 9, 4], [1, 4, 9, 16])
plt.ylabel('some numbers')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(-0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()