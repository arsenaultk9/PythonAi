# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:28:09 2019

@author: KEARS4
"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(6)

learningRate = 0.005
space = np.linspace(-1, 1)

X = np.array([
     [ 0.72, 0.82 ], [ 0.46, 0.98 ], [ 0.46, 0.80 ],
     [ 0.03, 0.93 ], [ 0.12, 0.25 ], [ -0.16, 0.84 ], 
     
     [ 0.79, -0.75 ], [ 0.91, -0.69 ], [ 0.66, 0.24 ], 
     [ 0.72, -0.15 ], [ 0.35, 0.01 ], [ 0.96, -0.47 ],
     
     [ -0.79, -0.92 ], [ -0.31, -0.96 ], [ -0.47, -0.88 ],
     [ -0.25, -0.46 ], [ -0.43, -0.65 ], [ -0.57, -0.97 ], 
     
     [ -0.47, -0.03 ], [ -0.72, 0.64 ], [ -0.57, 0.15 ], 
     [ -0.25, 0.43 ], [ -0.11, 0.10 ], [ -0.47, 0.45 ]])

y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])

w = np.array([np.random.normal(), np.random.normal()])

def activation_func(z):
    return 1 if z >= 0 else -1

#def plot_data():
#    plt.plot(X[0:12, 0] * w[0], X[0:12, 1] * w[1], 'ro')
#    plt.plot(X[13:29, 0] * w[0], X[13:29, 1] * w[1], 'bo')
#    
#    plt.plot(space, -space)
#    plt.axis([-1, 1, -1, 1])
#    plt.xlabel('x')
#    plt.ylabel('y')
#    
#    plt.axvline()
#    plt.axhline()
#    plt.show()
    
def plot_data_for_linear():
    plt.plot(X[0:6, 0], X[0:6, 1], 'ro')
    plt.plot(X[6:12, 0], X[6:12, 1], 'go')
    plt.plot(X[12:18, 0], X[12:18, 1], 'bo')
    plt.plot(X[18:24, 0], X[18:24, 1], 'mo')
    
    plt.plot(space, -space)
    plt.axis([-1, 1, -1, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.plot(space, -space * (w[0]/w[1]))
    
    plt.axvline()
    plt.axhline()
    plt.show()

plot_data_for_linear()

#run_errors = []
#for iteration in range(8):
#    y_hat = np.array(list(map(activation_func, X.dot(w))))
#    errors = y - y_hat
#    global_error = np.absolute(errors).sum()
#    run_errors.append(global_error)
#    
#    plot_data()
#    plot_data_for_linear()
#
#    print('global error: {}', global_error)
#    print('w: {}', w)
#    for input_index in range(errors.size):
#        local_error = errors[input_index]
#        x_cur = X[input_index]
#        
#        for w_index in range(w.size):
#            w += learningRate*local_error*x_cur
#
## plot error            
#plt.plot(run_errors)
#plt.xlabel('iteration')
#plt.ylabel('error')
#plt.show()
    