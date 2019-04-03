# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:28:09 2019

@author: KEARS4
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

np.random.seed(6)

learningRate = 0.05
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

# TODO: Check if shape of array [2, 4] vs [4, 2] would resolve in better dot product (maybe not).
y = np.array([
        [1, -1, -1, -1], [1, -1, -1, -1], [1, -1, -1, -1],
        [1, -1, -1, -1], [1, -1, -1, -1], [1, -1, -1, -1],
        
        [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1],
        [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1],
        
        [-1, -1, 1, -1], [-1, -1, 1, -1], [-1, -1, 1, -1],
        [-1, -1, 1, -1], [-1, -1, 1, -1], [-1, -1, 1, -1],
        
        [-1, -1, -1, 1], [-1, -1, -1, 1], [-1, -1, -1, 1],
        [-1, -1, -1, 1], [-1, -1, -1, 1], [-1, -1, -1, 1]])

w = np.array([
        [np.random.normal(), np.random.normal()], 
        [np.random.normal(), np.random.normal()],
        [np.random.normal(), np.random.normal()],
        [np.random.normal(), np.random.normal()]
        ])

def activation_func(z):
    return 1 if z >= 0 else -1

def plot_data():  
    plt.plot(X[0:6, 0] * w[0, 0], X[0:6, 1]  * w[0, 1], 'ro')
    plt.plot(X[6:12, 0] * w[1, 0], X[6:12, 1]  * w[1, 1], 'go')
    plt.plot(X[12:18, 0] * w[2, 0], X[12:18, 1] * w[2, 1], 'bo')
    plt.plot(X[18:24, 0] * w[3, 0], X[18:24, 1]  * w[3, 1], 'mo')
    
    plt.plot(space, -space)
    plt.axis([-1, 1, -1, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.axvline()
    plt.axhline()
    plt.show()
    
def plot_data_for_linear():
    plt.plot(X[0:6, 0], X[0:6, 1], 'ro')
    plt.plot(X[6:12, 0], X[6:12, 1], 'go')
    plt.plot(X[12:18, 0], X[12:18, 1], 'bo')
    plt.plot(X[18:24, 0], X[18:24, 1], 'mo')
    
    plt.axis([-1, 1, -1, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    
    for current_w in w:
        plt.plot(space, -space * (current_w[0]/current_w[1]))
    
    plt.axvline()
    plt.axhline()
    plt.show()

run_errors = []

for iteration in range(15):
    errors = 0
    
    dot_products = [];
    for current_node in range(w.shape[0]):
        current_dot = X.dot(w[current_node])
        dot_products.append(current_dot)
    
    np_dot_products = np.array(dot_products).T
    plot_data_for_linear()
    plot_data()
    
    max_vals = np.max(np_dot_products, 1)
    activated_classes = []
    for x_col in range(np_dot_products.shape[0]):
        activated_classes_row = []
        
        for y_col in range(np_dot_products.shape[1]):
            currentval = 1 if np_dot_products[x_col, y_col] == max_vals[x_col] else 0
            activated_classes_row.append(currentval)
        
        activated_classes.append(activated_classes_row)
    
    activated_classes = np.array(activated_classes)
    
    for row in range(y.shape[0]):
        for column in range(y.shape[1]):
            error = y[row, column] - activated_classes[row, column]
            errors += error;
            
            w_current_node = w[column]
            for w_index in range(w_current_node.size):
                w_current_node += learningRate*error*X[row]
    
    run_errors.append(errors)
            
plot_data_for_linear()
plot_data()

# plot error            
plt.plot(run_errors)
plt.xlabel('iteration')
plt.ylabel('error')
plt.show()

#for iteration in range(1):
#    for current_node in range(w.shape[0]):  
#        current_activation = list(map(activation_func, X.dot(w[current_node])))
#        y_hat = np.array(current_activation)
#        errors = y[:,current_node] - y_hat
#        global_error = np.absolute(errors).sum()
#        run_errors.append(global_error)
#        
#        plot_data_for_linear()
#        plot_data()
#    
#        print('global error: {}', global_error)
#        print('w: {}', w)
#        
#        w_current_node = w[current_node]
#        for input_index in range(errors.size):
#            local_error = errors[input_index]
#            x_cur = X[input_index]
#            
#            for w_index in range(w_current_node.size):
#                w_current_node += learningRate*local_error*x_cur
#
## plot error            
#plt.plot(run_errors)
#plt.xlabel('iteration')
#plt.ylabel('error')
#plt.show()
#    