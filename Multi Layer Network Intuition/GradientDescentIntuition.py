import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def cost_function(y, y_hat):
    return ((y_hat - y)**2)

y_index = 1
Y = np.array([-1, 0, 2])

# Gradiant graph in respect to the cost function
normal_dist = np.random.uniform(-5, 5, 400)
normal_dist = np.sort(normal_dist)

total_dist_cost = []
normal_dist_costs = np.zeros((Y.shape[0], normal_dist.shape[0]))

for x_index in range(len(normal_dist)):
    total_cost = 0

    for y_index in range(len(Y)):
        current_cost = cost_function(Y[y_index], normal_dist[x_index])
        total_cost += current_cost / Y.shape[0]
        normal_dist_costs[y_index][x_index] = current_cost

        
    total_dist_cost.append(total_cost)

plt.plot(normal_dist, total_dist_cost)

for normal_dist_cost in normal_dist_costs:
    plt.plot(normal_dist, normal_dist_cost)

plt.show()