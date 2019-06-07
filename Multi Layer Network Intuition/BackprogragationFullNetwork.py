import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

np.random.seed(42)


def relu(val):
    return np.min(val, 0)


def relu_derivative(val):
    return 1 if val > 0 else 0


def sigmoid(val):
    return 1 / (1 + math.exp(-val))


def sigmoid_derivative(val):
    return sigmoid(val) * (1 - sigmoid(val))


def cost_function(y, y_hat):
    return ((y_hat - y)**2)


def cost_derivative(y, y_hat):
    return 2 * (y_hat - y)


dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# normalize X
X = X / np.linalg.norm(X)

learning_rate = 0.05
x = 0.5
y = 1

w1 = np.random.rand(8, 1)
w2 = np.random.rand(1, 1)

product_w1 = np.dot(X, w1)
activation_w1 = np.array(list(map(sigmoid, product_w1)))
activation_w2 = np.array(list(map(sigmoid, (activation_w1 * w2)[0])))

cost = np.array(list(map(cost_function, Y, activation_w2)))

for index in range(23):
    activation_w1 = sigmoid(x*w1)
    y_hat = sigmoid(activation_w1*w2)

    y_hat_derivative = cost_derivative(y, y_hat)

    derivative_w2 = y_hat_derivative * sigmoid_derivative(activation_w1)
    derivative_w1 = derivative_w2 * sigmoid_derivative(x*w1)

    print('y_hat :', y_hat)
    print('w1 :', w1)
    print('w2 :', w2)
    print('activation_w1 :', activation_w1)
    print('cost :', cost_function(y, y_hat))
    print('cost derivative:', y_hat_derivative)
    print('derivative_w2 :', derivative_w2)
    print('derivative_w1 :', derivative_w1)

    print('')
    print('===========================================')
    print('')

    w1 = w1 - learning_rate * derivative_w1
    w2 = w2 - learning_rate * derivative_w2

# # Gradiant graph in respect to the cost function
# normal_dist = np.random.uniform(-3, 3, 400)
# normal_dist = np.sort(normal_dist)

# normal_dist_cost = []
# for normal_dist_point in normal_dist:
#     current_cost = cost_function(y, normal_dist_point)
#     normal_dist_cost.append(current_cost)

# plt.plot(normal_dist, normal_dist_cost, 'ro')
# plt.show()
