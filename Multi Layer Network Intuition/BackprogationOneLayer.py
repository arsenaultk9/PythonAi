import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

np.random.seed(420)


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

learning_rate = 0.5

w1 = np.random.rand(8, 1)

for iteration in range(6000):
    cost = 0
    y_hats = []

    for index in range(len(X)):
        product_w1 = np.dot(X[index], w1)
        activation_w1 = sigmoid(product_w1)

        current_y_hat = activation_w1
        y_hats.append(current_y_hat)
        current_y = Y[index]

        cost += cost_function(current_y, current_y_hat) / X.shape[0]

        current_y_hat_derivative = cost_derivative(
            current_y, current_y_hat) / X.shape[0]

        # W1 update
        current_activation_derivative = current_y_hat_derivative * \
            sigmoid_derivative(product_w1)

        current_derivative_w1 = current_activation_derivative * X[index]
        current_derivative_w1 = current_derivative_w1.reshape(8, 1)

        # current_derivative_w1 = current_derivative_w1 * activation_w1
        # current_derivative_w1 = current_derivative_w1.reshape(8, 1)
        w1 = w1 - learning_rate * current_derivative_w1

    print('cost: ', cost)
    print('y: ', Y[0])
    print('y_hat: ', y_hats[0])
