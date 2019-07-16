import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import preprocessing

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
X = preprocessing.scale(X)

learning_rate = 0.05

w1 = np.random.rand(9, 4)
w2 = np.random.rand(5, 1)

for iteration in range(9000):
    cost = 0
    y_hats = []

    for index in range(len(X)):
        current_x = np.r_[X[index], 1]  # Add bias
        product_w1 = np.dot(current_x, w1)
        activation_w1 = np.array(list(map(sigmoid, product_w1)))
        activation_w1 = np.r_[activation_w1, 1]  # Add bias

        product_w2 = np.dot(activation_w1, w2)
        activation_w2 = np.array(list(map(sigmoid, product_w2)))

        current_y_hat = activation_w2[0]
        y_hats.append(current_y_hat)
        current_y = Y[index]

        cost += cost_function(current_y, current_y_hat) / X.shape[0]

        current_y_hat_derivative = cost_derivative(
            current_y, current_y_hat) / X.shape[0]

        # W2 update
        current_derivative_w2_activation = current_y_hat_derivative * \
            sigmoid_derivative(product_w2)

        current_derivative_w2 = current_derivative_w2_activation * activation_w1
        current_derivative_w2 = np.array(current_derivative_w2).reshape(5, 1)

        w2 = w2 - learning_rate * current_derivative_w2

        # W1 update
        current_derivative_w1_activation = current_derivative_w2[0:4] * \
            np.array(list(map(sigmoid_derivative, product_w1))).reshape(4, 1)

        current_derivative_w1 = np.dot(current_x.reshape(9, 1),
                                       current_derivative_w1_activation.reshape(1, 4))

        w1 = w1 - learning_rate * current_derivative_w1

    if(iteration % 100 == 0):
        print('iteration: ', iteration)
        print('cost: ', cost)
        print('y: ', Y[0:5])
        print('y_hat: ', y_hats[0:5])
