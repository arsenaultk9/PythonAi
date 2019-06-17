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

w1 = np.random.rand(8, 8)
w2 = np.random.rand(8, 1)

for iteration in range(600):
    cost = 0

    for index in range(len(X)):
        product_w1 = np.dot(X[index], w1)
        activation_w1 = np.array(list(map(sigmoid, product_w1)))

        product_w2 = np.dot(activation_w1, w2)
        activation_w2 = np.array(list(map(sigmoid, product_w2)))

        current_y_hat = activation_w2[0]
        current_y = Y[index]

        cost += cost_function(current_y, current_y_hat)

        current_y_hat_derivative = cost_derivative(
            current_y, current_y_hat) / X.shape[0]

        current_derivative_w2 = current_y_hat_derivative * \
            sigmoid_derivative(activation_w2[0])

        current_derivative_w1 = current_derivative_w2 * \
            sigmoid_derivative(activation_w1[0])

        w1 = w1 - learning_rate * current_derivative_w1
        w2 = w2 - learning_rate * current_derivative_w2

    print('cost: ', cost)

# product_w1 = np.dot(X, w1)
# activation_w1 = np.array(list(map(sigmoid, product_w1)))
# activation_w2 = np.array(list(map(sigmoid, (activation_w1 * w2)[0])))


# # Gradiant graph in respect to the cost function
# normal_dist = np.random.uniform(-3, 3, 400)
# normal_dist = np.sort(normal_dist)

# normal_dist_cost = []
# for normal_dist_point in normal_dist:
#     current_cost = cost_function(y, normal_dist_point)
#     normal_dist_cost.append(current_cost)

# plt.plot(normal_dist, normal_dist_cost, 'ro')
# plt.show()
