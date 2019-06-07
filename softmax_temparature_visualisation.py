# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def plot(data, title):
    plt.plot(data)
    plt.title(title)
    plt.show()


normal_distribution = np.sort(np.random.random(100))

log_temparature_distributions = []
exponential_distributions = []
normalized_distributions = []
probability_distributions = []

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print('----- diversity:', diversity)
    plot(normal_distribution, 'normal distribution')

    log_temparature_distribution = np.log(normal_distribution) / diversity
    log_temparature_distributions.append(log_temparature_distribution)
    plot(log_temparature_distribution, 'log temperature distribution')

    exponential_distribution = np.exp(log_temparature_distribution)
    exponential_distributions.append(exponential_distribution)
    plot(exponential_distribution, 'exponential distribution')

    normalized_distribution = exponential_distribution / \
        np.sum(exponential_distribution)
    normalized_distributions.append(normalized_distribution)
    plot(normalized_distribution, 'normalized distribution')

    probability_distribution = np.random.multinomial(
        1, normalized_distribution, 1)
    probability_distributions.append(probability_distribution[0])


log_temparature_distributions = np.swapaxes(
    np.array(log_temparature_distributions), 0, 1)

exponential_distributions = np.swapaxes(
    np.array(exponential_distributions), 0, 1)

normalized_distributions = np.swapaxes(
    np.array(normalized_distributions), 0, 1)

probability_distributions = np.array(probability_distributions)
probability_distributions = np.swapaxes(
    np.array(probability_distributions), 0, 1)
