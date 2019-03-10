import numpy as np
from scipy.stats import truncnorm


@np.vectorize
def sigmoid(x):
    return 1. / (1. + (np.e ** -x))


def truncated_norm(mean=0, sd=1, low=0, high=10):
    return truncnorm((low - mean) / sd,
                     (high - mean) / sd,
                     loc=mean,
                     scale=sd)
