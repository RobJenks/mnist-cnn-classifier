import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit


@np.vectorize
def sigmoid_logistic_direct(x):
    return 1. / (1. + (np.e ** -x))


def sigmoid_logistic(x):
    return expit(x)


def truncated_norm(mean=0, sd=1, low=0, high=10):
    return truncnorm((low - mean) / sd,
                     (high - mean) / sd,
                     loc=mean,
                     scale=sd)
