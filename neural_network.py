import numpy as np
import functions


class NeuralNetwork:

    def __init__(self, input_node_count, output_node_count, hidden_node_count,
                 learning_rate, activation_fn):
        self.in_node_count = input_node_count
        self.out_node_count = output_node_count
        self.hidden_node_count = hidden_node_count
        self.learn_rate = learning_rate
        self.activation_fn = activation_fn
        self.weight_ih, self.weight_ho = [], []

    def initialise_weight_matrices(self):
        # Generate randomised in->hidden weights from a truncated normal dist
        nr = 1. / np.sqrt(self.in_node_count)
        trunc_norm = functions.truncated_norm(mean=0, sd=1, low=-nr, high=nr)
        self.weight_ih = trunc_norm.rvs((self.hidden_node_count, self.in_node_count))

        # Generate randomised hidden->out weights from an equivalent distribution
        nr = 1. / np.sqrt(self.hidden_node_count)
        trunc_norm = functions.truncated_norm(mean=0, sd=1, low=-nr, high=nr)
        self.weight_ho = trunc_norm.rvs((self.out_node_count, self.hidden_node_count))


