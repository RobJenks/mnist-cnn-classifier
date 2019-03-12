import numpy as np
import functions


class NeuralNetwork:

    def __init__(self, input_node_count, output_node_count, hidden_layers,
                 learning_rate, bias, activation_fn):
        self.layers = [input_node_count, *hidden_layers, output_node_count]
        self.layer_count = len(self.layers)
        self.learn_rate = learning_rate
        self.bias = bias
        self.activation_fn = activation_fn
        self.weights = []

        self.initialise_weight_matrices()

    # Build initial weight matrices based around truncated normal distributions
    def initialise_weight_matrices(self):
        bias_node = 1 if self.bias else 0
        for i in range(1, self.layer_count):
            input_nodes = self.layers[i - 1]
            output_nodes = self.layers[i]

            # Generate randomised inter-layer weights from a truncated normal dist
            n = (input_nodes + bias_node) * output_nodes
            nr = 1. / np.sqrt(input_nodes)
            trunc_norm = functions.truncated_norm(mean=0, sd=1, low=-nr, high=nr)

            weights = trunc_norm.rvs(n).reshape((output_nodes, input_nodes + bias_node))
            self.weights.append(weights)

    # Execute training and adjust network weights for the given set of training data, for a set number of epochs
    def train(self, input_data, target_labels_vec, epochs=1):
        for epoch in range(epochs):
            for x in zip(input_data, target_labels_vec):
                self.train_item(*x)

    # Evaluate and adjust network weights based upon the provided training item
    def train_item(self, input_vec, target_vec):
        input_vec, target_vec = (np.array(x, ndmin=2).T for x in [input_vec, target_vec])

        # Evaluate through each network layer
        layer_results, output = [input_vec], None
        for layer in range(self.layer_count - 1):
            if self.bias:
                layer_results[-1] = np.concatenate((layer_results[-1], [[self.bias]]))

            output = self.activation_fn(
                np.dot(self.weights[layer], layer_results[-1])
            )

            layer_results.append(output)

        # Back-propagate to tune network weights
        output_errors = target_vec - output
        for layer in range(self.layer_count - 1, 0, -1):
            layer_out = layer_results[layer]
            layer_in = layer_results[layer - 1]

            if self.bias and layer != self.layer_count-1:
                layer_out = layer_out[:-1, :].copy()

            adjustment = np.dot(output_errors * layer_out * (1.0 - layer_out), layer_in.T)
            self.weights[layer - 1] += (adjustment * self.learn_rate)

            output_errors = np.dot(self.weights[layer - 1].T, output_errors)

            if self.bias:
                output_errors = output_errors[:-1, :]

    # Evaluate the given input data against the network
    def execute(self, input_vec):
        if self.bias:
            input_vec = np.concatenate((input_vec, [self.bias]))

        input_vec = np.array(input_vec, ndmin=2).T
        output_vec = None

        # Apply each network later in turn
        for layer in range(1, self.layer_count):
            output_vec = self.activation_fn(
                np.dot(self.weights[layer - 1], input_vec)
            )

            input_vec = output_vec
            if self.bias:
                input_vec = np.concatenate((input_vec, [[self.bias]]))

        return output_vec

    # Evaluate the given data and calculate a confusion matrix
    def confusion_matrix(self, data, labels, label_count):
        conf = np.zeros((label_count, label_count), int)

        # Evaluate against the network
        for i, x in enumerate(data):
            result = self.execute(x)
            highest_confidence_prediction = result.argmax()

            # Plot the intersection of { classifier_result, actual_label } in the CM
            actual = int(labels[i][0])
            conf[highest_confidence_prediction, actual] += 1

        return conf

    # Returns precision of the network predictions for a given label, based on the provided conf matrix
    @staticmethod
    def precision(label, confusion_matrix):
        # (correct predictions / total predictions) for the label
        column = confusion_matrix[:, label]
        return confusion_matrix[label, label] / column.sum()

    # Returns recall of the network predictions for a given label, based on the provided conf matrix
    @staticmethod
    def recall(label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()
    
    # Execute the network and assess performance against the given label set.  Returns (passes, fails)
    def evaluate(self, data, labels):
        matches = sum(self.execute(x).argmax() == labels[i][0] for i, x in enumerate(data))
        return matches, len(data)-matches
