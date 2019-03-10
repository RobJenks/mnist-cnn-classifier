import numpy as np
import mnist, functions


class NeuralNetwork:

    def __init__(self, input_node_count, output_node_count, hidden_node_count,
                 learning_rate, activation_fn):
        self.in_node_count = input_node_count
        self.out_node_count = output_node_count
        self.hidden_node_count = hidden_node_count
        self.learn_rate = learning_rate
        self.activation_fn = activation_fn
        self.weight_ih, self.weight_ho = None, None

        self.initialise_weight_matrices()

    # Build initial weight matrices based around truncated normal distributions
    def initialise_weight_matrices(self):
        # Generate randomised in->hidden weights from a truncated normal dist
        nr = 1. / np.sqrt(self.in_node_count)
        trunc_norm = functions.truncated_norm(mean=0, sd=1, low=-nr, high=nr)
        self.weight_ih = trunc_norm.rvs((self.hidden_node_count, self.in_node_count))

        # Generate randomised hidden->out weights from an equivalent distribution
        nr = 1. / np.sqrt(self.hidden_node_count)
        trunc_norm = functions.truncated_norm(mean=0, sd=1, low=-nr, high=nr)
        self.weight_ho = trunc_norm.rvs((self.out_node_count, self.hidden_node_count))

    # Execute training and adjust network weights for the given set of training data, for a set number of epochs
    def train(self, epochs, input_data, target_labels_vec):
        for epoch in range(epochs):
            for x in zip(input_data, target_labels_vec):
                self.train_item(*x)

    # Evaluate and adjust network weights based upon the provided training item
    def train_item(self, input_vec, target_vec):
        input_vec, target_vec = (np.array(x, ndmin=2).T for x in [input_vec, target_vec])

        # Input -> hidden layer
        out_vec = np.dot(self.weight_ih, input_vec)
        output_hidden = self.activation_fn(out_vec)

        # Hidden layer -> output
        out_vec = np.dot(self.weight_ho, output_hidden)
        output_network = self.activation_fn(out_vec)

        # Calculate and adjust for output errors
        output_errors = target_vec - output_network
        adjustment = np.dot(output_errors * output_network * (1.0 - output_network), output_hidden.T)
        adjustment *= self.learn_rate

        self.weight_ho += adjustment

        # Calculate and adjust for hidden layer errors
        hidden_errors = np.dot(self.weight_ho.T, output_errors)
        adjustment = np.dot(hidden_errors * output_hidden * (1.0 - output_hidden), input_vec.T)
        adjustment *= self.learn_rate

        self.weight_ih += adjustment

    # Evaluate the given input data against the network
    def execute(self, input_vec):
        input_vec = np.array(input_vec, ndmin=2).T

        # Process hidden layer
        result = self.activation_fn(
            np.dot(self.weight_ih, input_vec)
        )

        # Process output layer
        result = self.activation_fn(
            np.dot(self.weight_ho, result)
        )

        return result

    # Evaluate the given data and calculate a confusion matrix
    def confusion_matrix(self, data, labels):
        conf = np.zeros((mnist.label_count(), mnist.label_count()), int)

        # Evaluate against the network
        for i, x in enumerate(data):
            result = self.execute(x)
            highest_confidence_prediction = result.argmax()

            # Plot the intersection of { classifier_result, actual_label } in the CM
            actual = int(labels[i][0])
            conf[highest_confidence_prediction, actual] += 1

        return conf

    # Returns precision of the network predictions for a given label, based on the provided conf matrix
    def precision(self, label, confusion_matrix):
        # (correct predictions / total predictions) for the label
        column = confusion_matrix[:, label]
        return confusion_matrix[label, label] / column.sum()

    # Returns recall of the network predictions for a given label, based on the provided conf matrix
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()
    
    # Execute the network and assess performance against the given label set.  Returns (passes, fails)
    def evaluate(self, data, labels):
        matches = sum(self.execute(x) == labels[i] for i, x in enumerate(data))
        return matches, len(data)-matches
















