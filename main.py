import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import json
import mnist
import functions
from neural_network import NeuralNetwork


def main():
    execute(
        mnist.Type.Extended,
        'load-network' in sys.argv,
        'save-network' in sys.argv,
        'skip-stats' in sys.argv
    )


def execute(t: mnist.Type, load_network: bool, save_network: bool, skip_stats: bool):
    if not mnist.binary_data_available(t):
        print("Generating binary dataset on first-run for faster read times; this is a one-time activity...")
        mnist.generate_binary_data(t)

    # Retrieve and normalise mnist data
    print("Initialising network and datasets...")
    train_data, train_labels = mnist.labelled_training_data(t)
    test_data, test_labels = mnist.labelled_test_data(t)
    label_mapping, reverse_label_mapping = mnist.get_label_mappings(t)

    # Use one-hot label representation for CNN classification
    train_labels_vec = one_hot_labels(train_labels, mnist.label_count(t))

    # Either generate and train the network, or load a saved binary network state
    if load_network:
        print("Loading stored network state")
        network = NeuralNetwork.load_network_state("data/network-state.json")
    else:
        config = get_network_configuration(t)
        print(f"Generating network (config={config})")
        network = NeuralNetwork(input_node_count=config[0],
                                hidden_layers=config[1:-1],
                                output_node_count=config[-1],
                                learning_rate=0.1,
                                bias=1,
                                activation_fn=functions.sigmoid_logistic)

        print("Training network...")
        network.train(train_data, train_labels_vec, epochs=1)

    if not skip_stats:
        print("\nSample test predictions with confidence:")
        for i in range(20):
            result = network.execute(test_data[i])
            print(int(test_labels[i][0]), np.argmax(result), np.max(result))

        for x in [("training", train_data, train_labels), ("test", test_data, test_labels)]:
            matches, fails = network.evaluate(*x[1:3])
            print("Accuracy over {} data: {}".format(x[0], matches / (matches + fails)))

        conf = network.confusion_matrix(train_data, train_labels, mnist.label_count(t))
        print("\nNetwork confusion matrix:")
        print(conf)

        for i in range(mnist.label_count(t)):
            print(f"Data item {i}: Precision = {network.precision(i, conf)}, Recall = {network.recall(i, conf)}")

    if save_network:
        print("Saving network state...")
        network.save_network_state("data/network-state.json")

    # Render sample of classified results
    render_classified_data_sample(network, test_data, 20, t)

    print("\nClassifying test data...")
    classified_data = network.classify_data(test_data, reverse_label_mapping)

    render_message("This is a test message\nAnd so is this", classified_data, t)


# Return a network layer configuration for each target dataset type
def get_network_configuration(t: mnist.Type):
    config = {
        mnist.Type.Modified: [mnist.image_size(), 100, mnist.label_count(t)],
        mnist.Type.Extended: [mnist.image_size(), 360, mnist.label_count(t)]
    }

    return config[t]


# One-hot representation for labels within the given value range
def one_hot_labels(labels, label_range: int):
    rng = np.arange(label_range)
    return [[0.99 if x == 1 else 0.01 for x in (rng == label).astype(np.float)] for label in labels]


# Apply a label mapping from ( * -> [0 n) )
def apply_label_mapping(labels, label_mapping):
    for i in range(len(labels)):
        labels[i][0] = label_mapping[int(labels[i][0])]


def render_classified_data_sample(network, data, sample_n, t: mnist.Type):
    data_size = len(data)
    size = mnist.image_dimensions()

    dim = (mnist.label_count(t) * size, sample_n * size)
    flip_canvas = (mnist.label_count(t) > sample_n)
    img = np.zeros(shape=tuple(reversed(dim)) if flip_canvas else dim)

    ix, rendered = 0, [0] * mnist.label_count(t)
    while any(x != sample_n for x in rendered) and ix != data_size:
        prediction = network.execute(data[ix]).argmax()
        if rendered[prediction] != sample_n:
            px, py = prediction*size, rendered[prediction]*size
            if flip_canvas:
                px, py = py, px

            if mnist.transpose_output(t):
                img[px:px+size, py:py+size] = np.transpose(data[ix].reshape(size, size))
            else:
                img[px:px+size, py:py+size] = data[ix].reshape(size, size)

            rendered[prediction] += 1

        ix += 1

    plt.imshow(img, cmap="Greys")
    plt.show()


def render_message(msg: str, classified_data, t: mnist.Type):
    size = mnist.image_dimensions()
    msg = msg.upper()

    allowed_chars = set(x for x in [*classified_data, ord(' '), ord('\n')])
    if any(ord(x) not in allowed_chars for x in msg):
        print(f"Cannot render message \"{msg}\"; dataset does not include all required characters")
        print(f"Disallowed chars: {set(x for x in msg if ord(x) not in allowed_chars)}")
        return

    space_size = int(float(size) * 0.6)
    line_count = 1 + sum((1 if x == '\n' else 0) for x in msg)
    longest_line_size = longest_line_length(msg, size, space_size)

    img = np.zeros(shape=((line_count*size), longest_line_size))
    transform = np.transpose if mnist.transpose_output(t) else lambda x: x
    px, py = 0, 0
    for x in msg:
        if x == ' ':
            px += space_size
        elif x == '\n':
            py += size
            px = 0
        else:
            img[py:py+size, px:px+size] = transform(random.choice(classified_data.get(ord(x))).reshape(size, size))
            px += size

    plt.imshow(img, cmap="Greys")
    plt.show()


def show_image(data):
    img = data.reshape((mnist.image_dimensions(), mnist.image_dimensions()))
    plt.imshow(img, cmap="Greys")
    plt.show()


# Returns the length of the longest line in the supplied string, given the respective pixel widths of each char type
def longest_line_length(s: str, letter_width, space_width) -> int:
    longest, length = 0, 0

    for x in s:
        if x == '\n':
            longest = max(longest, length)
            length = 0
        elif x == ' ':
            length += space_width
        else:
            length += letter_width

    return max(longest, length)


if __name__ == "__main__":
    main()
