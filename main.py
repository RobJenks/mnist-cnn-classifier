import numpy as np
import matplotlib.pyplot as plt
import mnist, functions
from neural_network import NeuralNetwork


def main():
    if not mnist.binary_data_available():
        print("Generating binary dataset on first-run for faster read times; this is a one-time activity...")
        mnist.generate_binary_data()

    # Retrieve and normalise mnist data
    print("Initialising network and datasets...")
    train_data, train_labels = mnist.labelled_training_data()
    test_data, test_labels = mnist.labelled_test_data()

    # Use one-hot label representation for CNN classification
    train_labels_vec = one_hot_labels(train_labels, mnist.label_count())
    test_labels_vec = one_hot_labels(test_labels, mnist.label_count())

    network = NeuralNetwork(input_node_count=mnist.image_size(),
                            hidden_node_count=100,
                            output_node_count=10,
                            learning_rate=0.1,
                            activation_fn=functions.sigmoid)

    print("Training network...")
    for x in zip(train_data, train_labels_vec):
        network.train(*x)

    for i in range(20):
        result = network.execute(test_data[i])
        print(int(test_labels[i][0]), np.argmax(result), np.max(result))


# One-hot representation for labels within the given value range
def one_hot_labels(labels, label_range: int):
    rng = np.arange(label_range)
    return [[0.99 if x == 1 else 0.01 for x in (rng == label).astype(np.float)] for label in labels]


def show_image(data):
    img = data.reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()


if __name__ == "__main__":
    main()
