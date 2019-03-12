import numpy as np
import matplotlib.pyplot as plt
import mnist
import functions
from neural_network import NeuralNetwork


def main():
    execute(mnist.Type.Modified)


def execute(t: mnist.Type):
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

    config = get_network_configuration(t)
    network = NeuralNetwork(input_node_count=config[0],
                            hidden_layers=config[1:-1],
                            output_node_count=config[-1],
                            learning_rate=0.1,
                            bias=1,
                            activation_fn=functions.sigmoid_logistic)

    print("Training network...")
    network.train(train_data, train_labels_vec, epochs=1)

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

    render_classified_data_sample(network, test_data, 20, mnist.label_count(t))


# Return a network layer configuration for each target dataset type
def get_network_configuration(t: mnist.Type):
    config = {
        mnist.Type.Modified: [mnist.image_size(), 100, mnist.label_count(t)],
        mnist.Type.Extended: [mnist.image_size(), 100, mnist.label_count(t)]
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


def render_classified_data_sample(network, data, sample_n, label_count):
    img = np.zeros(shape=(
        label_count * mnist.image_dimensions(),
        sample_n * mnist.image_dimensions()))

    ix, rendered = 0, [0] * label_count
    while any(x != sample_n for x in rendered):
        prediction = network.execute(data[ix]).argmax()
        if rendered[prediction] != sample_n:
            img[prediction*mnist.image_dimensions():prediction*mnist.image_dimensions() + mnist.image_dimensions(),
                rendered[prediction]*mnist.image_dimensions():rendered[prediction]*mnist.image_dimensions() + mnist.image_dimensions()
            ] = data[ix].reshape(mnist.image_dimensions(), mnist.image_dimensions())

            rendered[prediction] += 1

        ix += 1

    plt.imshow(img, cmap="Greys")
    plt.show()


def show_image(data):
    img = data.reshape((mnist.image_dimensions(), mnist.image_dimensions()))
    plt.imshow(img, cmap="Greys")
    plt.show()


if __name__ == "__main__":
    main()
