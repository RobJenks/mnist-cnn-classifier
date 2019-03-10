import numpy as np
import matplotlib.pyplot as plt
import mnist


def main():
    if not mnist.binary_data_available():
        print("Generating binary dataset on first-run for faster read times; this is a one-time activity")
        mnist.generate_binary_data()

    # Retrieve and normalise mnist data
    train_data, train_labels = mnist.labelled_training_data()
    test_data, test_labels = mnist.labelled_test_data()

    # Use one-hot label representation for CNN classification
    train_labels = one_hot_labels(train_labels, mnist.label_count())
    test_labels = one_hot_labels(test_labels, mnist.label_count())

    show_image(train_data[12])


# One-hot representation for a value within the given value range
def one_hot_labels(labels, label_range: int):
    rng = np.arange(label_range)
    return [[0.99 if x == 1 else 0.01 for x in (rng == label).astype(np.float)] for label in labels]


def show_image(data):
    img = data.reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()


if __name__ == "__main__":
    main()
