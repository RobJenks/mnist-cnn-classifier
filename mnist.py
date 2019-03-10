import os.path
from typing import Tuple
import numpy as np
import pickle
import zipfile as zip


def image_dimensions() -> int:
    return 28   # Standard mnist dataset dimensions


def image_size() -> int:
    return image_dimensions() * image_dimensions()


def label_count() -> int:
    return 10   # mnist digit classification [0 9]


# Returns a tuple (normalised-data, labels) for the given dataset
def get_labelled_data(path):
    data = np.loadtxt(path, delimiter=",")
    return (
        bounded_normalise_data(np.asfarray(data[:, 1:])),
        np.asfarray(data[:, :1])
    )


def labelled_training_data():
    return read_binary("data/mnist-digit-train.binary")


def labelled_test_data():
    return read_binary("data/mnist-digit-test.binary")


def labelled_training_source_data():
    return get_labelled_data("data/mnist-digit-train.csv")


def labelled_test_source_data():
    return get_labelled_data("data/mnist-digit-test.csv")


# Normalises mnist data from 0-255 grayscale down to (0 1) EXCLUSIVE range
def bounded_normalise_data(data):
    return data / ((255 * 0.99) + 0.01)


# Generate binary representation of source data
def write_binary(data: Tuple, path: str):
    with open(path, "bw") as file:
        pickle.dump(data, file)


# Retrieve binary content from file
def read_binary(path: str) -> Tuple:
    with open(path, "br") as file:
        return pickle.load(file)


# Generate binary versions of source data
def generate_binary_data():
    # Unzip compressed raw data
    extract_compressed("data/mnist-digit-train.zip", "data")
    extract_compressed("data/mnist-digit-test.zip", "data")

    # Generate binary representations
    write_binary(labelled_training_source_data(), "data/mnist-digit-train.binary")
    write_binary(labelled_test_source_data(), "data/mnist-digit-test.binary")

    # Delete raw source data
    os.remove("data/mnist-digit-train.csv")
    os.remove("data/mnist-digit-test.csv")


# Check whether binary versions of source data currently exist
def binary_data_available() -> bool:
    return os.path.isfile("data/mnist-digit-train.binary") \
       and os.path.isfile("data/mnist-digit-test.binary")


def extract_compressed(path, target_directory):
    file = zip.ZipFile(path, 'r')
    file.extractall(target_directory)
    file.close()
