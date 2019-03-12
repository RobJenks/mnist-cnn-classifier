import os.path
from typing import Tuple, List, Dict
from enum import IntEnum
import numpy as np
import pickle
import zipfile
import util


class Type(IntEnum):
    Modified = 0
    Extended = 1


def resolve_type(typ: Type):
    return TYPES[typ][0]


def image_dimensions() -> int:
    return 28   # Standard mnist dataset dimensions


def image_size() -> int:
    return image_dimensions() * image_dimensions()


def label_count(typ: Type) -> int:
    return TYPES[typ][1]


# Return label mappings (label -> index, index -> label)
def get_label_mappings(typ: Type) -> Tuple[Dict,  List]:
    mapping_path = f"data/mnist-{resolve_type(typ)}-label-mapping.txt"

    # Generate default if no explicit mapping is provided
    if not os.path.isfile(mapping_path):
        mapping = {x: x for x in range(label_count(typ))}
    else:
        mapping = {int(y): int(x) for x, y in [line.split() for line in util.read_file(mapping_path)]}

    reverse_mapping = [-1] * label_count(typ)
    for (x, y) in mapping.items():
        reverse_mapping[y] = x

    return mapping, reverse_mapping


# Returns a tuple (normalised-data, labels) for the given dataset
def get_labelled_data(path):
    data = np.loadtxt(path, delimiter=",")
    return (
        bounded_normalise_data(np.asfarray(data[:, 1:])),
        np.asfarray(data[:, :1])
    )


def labelled_training_data(typ: Type):
    return read_binary(f"data/mnist-{resolve_type(typ)}-train.binary")


def labelled_test_data(typ: Type):
    return read_binary(f"data/mnist-{resolve_type(typ)}-test.binary")


def labelled_training_source_data(typ: Type):
    return get_labelled_data(f"data/mnist-{resolve_type(typ)}-train.csv")


def labelled_test_source_data(typ: Type):
    return get_labelled_data(f"data/mnist-{resolve_type(typ)}-test.csv")


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
def generate_binary_data(typ: Type):
    # Unzip compressed raw data
    extract_compressed(f"data/mnist-{resolve_type(typ)}-train.zip", "data")
    extract_compressed(f"data/mnist-{resolve_type(typ)}-test.zip", "data")

    # Generate binary representations
    write_binary(labelled_training_source_data(typ), f"data/mnist-{resolve_type(typ)}-train.binary")
    write_binary(labelled_test_source_data(typ), f"data/mnist-{resolve_type(typ)}-test.binary")

    # Delete raw source data
    os.remove(f"data/mnist-{resolve_type(typ)}-train.csv")
    os.remove(f"data/mnist-{resolve_type(typ)}-test.csv")


# Check whether binary versions of source data currently exist
def binary_data_available(typ: Type) -> bool:
    return os.path.isfile(f"data/mnist-{resolve_type(typ)}-train.binary") \
       and os.path.isfile(f"data/mnist-{resolve_type(typ)}-test.binary")


def extract_compressed(path, target_directory):
    file = zipfile.ZipFile(path, 'r')
    file.extractall(target_directory)
    file.close()


# Mapping of supported *NIST types
TYPES = {
    Type.Modified: ("modified", 10),
    Type.Extended: ("extended", 62)
}
