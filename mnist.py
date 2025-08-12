import numpy as np
import tensorflow as tf
import os

IMAGE_SIZE = 28


def load_train_data(data_path=None, validation_size=500):
    """
    Load mnist data. If data_path is provided and exists, load from CSV. 
    Otherwise, automatically download from TensorFlow datasets.
    :return: 3D Tensor input of train and validation set with 2D Tensor of one hot encoded image labels
    """
    # Check if CSV file exists
    if data_path and os.path.exists(data_path):
        print(f"Loading MNIST data from CSV file: {data_path}")
        # Original CSV loading code
        train_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
        x_train = train_data[:, 1:] / 255.0  # Normalize to [0, 1]

        # Get label and one-hot encode
        y_train = train_data[:, 0]
        y_train = (np.arange(10) == y_train[:, None]).astype(np.float32)

        # get a validation set and remove it from the train set
        x_train, x_val, y_train, y_val = x_train[0:(len(x_train) - validation_size), :], x_train[(len(x_train) - validation_size):len(x_train), :], \
                                         y_train[0:(len(y_train) - validation_size), :], y_train[(len(y_train) - validation_size):len(y_train), :]
        # reformat the data so it's not flat
        x_train = x_train.reshape(len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1)
        x_val = x_val.reshape(len(x_val), IMAGE_SIZE, IMAGE_SIZE, 1)
    else:
        print("CSV file not found. Downloading MNIST dataset from TensorFlow...")
        # Load MNIST from TensorFlow datasets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize to [0, 1] and add channel dimension
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
        x_test = np.expand_dims(x_test, axis=-1)
        
        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, 10).astype(np.float32)
        y_test = tf.keras.utils.to_categorical(y_test, 10).astype(np.float32)
        
        # Create validation set from training data
        x_val = x_train[-validation_size:]
        y_val = y_train[-validation_size:]
        x_train = x_train[:-validation_size]
        y_train = y_train[:-validation_size]

    print(f"Training data: {x_train.shape}, Validation data: {x_val.shape}")
    return x_train, x_val, y_train, y_val


def load_test_data(data_path=None):
    """
    Load mnist test data. If data_path is provided and exists, load from CSV.
    Otherwise, automatically download from TensorFlow datasets.
    :return: 3D Tensor input of test set with 2D Tensor of one hot encoded image labels
    """
    # Check if CSV file exists
    if data_path and os.path.exists(data_path):
        print(f"Loading MNIST test data from CSV file: {data_path}")
        test_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
        x_test = test_data[:, 1:] / 255.0  # Normalize to [0, 1]

        y_test = np.array(test_data[:, 0])
        y_test = (np.arange(10) == y_test[:, None]).astype(np.float32)

        x_test = x_test.reshape(len(x_test), IMAGE_SIZE, IMAGE_SIZE, 1)
    else:
        print("CSV file not found. Using MNIST test data from TensorFlow...")
        # Load MNIST from TensorFlow datasets
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize to [0, 1] and add channel dimension
        x_test = x_test.astype(np.float32) / 255.0
        x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension
        
        # One-hot encode labels
        y_test = tf.keras.utils.to_categorical(y_test, 10).astype(np.float32)

    print(f"Test data: {x_test.shape}")
    return x_test, y_test