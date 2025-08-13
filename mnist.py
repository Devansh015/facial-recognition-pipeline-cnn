import numpy as np

IMAGE_SIZE = 28


def load_train_data(data_path, validation_size=500):
    """
    Load mnist data. Each row in csv is formatted (label, input)
    :return: 3D Tensor input of train and validation set with 2D Tensor of one hot encoded image labels
    """
    # Data format in Kaggle CSVs has a header row; skip it and filter any bad rows.
    train_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32, skip_header=1)
    # Drop any rows that contain NaNs/Infs
    mask = ~(np.isnan(train_data).any(axis=1) | np.isinf(train_data).any(axis=1))
    train_data = train_data[mask]

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

    return x_train, x_val, y_train, y_val


def load_test_data(data_path):
    """
    Load mnist test data
    :return: 3D Tensor input of train and validation set with 2D Tensor of one hot encoded image labels
    """
    test_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32, skip_header=1)
    # Drop any rows that contain NaNs/Infs
    mask = ~(np.isnan(test_data).any(axis=1) | np.isinf(test_data).any(axis=1))
    test_data = test_data[mask]

    x_test = test_data[:, 1:] / 255.0  # Normalize to [0, 1]

    y_test = np.array(test_data[:, 0])
    y_test = (np.arange(10) == y_test[:, None]).astype(np.float32)

    x_test = x_test.reshape(len(x_test), IMAGE_SIZE, IMAGE_SIZE, 1)

    return x_test, y_test