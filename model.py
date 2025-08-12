import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, input_shape=(28, 28, 1), num_labels=10):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(num_labels)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x