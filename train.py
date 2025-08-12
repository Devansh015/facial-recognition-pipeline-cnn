import argparse
import tensorflow as tf
import numpy as np

import mnist
from model import Model

NUM_LABELS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--checkpoint_file_path', type=str, default='checkpoints/model.ckpt-10000')
    parser.add_argument('--train_data', type=str, default='data/mnist_train.csv')
    parser.add_argument('--summary_dir', type=str, default='graphs')
    args = parser.parse_args()

    x_train, x_val, y_train, y_val = mnist.load_train_data(args.train_data)

    model = Model(input_shape=(mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1), num_labels=NUM_LABELS)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.summary_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint_file_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1)

    model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.num_iter // (len(x_train) // args.batch_size),
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, checkpoint_callback]
    )


if __name__ == '__main__':
    main()