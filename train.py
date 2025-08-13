import argparse
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

import mnist
from model import Model

NUM_LABELS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--checkpoint_file_path', type=str, default='checkpoints/model.weights.h5')
    parser.add_argument('--train_data', type=str, default='data/mnist_train.csv')
    parser.add_argument('--summary_dir', type=str, default='graphs')
    args = parser.parse_args()

    # Resolve paths relative to this script directory
    base_dir = Path(__file__).resolve().parent
    if not os.path.isabs(args.train_data):
        args.train_data = str(base_dir / args.train_data)
    if not os.path.isabs(args.summary_dir):
        args.summary_dir = str(base_dir / args.summary_dir)
    if not os.path.isabs(args.checkpoint_file_path):
        args.checkpoint_file_path = str(base_dir / args.checkpoint_file_path)

    # Ensure output directories exist
    os.makedirs(args.summary_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_file_path), exist_ok=True)

    x_train, x_val, y_train, y_val = mnist.load_train_data(args.train_data)

    print('x_train min:', np.min(x_train), 'max:', np.max(x_train), 'any NaN:', np.isnan(x_train).any(), 'any Inf:', np.isinf(x_train).any())
    print('y_train min:', np.min(y_train), 'max:', np.max(y_train), 'any NaN:', np.isnan(y_train).any(), 'any Inf:', np.isinf(y_train).any())
    print('x_val min:', np.min(x_val), 'max:', np.max(x_val), 'any NaN:', np.isnan(x_val).any(), 'any Inf:', np.isinf(x_val).any())
    print('y_val min:', np.min(y_val), 'max:', np.max(y_val), 'any NaN:', np.isnan(y_val).any(), 'any Inf:', np.isinf(y_val).any())

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

    # Compute steps per epoch and guard against zero
    steps_per_epoch = max(1, len(x_train) // args.batch_size)
    epochs = max(1, args.num_iter // steps_per_epoch)

    model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, checkpoint_callback]
    )


if __name__ == '__main__':
    main()