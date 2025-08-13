import argparse
import os
from pathlib import Path
import tensorflow as tf

from face_data import load_lfw
from face_model import build_face_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/lfw')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--summary_dir', type=str, default='graphs_faces')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_faces/model.h5')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    if not os.path.isabs(args.data_dir):
        args.data_dir = str(base_dir / args.data_dir)
    if not os.path.isabs(args.summary_dir):
        args.summary_dir = str(base_dir / args.summary_dir)
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = str(base_dir / args.checkpoint)

    os.makedirs(args.summary_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    train_ds, val_ds, class_names = load_lfw(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    model = build_face_classifier(num_classes=len(class_names), input_shape=(args.image_size, args.image_size, 3))

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.summary_dir),
        tf.keras.callbacks.ModelCheckpoint(args.checkpoint, save_best_only=True, monitor='val_accuracy', mode='max')
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
