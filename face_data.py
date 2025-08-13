import os
from pathlib import Path
from typing import Tuple, List, Optional

import tensorflow as tf


def load_lfw(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 1337,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Load LFW dataset from a directory structure like data/lfw/<person_name>/<images>.
    Returns training and validation datasets and the list of class names.
    """
    base = Path(data_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"LFW directory not found: {base}. Run scripts/download_lfw.sh first.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        base,
        labels='inferred',
        label_mode='categorical',
        validation_split=validation_split,
        subset='training',
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=shuffle,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        base,
        labels='inferred',
        label_mode='categorical',
        validation_split=validation_split,
        subset='validation',
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=False,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    # Normalize to [0,1]
    def _norm(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    train_ds = train_ds.map(_norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(_norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names
