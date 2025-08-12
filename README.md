---

## Overview

- Building the CNN architecture (convolution, pooling, fully connected layers)
- Training on CPU vs GPU
- Using AWS GPU instances for faster performance
- Visualizing results in TensorBoard

---

## Tech Stack

- **Programming Language**: Python (2.7 or 3.5 in the original tutorial)
- **Framework**: TensorFlow 2x
- **Visualization**: TensorBoard
- **Deployment Environment**: Local machine or AWS GPU instances (e.g., `g2.2xlarge`)
- **Dataset**: MNIST handwritten digits
- **Tools**: CUDA (for GPU acceleration), cuDNN

---

## Features

- CNN implementation with TensorFlow
- Configurable convolutional, pooling, and fully connected layers
- Softmax cross-entropy loss with gradient descent optimization
- Training on CPU or GPU
- AWS GPU instance setup instructions
- TensorBoard integration for training visualization
- High accuracy (~98%) on MNIST over 10,000 iterations

---

## Architecture

**Data Flow**:
1. **Input Layer** – Accepts MNIST images (28×28 pixels, grayscale).
2. **Convolutional Layers** – Extract spatial features using learned filters.
3. **Activation** – ReLU for non-linear transformation.
4. **Pooling Layers** – Downsample feature maps (max pooling).
5. **Fully Connected Layer** – Flattened output to dense units.
6. **Output Layer** – 10 neurons (one per digit class), softmax activation.

## Installation
```bash
pip install tensorflow 
pip install matplotlib numpy
```
```bash
3️⃣ Download the MNIST Dataset
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv # Both train and test available from the same page
```
```bash
4️⃣ Run Training (CPU)
python3 train.py
```
