# INFO 6205 Spring 2026 – Final Project Part 1

---

## Project Structure

```
├── data/
│   ├── train-images.idx3-ubyte   # 60,000 training images
│   ├── train-labels.idx1-ubyte   # 60,000 training labels
│   ├── t10k-images.idx3-ubyte    # 10,000 test images
│   └── t10k-labels.idx1-ubyte    # 10,000 test labels
├── src/
│   ├── main/java/edu/neu/coe/info6205/neuralnet/
│   │   ├── ActivationFunction.java   # Activation function interface
│   │   ├── Layer.java                # Single neural network layer
│   │   ├── Matrix.java               # Matrix operations
│   │   ├── MNISTLoader.java          # MNIST data loader
│   │   ├── NeuralNetwork.java        # Neural network model
│   │   ├── ReLU.java                 # ReLU activation function
│   │   ├── Softmax.java              # Softmax activation function
│   │   ├── TestMNIST.java            # Data loading verification
│   │   └── Trainer.java              # Training entry point
│   └── test/java/edu/neu/coe/info6205/neuralnet/
│       ├── LayerTest.java
│       ├── MatrixTest.java
│       ├── NeuralNetworkTest.java
│       ├── ReLUTest.java
│       └── SoftmaxTest.java
└── pom.xml
```

---

## Network Architecture

```
Input (784)  →  Hidden Layer (128, ReLU)  →  Output Layer (10, Softmax)
```

- **Input**: 28×28 pixel images flattened to 784-dimensional vectors
- **Hidden layer**: 128 neurons with ReLU activation
- **Output layer**: 10 neurons with Softmax activation (one per digit 0–9)
- **Loss function**: Cross-entropy
- **Optimizer**: SGD with gradient clipping (clip value = 5.0)

---

## Training Parameters

| Parameter      | Value  |
|----------------|--------|
| Epochs         | 5      |
| Learning rate  | 0.001  |
| Training set   | 60,000 |
| Test set       | 10,000 |
| Random seed    | 42     |

---

## Requirements

- Java 11+
- Maven 3.6+

---

## How to Run

1. Sync Project via `pom.xml`
2. Run `Trainer.java` to train the model
3. Run unit tests in the `test` folder

