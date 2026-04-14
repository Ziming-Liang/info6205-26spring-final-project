package edu.neu.coe.info6205.neuralnet;

import java.util.Random;

/**
 * Simple feedforward neural network for MNIST digit classification
 * Architecture: 784 -> 128 (ReLU) -> 10 (Softmax)
 */
public class NeuralNetwork {
    private final Layer hiddenLayer;
    private final Layer outputLayer;
    private final Random random;

    /**
     * Create neural network with default architecture
     */
    public NeuralNetwork() {
        this(new Random(42)); // Fixed seed for reproducibility
    }

    /**
     * Create neural network with custom random seed
     */
    public NeuralNetwork(Random random) {
        this.random = random;
        this.hiddenLayer = new Layer(784, 128, new ReLU(), random);
        this.outputLayer = new Layer(128, 10, new Softmax(), random);
    }

    /**
     * Forward propagation - make prediction
     * @param input input vector (784 pixels)
     * @return output probabilities (10 classes)
     */
    public Matrix predict(Matrix input) {
        Matrix hidden = hiddenLayer.forward(input);
        return outputLayer.forward(hidden);
    }

    /**
     * Train on a single example
     * @param input input vector (784 pixels)
     * @param target target one-hot vector (10 classes)
     * @param learningRate learning rate
     * @return loss value
     */
    public double train(Matrix input, Matrix target, double learningRate) {
        // Forward propagation
        Matrix output = predict(input);

        // Compute loss (cross-entropy)
        double loss = computeLoss(output, target);

        // Backward propagation
        // For softmax + cross-entropy, gradient is simply: output - target
        Matrix outputGradient = output.subtract(target);

        // Backprop through output layer
        Matrix hiddenGradient = outputLayer.backward(outputGradient, learningRate);

        // Backprop through hidden layer
        hiddenLayer.backward(hiddenGradient, learningRate);

        return loss;
    }

    /**
     * Compute cross-entropy loss
     * Loss = -sum(target * log(output))
     */
    private double computeLoss(Matrix output, Matrix target) {
        double loss = 0;
        for (int i = 0; i < output.getRows(); i++) {
            double predicted = output.get(i, 0);
            double actual = target.get(i, 0);
            if (actual > 0) { // Only compute for the correct class
                loss -= actual * Math.log(predicted + 1e-15); // Add epsilon to avoid log(0)
            }
        }
        return loss;
    }

    /**
     * Get predicted class (0-9)
     */
    public int getPredictedClass(Matrix output) {
        int maxIndex = 0;
        double maxValue = output.get(0, 0);
        for (int i = 1; i < output.getRows(); i++) {
            if (output.get(i, 0) > maxValue) {
                maxValue = output.get(i, 0);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Evaluate accuracy on a dataset
     */
    public double evaluate(double[][] images, int[] labels) {
        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            Matrix input = new Matrix(images[i]);
            Matrix output = predict(input);
            int predicted = getPredictedClass(output);
            if (predicted == labels[i]) {
                correct++;
            }
        }
        return (double) correct / images.length * 100.0;
    }
}