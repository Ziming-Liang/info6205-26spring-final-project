package edu.neu.coe.info6205.neuralnet;

/**
 * Interface for activation functions
 */
public interface ActivationFunction {
    /**
     * Apply activation function to input matrix
     */
    Matrix apply(Matrix input);

    /**
     * Compute derivative of activation function
     * Used in backpropagation
     */
    Matrix derivative(Matrix input);
}