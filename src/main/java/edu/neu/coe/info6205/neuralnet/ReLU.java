package edu.neu.coe.info6205.neuralnet;

/**
 * ReLU activation function
 * f(x) = max(0, x)
 */
public class ReLU implements ActivationFunction {

    @Override
    public Matrix apply(Matrix input) {
        return input.applyFunction(x -> Math.max(0, x));
    }

    @Override
    public Matrix derivative(Matrix input) {
        // Derivative: 1 if x > 0, else 0
        return input.applyFunction(x -> x > 0 ? 1.0 : 0.0);
    }
}