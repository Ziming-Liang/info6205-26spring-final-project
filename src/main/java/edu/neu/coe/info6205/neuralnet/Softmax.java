package edu.neu.coe.info6205.neuralnet;

/**
 * Softmax activation function
 * Converts raw scores into probabilities that sum to 1
 */
public class Softmax implements ActivationFunction {
    //input:[1,2,3,4,5...] output[1%,2%,3%,4%,5%...]
    @Override
    public Matrix apply(Matrix input) {
        // Subtract max
        double max = findMax(input);
        Matrix shifted = input.applyFunction(x -> x - max);

        // Compute exp(x) for each element
        Matrix exp = shifted.applyFunction(Math::exp);

        // Sum all exp
        double sum = 0;
        for (int i = 0; i < exp.getRows(); i++) {
            for (int j = 0; j < exp.getCols(); j++) {
                sum += exp.get(i, j);
            }
        }

        // Divide each element by sum
        final double finalSum = sum;
        return exp.applyFunction(x -> x / finalSum);
    }

    @Override
    public Matrix derivative(Matrix input) {
        return input;
    }

    private double findMax(Matrix m) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                max = Math.max(max, m.get(i, j));
            }
        }
        return max;
    }
}