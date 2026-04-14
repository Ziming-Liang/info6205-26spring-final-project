package edu.neu.coe.info6205.neuralnet;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Layer class
 */
public class LayerTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testLayerCreation() {
        Random random = new Random(42);
        Layer layer = new Layer(3, 2, new ReLU(), random);

        assertEquals(2, layer.getWeights().getRows());
        assertEquals(3, layer.getWeights().getCols());
        assertEquals(2, layer.getBiases().getRows());
        assertEquals(1, layer.getBiases().getCols());
    }

    @Test
    public void testForwardPropagation() {
        Random random = new Random(42);
        Layer layer = new Layer(3, 2, new ReLU(), random);

        double[] inputData = {1.0, 2.0, 3.0};
        Matrix input = new Matrix(inputData);

        Matrix output = layer.forward(input);

        assertEquals(2, output.getRows());
        assertEquals(1, output.getCols());

        // Output should be non-negative (ReLU)
        assertTrue(output.get(0, 0) >= 0);
        assertTrue(output.get(1, 0) >= 0);
    }

    @Test
    public void testForwardPropagationWithSoftmax() {
        Random random = new Random(42);
        Layer layer = new Layer(3, 10, new Softmax(), random);

        double[] inputData = {1.0, 2.0, 3.0};
        Matrix input = new Matrix(inputData);

        Matrix output = layer.forward(input);

        assertEquals(10, output.getRows());
        assertEquals(1, output.getCols());

        // Softmax outputs should sum to 1
        double sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += output.get(i, 0);
            // Each output should be between 0 and 1
            assertTrue(output.get(i, 0) >= 0 && output.get(i, 0) <= 1);
        }
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void testBackwardPropagation() {
        Random random = new Random(42);
        Layer layer = new Layer(3, 2, new ReLU(), random);

        double[] inputData = {1.0, 2.0, 3.0};
        Matrix input = new Matrix(inputData);

        // Forward pass
        Matrix output = layer.forward(input);

        // Create a dummy gradient
        double[][] gradData = {{0.1}, {0.2}};
        Matrix outputGradient = new Matrix(gradData);

        // Backward pass
        Matrix inputGradient = layer.backward(outputGradient, 0.01);

        // Input gradient should have same shape as input
        assertEquals(3, inputGradient.getRows());
        assertEquals(1, inputGradient.getCols());
    }

    @Test
    public void testWeightUpdate() {
        Random random = new Random(42);
        Layer layer = new Layer(2, 2, new ReLU(), random);

        // Get initial weights
        Matrix initialWeights = layer.getWeights();
        double initialWeight00 = initialWeights.get(0, 0);

        // Forward and backward pass
        double[] inputData = {1.0, 1.0};
        Matrix input = new Matrix(inputData);
        layer.forward(input);

        double[][] gradData = {{1.0}, {1.0}};
        Matrix outputGradient = new Matrix(gradData);
        layer.backward(outputGradient, 0.01);

        // Weights should have changed
        Matrix updatedWeights = layer.getWeights();
        assertNotEquals(initialWeight00, updatedWeights.get(0, 0), DELTA);
    }
}