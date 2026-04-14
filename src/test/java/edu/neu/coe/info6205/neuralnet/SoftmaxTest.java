package edu.neu.coe.info6205.neuralnet;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Softmax activation function
 */
public class SoftmaxTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testSoftmaxSumsToOne() {
        Softmax softmax = new Softmax();
        double[] data = {1.0, 2.0, 3.0, 4.0};
        Matrix input = new Matrix(data);
        Matrix output = softmax.apply(input);

        // Softmax outputs should sum to 1
        double sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += output.get(i, 0);
        }
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void testSoftmaxAllPositive() {
        Softmax softmax = new Softmax();
        double[] data = {1.0, 2.0, 3.0};
        Matrix input = new Matrix(data);
        Matrix output = softmax.apply(input);

        // All outputs should be in (0, 1)
        for (int i = 0; i < 3; i++) {
            assertTrue(output.get(i, 0) > 0);
            assertTrue(output.get(i, 0) < 1);
        }
    }

    @Test
    public void testSoftmaxMaxValueHighestProbability() {
        Softmax softmax = new Softmax();
        double[] data = {1.0, 2.0, 5.0, 3.0};  // 5.0 is max
        Matrix input = new Matrix(data);
        Matrix output = softmax.apply(input);

        // Find max value index
        int maxIndex = 2;  // 5.0 at index 2
        double maxProb = output.get(maxIndex, 0);

        // Max value should have highest probability
        for (int i = 0; i < 4; i++) {
            if (i != maxIndex) {
                assertTrue(output.get(i, 0) < maxProb);
            }
        }
    }

    @Test
    public void testSoftmaxWithNegativeValues() {
        Softmax softmax = new Softmax();
        double[] data = {-2.0, -1.0, 0.0, 1.0};
        Matrix input = new Matrix(data);
        Matrix output = softmax.apply(input);

        // Even with negative inputs, outputs should be valid probabilities
        double sum = 0;
        for (int i = 0; i < 4; i++) {
            assertTrue(output.get(i, 0) >= 0);
            assertTrue(output.get(i, 0) <= 1);
            sum += output.get(i, 0);
        }
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void testSoftmaxWithRandomInput() {
        Softmax softmax = new Softmax();
        Random random = new Random(42);

        // Generate random input (10 neurons, simulating output layer)
        double[] data = new double[10];
        for (int i = 0; i < 10; i++) {
            data[i] = random.nextDouble() * 10 - 5;  // [-5, 5]
        }
        Matrix input = new Matrix(data);
        Matrix output = softmax.apply(input);

        // Verify Softmax properties
        double sum = 0;
        for (int i = 0; i < 10; i++) {
            assertTrue(output.get(i, 0) >= 0);
            assertTrue(output.get(i, 0) <= 1);
            sum += output.get(i, 0);
        }
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void testSoftmaxLargeValues() {
        Softmax softmax = new Softmax();
        double[] data = {100.0, 200.0, 300.0};  // Large values
        Matrix input = new Matrix(data);
        Matrix output = softmax.apply(input);

        // Should not overflow and sum to 1
        double sum = 0;
        for (int i = 0; i < 3; i++) {
            assertFalse(Double.isNaN(output.get(i, 0)));
            assertFalse(Double.isInfinite(output.get(i, 0)));
            sum += output.get(i, 0);
        }
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void testSoftmaxEqualValues() {
        Softmax softmax = new Softmax();
        double[] data = {2.0, 2.0, 2.0, 2.0};  // All equal
        Matrix input = new Matrix(data);
        Matrix output = softmax.apply(input);

        // All output probabilities should be equal (25% each)
        for (int i = 0; i < 4; i++) {
            assertEquals(0.25, output.get(i, 0), DELTA);
        }
    }
}