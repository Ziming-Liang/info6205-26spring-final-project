package edu.neu.coe.info6205.neuralnet;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ReLU activation function
 */
public class ReLUTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testReLUPositiveValues() {
        ReLU relu = new ReLU();
        double[][] data = {{1.5, 2.3}, {0.5, 3.8}};
        Matrix input = new Matrix(data);
        Matrix output = relu.apply(input);

        // Positive values should remain unchanged
        assertEquals(1.5, output.get(0, 0), DELTA);
        assertEquals(2.3, output.get(0, 1), DELTA);
        assertEquals(0.5, output.get(1, 0), DELTA);
        assertEquals(3.8, output.get(1, 1), DELTA);
    }

    @Test
    public void testReLUNegativeValues() {
        ReLU relu = new ReLU();
        double[][] data = {{-2.0, -1.5}, {-0.5, -3.0}};
        Matrix input = new Matrix(data);
        Matrix output = relu.apply(input);

        // Negative values should become 0
        assertEquals(0.0, output.get(0, 0), DELTA);
        assertEquals(0.0, output.get(0, 1), DELTA);
        assertEquals(0.0, output.get(1, 0), DELTA);
        assertEquals(0.0, output.get(1, 1), DELTA);
    }

    @Test
    public void testReLUMixedValues() {
        ReLU relu = new ReLU();
        double[][] data = {{-2.0, 1.0}, {-0.5, 3.0}};
        Matrix input = new Matrix(data);
        Matrix output = relu.apply(input);

        // Mixed values: negative → 0, positive unchanged
        assertEquals(0.0, output.get(0, 0), DELTA);  // -2 → 0
        assertEquals(1.0, output.get(0, 1), DELTA);  // 1 → 1
        assertEquals(0.0, output.get(1, 0), DELTA);  // -0.5 → 0
        assertEquals(3.0, output.get(1, 1), DELTA);  // 3 → 3
    }

    @Test
    public void testReLUZero() {
        ReLU relu = new ReLU();
        double[][] data = {{0.0}};
        Matrix input = new Matrix(data);
        Matrix output = relu.apply(input);

        // Zero should remain zero
        assertEquals(0.0, output.get(0, 0), DELTA);
    }

    @Test
    public void testReLUWithRandomInput() {
        ReLU relu = new ReLU();
        Random random = new Random(42);

        // Generate random matrix with positive and negative values
        double[][] data = new double[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                data[i][j] = random.nextDouble() * 10 - 5; // [-5, 5]
            }
        }
        Matrix input = new Matrix(data);
        Matrix output = relu.apply(input);

        // Verify ReLU rule
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (input.get(i, j) < 0) {
                    assertEquals(0.0, output.get(i, j), DELTA);
                } else {
                    assertEquals(input.get(i, j), output.get(i, j), DELTA);
                }
            }
        }
    }

    @Test
    public void testReLUDerivative() {
        ReLU relu = new ReLU();
        double[][] data = {{-2.0, 1.0}, {0.0, 3.0}};
        Matrix input = new Matrix(data);
        Matrix derivative = relu.derivative(input);

        // ReLU derivative: x > 0 → 1, x <= 0 → 0
        assertEquals(0.0, derivative.get(0, 0), DELTA);  // -2 → 0
        assertEquals(1.0, derivative.get(0, 1), DELTA);  // 1 → 1
        assertEquals(0.0, derivative.get(1, 0), DELTA);  // 0 → 0
        assertEquals(1.0, derivative.get(1, 1), DELTA);  // 3 → 1
    }
}