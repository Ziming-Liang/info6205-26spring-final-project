package edu.neu.coe.info6205.neuralnet;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Matrix class
 */
public class MatrixTest {

    private static final double DELTA = 1e-6; // Tolerance for floating point comparison

    @Test
    public void testConstructor() {
        Matrix m = new Matrix(3, 2);
        assertEquals(3, m.getRows());
        assertEquals(2, m.getCols());
        assertEquals(0.0, m.get(0, 0), DELTA);
    }

    @Test
    public void testConstructorFromArray() {
        double[][] data = {{1, 2}, {3, 4}, {5, 6}};
        Matrix m = new Matrix(data);
        assertEquals(3, m.getRows());
        assertEquals(2, m.getCols());
        assertEquals(1.0, m.get(0, 0), DELTA);
        assertEquals(4.0, m.get(1, 1), DELTA);
    }

    @Test
    public void testConstructorFrom1DArray() {
        double[] data = {1, 2, 3};
        Matrix m = new Matrix(data);
        assertEquals(3, m.getRows());
        assertEquals(1, m.getCols());
        assertEquals(2.0, m.get(1, 0), DELTA);
    }

    @Test
    public void testAddition() {
        double[][] data1 = {{1, 2}, {3, 4}};
        double[][] data2 = {{5, 6}, {7, 8}};
        Matrix m1 = new Matrix(data1);
        Matrix m2 = new Matrix(data2);
        Matrix result = m1.add(m2);

        assertEquals(6.0, result.get(0, 0), DELTA);
        assertEquals(8.0, result.get(0, 1), DELTA);
        assertEquals(10.0, result.get(1, 0), DELTA);
        assertEquals(12.0, result.get(1, 1), DELTA);
    }

    @Test
    public void testSubtraction() {
        double[][] data1 = {{5, 6}, {7, 8}};
        double[][] data2 = {{1, 2}, {3, 4}};
        Matrix m1 = new Matrix(data1);
        Matrix m2 = new Matrix(data2);
        Matrix result = m1.subtract(m2);

        assertEquals(4.0, result.get(0, 0), DELTA);
        assertEquals(4.0, result.get(0, 1), DELTA);
        assertEquals(4.0, result.get(1, 0), DELTA);
        assertEquals(4.0, result.get(1, 1), DELTA);
    }

    @Test
    public void testMatrixMultiplication() {
        double[][] data1 = {{1, 2, 3}, {4, 5, 6}};
        double[][] data2 = {{7, 8}, {9, 10}, {11, 12}};
        Matrix m1 = new Matrix(data1);
        Matrix m2 = new Matrix(data2);
        Matrix result = m1.multiply(m2);

        assertEquals(2, result.getRows());
        assertEquals(2, result.getCols());
        // First row, first col: 1*7 + 2*9 + 3*11 = 58
        assertEquals(58.0, result.get(0, 0), DELTA);
        // First row, second col: 1*8 + 2*10 + 3*12 = 64
        assertEquals(64.0, result.get(0, 1), DELTA);
        // Second row, first col: 4*7 + 5*9 + 6*11 = 139
        assertEquals(139.0, result.get(1, 0), DELTA);
        // Second row, second col: 4*8 + 5*10 + 6*12 = 154
        assertEquals(154.0, result.get(1, 1), DELTA);
    }

    @Test
    public void testElementMultiply() {
        double[][] data1 = {{1, 2}, {3, 4}};
        double[][] data2 = {{5, 6}, {7, 8}};
        Matrix m1 = new Matrix(data1);
        Matrix m2 = new Matrix(data2);
        Matrix result = m1.elementMultiply(m2);

        assertEquals(5.0, result.get(0, 0), DELTA);
        assertEquals(12.0, result.get(0, 1), DELTA);
        assertEquals(21.0, result.get(1, 0), DELTA);
        assertEquals(32.0, result.get(1, 1), DELTA);
    }

    @Test
    public void testTranspose() {
        double[][] data = {{1, 2, 3}, {4, 5, 6}};
        Matrix m = new Matrix(data);
        Matrix result = m.transpose();

        assertEquals(3, result.getRows());
        assertEquals(2, result.getCols());
        assertEquals(1.0, result.get(0, 0), DELTA);
        assertEquals(4.0, result.get(0, 1), DELTA);
        assertEquals(2.0, result.get(1, 0), DELTA);
        assertEquals(5.0, result.get(1, 1), DELTA);
        assertEquals(3.0, result.get(2, 0), DELTA);
        assertEquals(6.0, result.get(2, 1), DELTA);
    }

    @Test
    public void testApplyFunction() {
        double[][] data = {{1, 2}, {3, 4}};
        Matrix m = new Matrix(data);
        Matrix result = m.applyFunction(x -> x * 2);

        assertEquals(2.0, result.get(0, 0), DELTA);
        assertEquals(4.0, result.get(0, 1), DELTA);
        assertEquals(6.0, result.get(1, 0), DELTA);
        assertEquals(8.0, result.get(1, 1), DELTA);
    }

    @Test
    public void testRandomize() {
        Random random = new Random(42); // Fixed seed for reproducibility
        Matrix m = Matrix.randomize(10, 10, random);

        assertEquals(10, m.getRows());
        assertEquals(10, m.getCols());

        // Check that values are within Xavier initialization bounds
        double limit = Math.sqrt(6.0 / 20.0);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                assertTrue(m.get(i, j) >= -limit && m.get(i, j) <= limit);
            }
        }
    }

    @Test
    public void testZeros() {
        Matrix m = Matrix.zeros(3, 2);
        assertEquals(3, m.getRows());
        assertEquals(2, m.getCols());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(0.0, m.get(i, j), DELTA);
            }
        }
    }

    @Test
    public void testAdditionDimensionMismatch() {
        Matrix m1 = new Matrix(2, 3);
        Matrix m2 = new Matrix(3, 2);
        assertThrows(IllegalArgumentException.class, () -> m1.add(m2));
    }

    @Test
    public void testMultiplicationDimensionMismatch() {
        Matrix m1 = new Matrix(2, 3);
        Matrix m2 = new Matrix(2, 3);
        assertThrows(IllegalArgumentException.class, () -> m1.multiply(m2));
    }
}