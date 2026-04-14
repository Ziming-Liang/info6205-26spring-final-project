package edu.neu.coe.info6205.neuralnet;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for NeuralNetwork class
 */
public class NeuralNetworkTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testNetworkCreation() {
        NeuralNetwork nn = new NeuralNetwork();
        assertNotNull(nn);
    }

    @Test
    public void testPredictOutputShape() {
        NeuralNetwork nn = new NeuralNetwork();

        // Create random input (784 pixels)
        Random random = new Random(42);
        double[] inputData = new double[784];
        for (int i = 0; i < 784; i++) {
            inputData[i] = random.nextDouble();
        }
        Matrix input = new Matrix(inputData);

        Matrix output = nn.predict(input);

        // Output should be 10x1 (10 classes)
        assertEquals(10, output.getRows());
        assertEquals(1, output.getCols());

        // Outputs should sum to 1 (softmax)
        double sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += output.get(i, 0);
            // Each output should be a valid probability
            assertTrue(output.get(i, 0) >= 0 && output.get(i, 0) <= 1);
        }
        assertEquals(1.0, sum, DELTA);
    }

    @Test
    public void testGetPredictedClass() {
        NeuralNetwork nn = new NeuralNetwork();

        // Create output with clear maximum at index 5
        double[][] outputData = new double[10][1];
        outputData[5][0] = 0.9;
        outputData[3][0] = 0.1;
        Matrix output = new Matrix(outputData);

        int predicted = nn.getPredictedClass(output);
        assertEquals(5, predicted);
    }

    @Test
    public void testTrainReducesLoss() {
        NeuralNetwork nn = new NeuralNetwork(new Random(42));

        // Create simple input
        double[] inputData = new double[784];
        for (int i = 0; i < 784; i++) {
            inputData[i] = 0.5;
        }
        Matrix input = new Matrix(inputData);

        // Create target (one-hot for class 3)
        double[] targetData = new double[10];
        targetData[3] = 1.0;
        Matrix target = new Matrix(targetData);

        // Train multiple times
        double initialLoss = nn.train(input, target, 0.01);
        double finalLoss = initialLoss;

        for (int i = 0; i < 100; i++) {
            finalLoss = nn.train(input, target, 0.01);
        }

        // Loss should decrease
        assertTrue(finalLoss < initialLoss,
                "Loss should decrease after training. Initial: " + initialLoss + ", Final: " + finalLoss);
    }

    @Test
    public void testEvaluate() {
        NeuralNetwork nn = new NeuralNetwork(new Random(42));

        // Create dummy dataset (3 samples)
        double[][] images = new double[3][784];
        int[] labels = {0, 1, 2};

        Random random = new Random(42);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 784; j++) {
                images[i][j] = random.nextDouble();
            }
        }

        double accuracy = nn.evaluate(images, labels);

        // Accuracy should be between 0 and 100
        assertTrue(accuracy >= 0 && accuracy <= 100);
    }
}