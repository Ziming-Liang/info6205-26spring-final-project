package edu.neu.coe.info6205.neuralnet;

import java.io.IOException;
import java.util.Random;


public class Trainer {

    public static void main(String[] args) {
        try {
            System.out.println("=== MNIST Digit Classification ===\n");

            // Data path
            String dataPath = "data/";

            // Load data
            System.out.println("Loading MNIST dataset...");
            double[][] trainImages = MNISTLoader.loadImages(dataPath + "train-images.idx3-ubyte");
            int[] trainLabels = MNISTLoader.loadLabels(dataPath + "train-labels.idx1-ubyte");
            double[][] testImages = MNISTLoader.loadImages(dataPath + "t10k-images.idx3-ubyte");
            int[] testLabels = MNISTLoader.loadLabels(dataPath + "t10k-labels.idx1-ubyte");

            System.out.println("Dataset loaded successfully!\n");

            // Create neural network
            System.out.println("Creating neural network...");
            System.out.println("Architecture: 784 -> 128 (ReLU) -> 10 (Softmax)\n");
            NeuralNetwork nn = new NeuralNetwork(new Random(42));

            // Training parameters
            int epochs = 5;
            double learningRate = 0.001;  // ← 改这里：从0.01降到0.001
            int batchSize = 1;

            System.out.println("Training parameters:");
            System.out.println("  Epochs: " + epochs);
            System.out.println("  Learning rate: " + learningRate);
            System.out.println("  Training samples: " + trainImages.length);
            System.out.println("  Test samples: " + testImages.length);
            System.out.println();

            // Evaluate before training
            double initialAccuracy = nn.evaluate(testImages, testLabels);
            System.out.println("Initial test accuracy: " + String.format("%.2f%%", initialAccuracy));
            System.out.println("\nStarting training...\n");

            // Training loop
            for (int epoch = 0; epoch < epochs; epoch++) {
                long startTime = System.currentTimeMillis();
                double totalLoss = 0;

                for (int i = 0; i < trainImages.length; i++) {
                    Matrix input = new Matrix(trainImages[i]);
                    Matrix target = MNISTLoader.oneHot(trainLabels[i]);

                    double loss = nn.train(input, target, learningRate);
                    totalLoss += loss;

                    if ((i + 1) % 10000 == 0) {
                        System.out.println("  Epoch " + (epoch + 1) + " - Processed " + (i + 1) + "/" + trainImages.length + " samples");
                    }
                }

                long endTime = System.currentTimeMillis();
                double avgLoss = totalLoss / trainImages.length;
                double testAccuracy = nn.evaluate(testImages, testLabels);

                System.out.println("Epoch " + (epoch + 1) + "/" + epochs + " completed in " + (endTime - startTime) / 1000.0 + "s");
                System.out.println("  Average loss: " + String.format("%.4f", avgLoss));
                System.out.println("  Test accuracy: " + String.format("%.2f%%", testAccuracy));
                System.out.println();
            }

            // Final evaluation
            System.out.println("=== Training Complete ===");
            double finalAccuracy = nn.evaluate(testImages, testLabels);
            System.out.println("Final test accuracy: " + String.format("%.2f%%", finalAccuracy));

            // Generate confusion matrix
            System.out.println("\n=== Generating Confusion Matrix ===");
            int[][] confusionMatrix = generateConfusionMatrix(nn, testImages, testLabels);
            printConfusionMatrix(confusionMatrix);

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static int[][] generateConfusionMatrix(NeuralNetwork nn, double[][] images, int[] labels) {
        int[][] matrix = new int[10][10];

        for (int i = 0; i < images.length; i++) {
            Matrix input = new Matrix(images[i]);
            Matrix output = nn.predict(input);
            int predicted = nn.getPredictedClass(output);
            int actual = labels[i];
            matrix[actual][predicted]++;
        }

        return matrix;
    }

    private static void printConfusionMatrix(int[][] matrix) {
        System.out.println("\nConfusion Matrix (rows=actual, cols=predicted):");
        System.out.print("     ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%6d", i);
        }
        System.out.println();
        System.out.println("    " + "-".repeat(66));

        for (int i = 0; i < 10; i++) {
            System.out.printf("%2d | ", i);
            for (int j = 0; j < 10; j++) {
                System.out.printf("%6d", matrix[i][j]);
            }
            System.out.println();
        }

        System.out.println("\nPer-class accuracy:");
        for (int i = 0; i < 10; i++) {
            int total = 0;
            for (int j = 0; j < 10; j++) {
                total += matrix[i][j];
            }
            double accuracy = (double) matrix[i][i] / total * 100.0;
            System.out.printf("Digit %d: %.2f%% (%d/%d)\n", i, accuracy, matrix[i][i], total);
        }
    }
}