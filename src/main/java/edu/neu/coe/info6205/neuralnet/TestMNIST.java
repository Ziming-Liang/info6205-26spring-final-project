package edu.neu.coe.info6205.neuralnet;

import java.io.IOException;

public class TestMNIST {
    public static void main(String[] args) {
        try {
            System.out.println("Loading MNIST data...");

            String dataPath = "/Users/liang.zim/Desktop/info6205/final project part1/data/";

            // Load training data
            double[][] trainImages = MNISTLoader.loadImages(dataPath + "train-images.idx3-ubyte");
            int[] trainLabels = MNISTLoader.loadLabels(dataPath + "train-labels.idx1-ubyte");

            System.out.println("\nTraining set loaded successfully!");
            System.out.println("Number of training images: " + trainImages.length);
            System.out.println("Image size: " + trainImages[0].length + " pixels");

            // Load test data
            double[][] testImages = MNISTLoader.loadImages(dataPath + "t10k-images.idx3-ubyte");
            int[] testLabels = MNISTLoader.loadLabels(dataPath + "t10k-labels.idx1-ubyte");

            System.out.println("\nTest set loaded successfully!");
            System.out.println("Number of test images: " + testImages.length);

            // Print first training example
            System.out.println("\n=== First training example ===");
            System.out.println("Label: " + trainLabels[0]);
            System.out.println("Image visualization:");
            MNISTLoader.printImage(trainImages[0]);

            // Print first test example
            System.out.println("\n=== First test example ===");
            System.out.println("Label: " + testLabels[0]);
            System.out.println("Image visualization:");
            MNISTLoader.printImage(testImages[0]);

        } catch (IOException e) {
            System.err.println("Error loading MNIST data: " + e.getMessage());
            e.printStackTrace();
        }
    }
}