package edu.neu.coe.info6205.neuralnet;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.zip.GZIPInputStream;

/**
 * Loader for MNIST dataset
 * MNIST files are in IDX format (binary)
 */
public class MNISTLoader {

    /**
     * Load MNIST images from file
     * @param filepath path to images file
     * @return 2D array [numImages][784] with pixel values normalized to [0, 1]
     */
    public static double[][] loadImages(String filepath) throws IOException {
        InputStream is = new FileInputStream(filepath);

        // Handle gzip compressed files
        if (filepath.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }

        DataInputStream dis = new DataInputStream(is);

        // Read header
        int magicNumber = dis.readInt();
        if (magicNumber != 2051) {
            throw new IOException("Invalid MNIST image file - magic number: " + magicNumber);
        }

        int numImages = dis.readInt();
        int numRows = dis.readInt();
        int numCols = dis.readInt();

        System.out.println("Loading " + numImages + " images (" + numRows + "x" + numCols + ")");

        double[][] images = new double[numImages][numRows * numCols];

        // Read pixel data
        for (int i = 0; i < numImages; i++) {
            for (int j = 0; j < numRows * numCols; j++) {
                // Read unsigned byte and normalize to [0, 1]
                images[i][j] = (dis.readUnsignedByte()) / 255.0;
            }
        }

        dis.close();
        return images;
    }

    /**
     * Load MNIST labels from file
     * @param filepath path to labels file
     * @return array of labels (0-9)
     */
    public static int[] loadLabels(String filepath) throws IOException {
        InputStream is = new FileInputStream(filepath);

        // Handle gzip compressed files
        if (filepath.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }

        DataInputStream dis = new DataInputStream(is);

        // Read header
        int magicNumber = dis.readInt();
        if (magicNumber != 2049) {
            throw new IOException("Invalid MNIST label file - magic number: " + magicNumber);
        }

        int numLabels = dis.readInt();
        System.out.println("Loading " + numLabels + " labels");

        int[] labels = new int[numLabels];

        // Read label data 0-9
        for (int i = 0; i < numLabels; i++) {
            labels[i] = dis.readUnsignedByte();
        }

        dis.close();
        return labels;
    }

    /**
     * Convert label to one-hot encoding
     * @param label digit (0-9)
     * @return Matrix 10x1 with 1 at label index, 0 elsewhere
     */
    public static Matrix oneHot(int label) {
        double[] data = new double[10];
        data[label] = 1.0;
        return new Matrix(data);
    }

    /**
     * Print a sample image
     */
    public static void printImage(double[] image) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double pixel = image[i * 28 + j];
                if (pixel > 0.5) {
                    System.out.print("##");
                } else if (pixel > 0.2) {
                    System.out.print("..");
                } else {
                    System.out.print("  ");
                }
            }
            System.out.println();
        }
    }
}