package edu.neu.coe.info6205.neuralnet;

import java.util.Random;
import java.util.function.Function;

/**
 * Matrix class for neural network computations
 */
public class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;

    // Constructors

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    //从二维数组创建矩阵

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    // 从一维数组创建列向量 (n×1矩阵)用于输入向量
    public Matrix(double[] array) {
        this.rows = array.length;
        this.cols = 1;
        this.data = new double[rows][1];
        for (int i = 0; i < rows; i++) {
            this.data[i][0] = array[i];
        }
    }

    // Getters
    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    public void set(int row, int col, double value) {
        data[row][col] = value;
    }

    // Matrix addition
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // Matrix subtraction
    public Matrix subtract(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for subtraction");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    // Matrix multiplication (m×n) × (n×p) = (m×p)，
    // 用于前向传播：weights × input
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                    "Number of columns in first matrix must equal number of rows in second matrix");
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // Element-wise multiplication 对应位置相乘，
    // 用于反向传播梯度 × 激活函数导数
    public Matrix elementMultiply(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for element-wise multiplication");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    // Transpose (m×n) → (n×m)，
    // 用于反向传播计算梯度
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    // Apply function to each element
    public Matrix applyFunction(Function<Double, Double> func) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func.apply(this.data[i][j]);
            }
        }
        return result;
    }

    // Create random matrix with Xavier initialization
    //[-limit, limit]
    //初始化神经网络权重
    public static Matrix randomize(int rows, int cols, Random random) {
        Matrix result = new Matrix(rows, cols);
        double limit = Math.sqrt(6.0 / (rows + cols)); // Xavier initialization
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = random.nextDouble() * 2 * limit - limit;
            }
        }
        return result;
    }

    // Create matrix filled with zeros
    public static Matrix zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }

    // Print
    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%.4f ", data[i][j]);
            }
            System.out.println();
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < Math.min(rows, 3); i++) {
            sb.append("[");
            for (int j = 0; j < Math.min(cols, 3); j++) {
                sb.append(String.format("%.4f", data[i][j]));
                if (j < Math.min(cols, 3) - 1) sb.append(", ");
            }
            if (cols > 3) sb.append(", ...");
            sb.append("]");
            if (i < Math.min(rows, 3) - 1) sb.append(", ");
        }
        if (rows > 3) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }
}