package edu.neu.coe.info6205.neuralnet;

import java.util.Random;

/**
 * Represents a single layer in the neural network
 */
public class Layer {
    // 权重矩阵
    private Matrix weights;
    // 偏置向量
    private Matrix biases;
    // 激活函数（ReLU或Softmax）
    private final ActivationFunction activation;

    // 缓存：前向传播时保存，反向传播时使用
    private Matrix lastInput;      // 上一层的输入
    private Matrix lastZ;          // 激活前的值 (z = weights * input + biases)
    private Matrix lastOutput;     // 激活后的输出

    /**
     * 创建一层神经网络
     * @param inputSize 输入维度
     * @param outputSize 输出维度（神经元数量）
     * @param activation 激活函数
     * @param random 随机数生成器（用于初始化权重）
     */
    public Layer(int inputSize, int outputSize, ActivationFunction activation, Random random) {
        this.weights = Matrix.randomize(outputSize, inputSize, random);  // 随机初始化权重
        this.biases = Matrix.zeros(outputSize, 1);  // 偏置初始化为0
        this.activation = activation;
    }

    /**
     * 前向传播：计算这一层的输出
     * @param input 输入数据
     * @return 激活后的输出
     */
    public Matrix forward(Matrix input) {
        this.lastInput = input;  // 保存输入，反向传播时用
        this.lastZ = weights.multiply(input).add(biases);  // z = W * x + b
        this.lastOutput = activation.apply(lastZ);  // 应用激活函数（ReLU或Softmax）
        return lastOutput;
    }

    /**
     * 反向传播：计算梯度并更新权重
     * @param outputGradient 从下一层传来的梯度
     * @param learningRate 学习率
     * @return 传给上一层的梯度
     */
    public Matrix backward(Matrix outputGradient, double learningRate) {
        Matrix gradient;

        // Softmax层的梯度已经简化（output - target），其他层需要乘激活函数导数
        if (activation instanceof Softmax) {
            gradient = outputGradient;
        } else {
            Matrix activationGradient = activation.derivative(lastZ);  // 激活函数的导数
            gradient = outputGradient.elementMultiply(activationGradient);  // 逐元素相乘
        }

        // 计算权重梯度和偏置梯度
        Matrix weightGradient = gradient.multiply(lastInput.transpose());  // dW = gradient * inputᵀ
        Matrix biasGradient = gradient;  // dB = gradient

        // 梯度裁剪：防止梯度爆炸
        double clipValue = 5.0;

        // 更新权重：W = W - learningRate * clip(dW)
        weights = weights.subtract(weightGradient.applyFunction(x -> {
            double clipped = Math.max(-clipValue, Math.min(clipValue, x));  // 裁剪到[-5, 5]
            return clipped * learningRate;
        }));

        // 更新偏置：b = b - learningRate * clip(dB)
        biases = biases.subtract(biasGradient.applyFunction(x -> {
            double clipped = Math.max(-clipValue, Math.min(clipValue, x));
            return clipped * learningRate;
        }));

        // 计算传给上一层的梯度：Wᵀ * gradient
        return weights.transpose().multiply(gradient);
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }

    public Matrix getLastOutput() {
        return lastOutput;
    }
}