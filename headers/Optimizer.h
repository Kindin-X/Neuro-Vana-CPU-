#pragma once
#include<iostream>
#include<vector>
#include <unordered_map>
#include"tensor.h"

// 基类 Optimizer
class Optimizer {
public:
    double lr;  // 公共学习率

    explicit Optimizer(double learning_rate) : lr(learning_rate) {}
    virtual ~Optimizer() = default;  // 虚析构函数确保正确释放资源
    virtual void step(const std::vector<TensorPtr>& parameters) = 0;  // 纯虚函数接口
};




class SGDOptimizer : public Optimizer {
public:
    explicit SGDOptimizer(double lr = 0.01) : Optimizer(lr) {}

    void step(const std::vector<TensorPtr>& parameters)override {
        for (const auto& param : parameters) {
            if (!param->requires_grad || !param->grad) continue;

            auto& grad_data = param->grad->data; 
            auto& data = param->data;

            for (size_t i = 0; i < data.size(); ++i) {
                data[i] -= lr * grad_data[i];
            }
        }
    }
};



class AdamOptimizer : public Optimizer {
public:
    double beta1;
    double beta2;
    double epsilon;
    double beta1_pow_t;
    double beta2_pow_t;
    int timestep;

    std::unordered_map<TensorPtr, std::vector<double>> m;
    std::unordered_map<TensorPtr, std::vector<double>> v;

    AdamOptimizer(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : Optimizer(lr), beta1(beta1), beta2(beta2), epsilon(epsilon),
        beta1_pow_t(1.0), beta2_pow_t(1.0), timestep(0) {}

    void step(const std::vector<TensorPtr>& parameters) override {
        timestep += 1;
        beta1_pow_t *= beta1;
        beta2_pow_t *= beta2;

        for (const auto& param : parameters) {
            if (!param->requires_grad || !param->grad) continue;

            auto& grad = param->grad->data;
            auto& data = param->data;

            if (m.find(param) == m.end()) {
                m[param] = std::vector<double>(data.size(), 0.0);
                v[param] = std::vector<double>(data.size(), 0.0);
            }

            auto& m_vec = m[param];
            auto& v_vec = v[param];

            for (size_t i = 0; i < data.size(); ++i) {
                m_vec[i] = beta1 * m_vec[i] + (1 - beta1) * grad[i];
                v_vec[i] = beta2 * v_vec[i] + (1 - beta2) * grad[i] * grad[i];

                double m_hat = m_vec[i] / (1 - beta1_pow_t);
                double v_hat = v_vec[i] / (1 - beta2_pow_t);

                data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }
};



class AdamWOptimizer : public Optimizer {
public:
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    double beta1_pow_t;
    double beta2_pow_t;
    int timestep;

    std::unordered_map<TensorPtr, std::vector<double>> m;
    std::unordered_map<TensorPtr, std::vector<double>> v;

    AdamWOptimizer(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
        double epsilon = 1e-8, double weight_decay = 0.01)
        : Optimizer(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay),
        beta1_pow_t(1.0), beta2_pow_t(1.0), timestep(0) {}

    void step(const std::vector<TensorPtr>& parameters) override {
        timestep += 1;
        beta1_pow_t *= beta1;
        beta2_pow_t *= beta2;

        for (const auto& param : parameters) {
            if (!param->requires_grad || !param->grad) continue;

            auto& grad = param->grad->data;
            auto& data = param->data;

            if (m.find(param) == m.end()) {
                m[param] = std::vector<double>(data.size(), 0.0);
                v[param] = std::vector<double>(data.size(), 0.0);
            }

            auto& m_vec = m[param];
            auto& v_vec = v[param];

            for (size_t i = 0; i < data.size(); ++i) {
                // Weight Decay 是直接作用于权重本身
                grad[i] += weight_decay * data[i];

                m_vec[i] = beta1 * m_vec[i] + (1 - beta1) * grad[i];
                v_vec[i] = beta2 * v_vec[i] + (1 - beta2) * grad[i] * grad[i];

                double m_hat = m_vec[i] / (1 - beta1_pow_t);
                double v_hat = v_vec[i] / (1 - beta2_pow_t);

                data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }
};
