#pragma once
#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
#include <random>

// 带自动求导的逐元素 sigmoid 激活函数实现
// 输入: x 是一个 TensorPtr，输出 shape 与输入相同
// 反向传播链式法则: dy/dx = y * (1 - y)
//sigmoid
template<typename TensorPtr>
TensorPtr sigmoid(const TensorPtr& x) {
    const auto& in = x->get_data();
    std::vector<double> out_data(in.size());
    #pragma omp parallel for
    for (int i = 0; i < in.size(); ++i) {
        out_data[i] = 1.0 / (1.0 + std::exp(-in[i]));
    }
    TensorPtr out = Tensor::create(out_data, x->get_shape(), x->get_requires_grad());
    out->parents.push_back(x);
    if (out->get_requires_grad()) {
        out->set_grad_fn([x, out](TensorPtr grad_output) {
            if (!x->get_grad()) x->init_grad();
            auto& xg = x->get_grad()->get_data();
            const auto& yg = out->get_data();
            const auto& go = grad_output->get_data();
            #pragma omp parallel for
            for (int i = 0; i < yg.size(); ++i) {
                xg[i] += go[i] * yg[i] * (1.0 - yg[i]);
            }
            });
    }
    return out;
}




// 带自动求导的逐元素 tanh 激活函数实现
// 输入: x 是一个 TensorPtr，输出 shape 与输入相同
// 反向传播链式法则: dy/dx = 1 - y^2
//tanh

template<typename TensorPtr>
TensorPtr tanh_activation(const TensorPtr& x) {
    const auto& in = x->get_data();
    std::vector<double> out_data(in.size());
    #pragma omp parallel for
    for (int i = 0; i < in.size(); ++i) {
        out_data[i] = std::tanh(in[i]);
    }
    TensorPtr out = Tensor::create(out_data, x->get_shape(), x->get_requires_grad());
    out->parents.push_back(x);
    if (out->get_requires_grad()) {
        out->set_grad_fn([x, out](TensorPtr grad_output) {
            if (!x->get_grad()) x->init_grad();
            auto& xg = x->get_grad()->get_data();
            const auto& yg = out->get_data();
            const auto& go = grad_output->get_data();
            #pragma omp parallel for
            for (int i = 0; i < yg.size(); ++i) {
                xg[i] += go[i] * (1.0 - yg[i] * yg[i]);
            }
            });
    }
    return out;
}


// --------------------------- Layer Normalization ---------------------------
// 对每个样本（列）做特征归一化：
//   y = (x - μ) / sqrt(σ? + eps)
// 这里的 μ、σ? 都是针对 shape [hidden_size, batch_size] 中的每一列计算。
// 返回的 out 会继承 x.requires_grad，支持反向传播。
inline TensorPtr layer_norm(const TensorPtr& x, double eps = 1e-5) {
    const auto& X = x->get_data();
    const auto& shape = x->get_shape();
    size_t H = shape[0], B = shape[1];
    // 1) 计算每列的均值 μ[j]
    std::vector<double> mean(B, 0.0);
    for (size_t j = 0; j < B; ++j) {
        double s = 0;
        for (size_t i = 0; i < H; ++i) {
            s += X[i * B + j];
        }
        mean[j] = s / static_cast<double>(H);
    }
    // 2) 计算每列的方差 σ?[j]
    std::vector<double> var(B, 0.0);
    for (size_t j = 0; j < B; ++j) {
        double s2 = 0;
        for (size_t i = 0; i < H; ++i) {
            double d = X[i * B + j] - mean[j];
            s2 += d * d;
        }
        var[j] = s2 / static_cast<double>(H);
    }
    // 3) 输出归一化结果
    std::vector<double> Y(H * B);
    for (size_t j = 0; j < B; ++j) {
        double inv_std = 1.0 / std::sqrt(var[j] + eps);
        for (size_t i = 0; i < H; ++i) {
            Y[i * B + j] = (X[i * B + j] - mean[j]) * inv_std;
        }
    }
    // 创建 TensorPtr，保留梯度需求
    TensorPtr out = Tensor::create(Y, shape, x->get_requires_grad());
    // 如果需要梯度，可自行添加 grad_fn 实现或后续扩展
    return out;
}

// ------------------------------ Dropout ------------------------------
// 在训练时随机置零比例 p 的元素，并按 1/(1-p) 缩放。
// 返回的张量 inherits x.requires_grad，dropout mask 不参与反向传播（mask 对 grad 直接截断）。
inline TensorPtr dropout(const TensorPtr& x, double p) {
    if (p <= 0.0 || p >= 1.0) return x;  // p=0 或 1 时直接返回
    const auto& shape = x->get_shape();
    const auto& X = x->get_data();
    size_t N = X.size();

    // 1) 构造 mask
    std::vector<double> mask(N);
    std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(1.0 - p);
    for (size_t i = 0; i < N; ++i) {
        mask[i] = dist(gen) ? (1.0 / (1.0 - p)) : 0.0;
    }
    TensorPtr M = Tensor::create(mask, shape, false);

    // 2) 输出 x * mask
    return multiply(x, M);
}
