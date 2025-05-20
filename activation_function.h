#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
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
