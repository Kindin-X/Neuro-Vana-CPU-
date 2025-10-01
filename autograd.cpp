#include "autograd.h"
#include<fstream>
#include<unordered_set>
#include<functional>
#include<memory>
#include<string>
#include<iostream>
#include<cstdint>
#include <omp.h>
#include <stdexcept>
#include <algorithm>
// 自动求导算子
//元素级乘法
TensorPtr multiply(const TensorPtr& a, const TensorPtr& b) {
    std::vector<size_t> result_shape = compute_broadcast_shape(a->get_shape(), b->get_shape());
    Tensor result(result_shape);
    size_t total = result.get_data().size();
    int N = static_cast<int>(total);
   
    for (int i = 0; i < N; ++i) {
        auto multi_index = get_multi_index(i, result_shape);
        auto idx_a = compute_operand_index(multi_index, a->get_shape());
        auto idx_b = compute_operand_index(multi_index, b->get_shape());
        result.get_data()[i] = a->get_data()[a->calculate_index(idx_a)] *
                               b->get_data()[b->calculate_index(idx_b)];
    }

    TensorPtr out = Tensor::create(result.get_data(), result.get_shape(), a->get_requires_grad() || b->get_requires_grad());

    //  自动图构建
    out->parents.push_back(a);
    out->parents.push_back(b);

    if (out->get_requires_grad()) {
        out->set_grad_fn([a, b](TensorPtr grad_output) {
            if (a->get_requires_grad()) {
                Tensor grad_a_raw = *multiply(grad_output, b);  // dz/da = grad_output * b
                Tensor grad_a = reduce_grad_to_shape(grad_a_raw, a->get_shape());
                if (!a->get_grad()) a->init_grad();
                #pragma omp parallel for
                for (int i = 0; i < grad_a.get_data().size(); ++i)
                    a->get_grad()->get_data()[i] += grad_a.get_data()[i];
            }
            if (b->get_requires_grad()) {
                Tensor grad_b_raw = *multiply(grad_output, a);  // dz/db = grad_output * a
                Tensor grad_b = reduce_grad_to_shape(grad_b_raw, b->get_shape());
                if (!b->get_grad()) b->init_grad();
                #pragma omp parallel for
                for (int i = 0; i < grad_b.get_data().size(); ++i)
                    b->get_grad()->get_data()[i] += grad_b.get_data()[i];
            }
            });
    }

    return out;
}
//矩阵乘法
// tensor_ops.cpp

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
    // 维度检查
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::invalid_argument("matmul 只支持二维张量");
    }

    // 计算前向传播
    Tensor result = a->matmul(*b);  // 使用 Eigen 实现的 Tensor::matmul

    // 创建新张量并继承计算图依赖
    TensorPtr out = Tensor::create(result.get_data(), result.get_shape(),
        a->get_requires_grad() || b->get_requires_grad());

    out->parents.push_back(a);
    out->parents.push_back(b);

    // 注册反向传播
    if (out->get_requires_grad()) {
        out->set_grad_fn([a, b](TensorPtr grad_output) {
            if (!grad_output) return;

            // dL/dA = dL/dY * B^T
            if (a->get_requires_grad()) {
                Tensor grad_a_raw = grad_output->matmul(b->Transpose());
                if (!a->get_grad()) a->init_grad();
                auto& a_grad_data = a->get_grad()->get_data();
                auto& raw = grad_a_raw.get_data();
#pragma omp parallel for
                for (int i = 0; i < static_cast<int>(raw.size()); ++i)
                    a_grad_data[i] += raw[i];
            }

            // dL/dB = A^T * dL/dY
            if (b->get_requires_grad()) {
                Tensor grad_b_raw = a->Transpose().matmul(*grad_output);
                if (!b->get_grad()) b->init_grad();
                auto& b_grad_data = b->get_grad()->get_data();
                auto& raw = grad_b_raw.get_data();
#pragma omp parallel for
                for (int i = 0; i < static_cast<int>(raw.size()); ++i)
                    b_grad_data[i] += raw[i];
            }
            });
    }

    return out;
}


//加法

TensorPtr add(TensorPtr a, TensorPtr b, bool is_sub) {
    TensorPtr b_effective = b;

    // 如果是减法，构造 -b（但直接在这里修改，不需额外函数）
    if (is_sub) {
        std::vector<double> neg_data = b->get_data();
        for (auto& val : neg_data) {
            val = -val;
        }
        b_effective = Tensor::create(neg_data, b->get_shape(), b->get_requires_grad());
    }

    Tensor result = Tensor::broadcast_add(*a, *b_effective);
    TensorPtr out = Tensor::create(result.get_data(), result.get_shape(), a->get_requires_grad() || b->get_requires_grad());

    // 自动构图：注册父节点
    out->parents.push_back(a);
    out->parents.push_back(b);

    if (out->get_requires_grad()) {
        out->set_grad_fn([a, b, is_sub](TensorPtr grad_output) {
            if (a->get_requires_grad()) {
                Tensor grad_a = reduce_grad_to_shape(*grad_output, a->get_shape());
                if (!a->get_grad()) a->init_grad();
                #pragma omp parallel for
                for (int i = 0; i < grad_a.get_data().size(); ++i)
                    a->get_grad()->get_data()[i] += grad_a.get_data()[i];
            }
            if (b->get_requires_grad()) {
                Tensor grad_b = reduce_grad_to_shape(*grad_output, b->get_shape());
                if (!b->get_grad()) b->init_grad();
                float sign = is_sub ? -1.0f : 1.0f;
                #pragma omp parallel for
                for (int i = 0; i < grad_b.get_data().size(); ++i)
                    b->get_grad()->get_data()[i] += sign * grad_b.get_data()[i];
            }
            });
    }

    return out;
}

//mean
TensorPtr mean(const TensorPtr& a) {
    // 计算所有元素的总和
    double total = 0.0;
    for (double val : a->get_data()) {
        total += val;
    }

    // 获取元素总数
    size_t num_elements = a->get_data().size();

    // 计算平均值
    double mean_val = total / num_elements;

    // 创建标量张量，继承 a 的 requires_grad 属性
    TensorPtr out = std::make_shared<Tensor>(
        std::vector<double>{mean_val},
        std::vector<size_t>{1},
        a->get_requires_grad()
    );

    // 建立计算图，记录父节点
    out->parents.push_back(a);

    // 如果需要梯度，则设置梯度函数
    if (out->get_requires_grad()) {
        out->set_grad_fn([a](TensorPtr grad_output) {
            // 从 a 获取元素总数
            size_t num_elements = a->get_data().size();
            // 计算梯度缩放因子
            double scale = grad_output->get_data()[0] / num_elements;

            // 创建与 a 同形状的梯度张量
            Tensor grad_input(a->get_shape());
            // 将每个元素设置为 scale
            for (size_t i = 0; i < grad_input.get_data().size(); ++i) {
                grad_input.get_data()[i] = scale;
            }

            // 如果 a 的梯度未初始化，则初始化
            if (!a->get_grad()) {
                a->init_grad();
            }
            // 累加梯度到 a 的 grad
            for (size_t i = 0; i < grad_input.get_data().size(); ++i) {
                a->get_grad()->get_data()[i] += grad_input.get_data()[i];
            }
            });
    }

    return out;
}

//sum

TensorPtr sum(const TensorPtr& a, const std::vector<size_t>& axes) {
    if (!axes.empty()) {
        throw std::runtime_error("sum over specific axes not implemented yet.");
    }

    double total = 0.0;
    for (double val : a->get_data())
        total += val;

    // 只返回标量
    TensorPtr out = std::make_shared<Tensor>(
        std::vector<double>{total},
        std::vector<size_t>{1},
        a->get_requires_grad()
    );

    //  建立父节点依赖，保留图结构
    out->parents.push_back(a);

    if (out->get_requires_grad()) {
        out->set_grad_fn([a](TensorPtr grad_output) {
            // 将 scalar 反向传播为与 a 同形状的全 1 倍数张量
            Tensor grad_input(a->get_shape());
            for (size_t i = 0; i < grad_input.get_data().size(); ++i)
                grad_input.get_data()[i] = grad_output->get_data()[0];

            if (!a->get_grad()) a->init_grad();
            for (size_t i = 0; i < grad_input.get_data().size(); ++i)
                a->get_grad()->get_data()[i] += grad_input.get_data()[i];
            });
    }

    return out;
}


// 广播后梯度反向还原
Tensor reduce_grad_to_shape(const Tensor& grad, const std::vector<size_t>& target_shape) {
    const auto& grad_shape = grad.get_shape();
    if (grad_shape == target_shape) return grad;

    Tensor result(target_shape);
    for (size_t i = 0; i < grad.get_data().size(); ++i) {
        std::vector<size_t> multi_index = get_multi_index(i, grad_shape);

        std::vector<size_t> reduced_index;
        for (size_t d = 0; d < target_shape.size(); ++d) {
            if (target_shape[d] == 1)
                reduced_index.push_back(0);
            else
                // target_shape 尾部的维度与 grad 对应
                reduced_index.push_back(multi_index[multi_index.size() - target_shape.size() + d]);
        }

        size_t flat_idx = result.calculate_index(reduced_index);
        result.get_data()[flat_idx] += grad.get_data()[i];
    }
    return result;
}



//reshape
TensorPtr reshape(TensorPtr a, const std::vector<size_t>& new_shape) {
    // 检查元素个数是否一致
    size_t total_old = 1, total_new = 1;
    for (size_t d : a->get_shape()) total_old *= d;
    for (size_t d : new_shape) total_new *= d;
    if (total_old != total_new) {
        throw std::runtime_error("reshape: total elements mismatch.");
    }
    // forward: 拷贝当前 Tensor，更新形状（这里简单赋值，不重新复制 data）
    Tensor result = *a;
    result.shape = new_shape;
    TensorPtr out = std::make_shared<Tensor>(result);
    out->requires_grad = a->get_requires_grad();
    if (a->get_requires_grad()) {
        out->set_grad_fn([a, new_shape](TensorPtr grad_output) {
            // 反向传播时，将 grad_output 重塑回原始形状
            Tensor grad_input = grad_output->reshape(a->get_shape());
            if (!a->get_grad()) a->init_grad();
            for (size_t i = 0; i < grad_input.get_data().size(); ++i)
                a->get_grad()->get_data()[i] += grad_input.get_data()[i];
            });
    }
    return out;
}
//transpose
TensorPtr transpose(TensorPtr a, const std::vector<size_t>& dims) {
    // 目前只支持 2D 转置，要求 dims == {1, 0}
    if (a->get_shape().size() == 2 && dims == std::vector<size_t>{1, 0}) {
        Tensor result = a->Transpose();
        TensorPtr out = std::make_shared<Tensor>(result);
        out->requires_grad = a->get_requires_grad();
        if (a->get_requires_grad()) {
            out->set_grad_fn([a](TensorPtr grad_output) {
                // 2D 情况下，反向传播也是转置
                Tensor grad_input = grad_output->Transpose();
                if (!a->get_grad()) a->init_grad();
                for (size_t i = 0; i < grad_input.get_data().size(); ++i)
                    a->get_grad()->get_data()[i] += grad_input.get_data()[i];
                });
        }
        return out;
    }
    else {
        throw std::runtime_error("transpose for arbitrary dims not implemented yet.");
    }
}


//计算图可视化
void export_dot(const TensorPtr& root, const std::string& filename) {
    std::ofstream fout(filename);
    fout << "digraph ComputationGraph {\n";
    fout << "    rankdir=LR;\n";
    fout << "    node [shape=record, style=filled, fillcolor=lightblue];\n";

    std::unordered_set<Tensor*> visited;

    std::function<void(TensorPtr)> dfs = [&](TensorPtr t) {
        if (!t || visited.count(t.get())) return;
        visited.insert(t.get());

        std::string node_id = "node" + std::to_string(reinterpret_cast<uintptr_t>(t.get()));
        std::string label = t->name.empty() ? node_id : t->name;

        fout << "    " << node_id << " [label=\"" << label << "\\nshape: ";
        for (size_t s : t->get_shape()) fout << s << " ";
        fout << "\", shape=box];\n";

        for (const auto& p : t->parents) {
            std::string parent_id = "node" + std::to_string(reinterpret_cast<uintptr_t>(p.get()));
            fout << "    " << parent_id << " -> " << node_id << ";\n";
            dfs(p);
        }
        };

    dfs(root);
    fout << "}\n";
    fout.close();

    
}
