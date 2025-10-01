#pragma once
#include "tensor.h"

// 工具函数：广播反向梯度还原
Tensor reduce_grad_to_shape(const Tensor& grad, const std::vector<size_t>& target_shape);

// 自动求导算子
TensorPtr add(TensorPtr a, TensorPtr b,bool is_sub=false);
TensorPtr multiply(const TensorPtr& a, const TensorPtr& b);
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
// （1）求和（当前只支持全元素求和，即 axes 为空时求标量和）
TensorPtr sum(const TensorPtr& a, const std::vector<size_t>& axes = {});
//mean
TensorPtr mean(const TensorPtr& a);

// （2）重塑
TensorPtr reshape(TensorPtr a, const std::vector<size_t>& new_shape);

// （3）转置（这里实现了2D的转置，即 dims 必须为 {1, 0}）
TensorPtr transpose(TensorPtr a, const std::vector<size_t>& dims);
//可视化
void export_dot(const TensorPtr& root, const std::string& filename = "graph.dot");
