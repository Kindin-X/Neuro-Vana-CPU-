#include<iostream>
#include<string>
#include "tensor.h"
#include"autograd.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <omp.h>
#include <Eigen/Dense>
#include <sstream>
#include <unordered_set>





//广播机制的辅助函数：

// 辅助函数：计算广播后的形状
// 如果两个维度不匹配且都不为1，则抛出异常
std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) {
    int n = std::max(shape_a.size(), shape_b.size());
    std::vector<size_t> a = shape_a, b = shape_b;
    // 前置补 1
    while (a.size() < n) a.insert(a.begin(), 1);
    while (b.size() < n) b.insert(b.begin(), 1);
    std::vector<size_t> result(n);
    for (int i = 0; i < n; ++i) {
        size_t dim_a = a[i];
        size_t dim_b = b[i];
        if (dim_a == dim_b) {
            result[i] = dim_a;
        }
        else if (dim_a == 1) {
            result[i] = dim_b;
        }
        else if (dim_b == 1) {
            result[i] = dim_a;
        }
        else {
            throw std::invalid_argument("形状不匹配，无法进行广播");
        }
    }

    return result;
}

// 辅助函数：根据 flat_index 和 shape 计算多维索引（最后一维变化最快）
std::vector<size_t> get_multi_index(size_t flat_index, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
        indices[i] = flat_index % shape[i];
        flat_index /= shape[i];
    }
    return indices;
}

// 辅助函数：将广播后结果的多维索引转换为操作数对应的索引
std::vector<size_t> compute_operand_index(const std::vector<size_t>& multi_index, const std::vector<size_t>& operand_shape) {
    int n_result = multi_index.size();
    int n_operand = operand_shape.size();
    int offset = n_result - n_operand; // 补齐前面的维度
    std::vector<size_t> operand_index(n_operand);
    for (int i = 0; i < n_operand; ++i) {
        // 若该维度原 shape 为1，则始终取0；否则取对应位置
        operand_index[i] = (operand_shape[i] == 1) ? 0 : multi_index[i + offset];
    }
    return operand_index;
}


//定义全局变量，用于归一化和反归一化
double Tensor::data_min = 0.0; 
double Tensor::data_max = 0.0;

   
    // 构造函数：根据给定形状创建Tensor
Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
    : shape(std::move(shape)), requires_grad(requires_grad) {
    size_t total_size = 1;
    for (auto dim : this->shape) total_size *= dim;
    data.resize(total_size, 0.0);
}
//给定数据块来创建tensor
Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad)
    : data(std::move(values)), shape(std::move(shape)), requires_grad(requires_grad) {
    size_t total_size = 1;
    for (auto dim : this->shape) total_size *= dim;
    if (data.size() != total_size) throw std::invalid_argument("Shape doesn't match data size.");
}


    // 索引操作：返回指定位置元素的引用
    double& Tensor::operator()(const std::vector<size_t>& indices) {
        return data[calculate_index(indices)];
    }

    // 常量索引操作：返回指定位置元素的常量引用
    const double& Tensor::operator()(const std::vector<size_t>& indices) const {
        return data[calculate_index(indices)];
    }

    size_t Tensor::compute_flat_index(const std::vector<size_t>& indices) const {
        size_t index = 0, stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }




    //输出形状
    void Tensor::get_shape_print() {
        for (int dim : shape) {
            std::cout << dim << " ";
        }

    }
    void Tensor::print_grad() {
        if (!grad) {
            std::cout << "[警告] grad 未初始化！" << std::endl;
            return;
        }
        for (double v : this->get_grad()->get_data())
            std::cout << v << " ";
    }
    //访问部分（形状，数据，梯度需求，梯度，）
    const std::vector<size_t>& Tensor::get_shape() const { return shape; }
    std::vector<double>& Tensor::get_data() {return data;}

    const std::vector<double>& Tensor::get_data() const { return data; }
    bool Tensor::get_requires_grad() const { return requires_grad; }

    TensorPtr Tensor::get_grad() const { return grad; }
    //建立梯度，建立图，
    void Tensor::set_grad(TensorPtr grad_tensor) { grad = grad_tensor; }
    void Tensor::init_grad() {
        TensorPtr grad_tensor = std::make_shared<Tensor>(this->shape, false);
        this->set_grad(grad_tensor);
    }
    void Tensor::set_grad_fn(BackwardFn fn) { grad_fn = fn; }
    //反向传播
        void Tensor::backward() {
            if (!requires_grad) return;

            // 初始化 grad 为全 1（标量输出）
            if (!grad) {
                grad = std::make_shared<Tensor>(shape, false);
                for (double& v : grad->get_data())
                    v = 1.0;
            }

            std::unordered_set<Tensor*> visited;
            std::function<void(TensorPtr)> topo = [&](TensorPtr t) {
                if (visited.count(t.get())) return;
                visited.insert(t.get());

                if (t->grad_fn) t->grad_fn(t->grad);
                for (auto& p : t->parents) {
                    topo(p);
                }
                };

            topo(shared_from_this());
        }

    

    void Tensor::zero_grad() {
        if (grad) {
            std::fill(grad->data.begin(), grad->data.end(), 0.0);
        }
    }




    //****基础运算模块*********************************************************************************************************
    
   

    // 元素级加法，支持广播
    Tensor Tensor::operator+(const Tensor& other) const {
        TensorPtr a = std::make_shared<Tensor>(*this);
        TensorPtr b = std::make_shared<Tensor>(other);
        return *add(a, b);
    }

    // 元素级乘法，支持广播
    Tensor Tensor::operator*(const Tensor& other) const {
        TensorPtr a = std::make_shared<Tensor>(*this);
        TensorPtr b = std::make_shared<Tensor>(other);
        return *multiply(a, b);
    }

    // 矩阵乘法（仅限2D Tensor）
    Tensor Tensor::matmul(const Tensor & other) const {
            if (shape.size() != 2 || other.shape.size() != 2)
                throw std::invalid_argument("只支持2D矩阵乘法");
            if (shape[1] != other.shape[0])
                throw std::invalid_argument("矩阵维度不匹配");

            size_t m = shape[0];
            size_t n = shape[1];
            size_t p = other.shape[1];

            // Eigen::Map 直接包装已有 data，避免复制
            Eigen::Map<const Eigen::MatrixXd> A(this->data.data(), m, n);
            Eigen::Map<const Eigen::MatrixXd> B(other.data.data(), n, p);
            Eigen::MatrixXd C = A * B;

            // Eigen::MatrixXd 是 column-major，但我们统一使用 flat vector 存储
            std::vector<double> result_data(C.data(), C.data() + m * p);

            return Tensor(result_data, { m, p });
        }

    

    
    // 静态方法：创建全一Tensor
    Tensor Tensor::ones(const std::vector<size_t>& shape) {
        Tensor t(shape);
        for (int i = 0; i < t.data.size(); i++) {
            t.data[i] = 1.0;
        }
        return t;
    
    }

    //**************************************************************求逆矩阵************************************************************************
    Tensor Tensor::inverse() {
        if (shape.size() != 2) throw std::invalid_argument("形状不匹配");
        if (shape[0] != shape[1]) throw std::invalid_argument("矩阵必须为方阵");

        int n = shape[0];
        std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));

        // 直接从 data 读取矩阵
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                aug[i][j] = data[i * n + j]; // 直接索引，避免转换为二维向量
            }
            aug[i][n + i] = 1.0;  // 单位矩阵
        }

        const double eps = 1e-12;
        for (int i = 0; i < n; ++i) {
            int pivot = i;
            for (int row = i; row < n; ++row) {
                if (std::fabs(aug[row][i]) > std::fabs(aug[pivot][i])) {
                    pivot = row;
                }
            }
            if (std::fabs(aug[pivot][i]) < eps) {
                throw std::runtime_error("矩阵不可逆");
            }
            std::swap(aug[i], aug[pivot]);

            double pivotVal = aug[i][i];
            for (int j = 0; j < 2 * n; ++j) {
                aug[i][j] /= pivotVal;
            }

            for (int row = 0; row < n; ++row) {
                if (row != i) {
                    double factor = aug[row][i];
                    for (int j = 0; j < 2 * n; ++j) {
                        aug[row][j] -= factor * aug[i][j];
                    }
                }
            }
        }

        // 直接操作 inv.data
        Tensor inv({ static_cast<size_t>(n),static_cast<size_t>( n) });
        inv.data.resize(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                inv.data[i * n + j] = aug[i][n + j]; // 行优先存储

        return inv;
    }

     
  
    //**********************************************切片********************************************************
   

   

    // 递归复制数据
    void copySliceRecursive(const Tensor& src, Tensor& dst,
        const std::vector<Slice>& slices,
        std::vector<size_t>& dst_index,
        std::vector<size_t>& src_index,
        int dim) {
        if (dim == slices.size()) {
            // 当所有维度都确定后，将对应元素复制到结果张量中
            dst(dst_index) = src(src_index);
            return;
        }

        // 对当前维度，根据切片步长依次遍历
        for (int i = 0; i < dst.shape[dim]; i++) {
            // 计算原张量中对应的索引位置：起始位置 + 当前步数 * 步长
            src_index[dim] = slices[dim].start + i * slices[dim].step;
            dst_index[dim] = i;
            copySliceRecursive(src, dst, slices, dst_index, src_index, dim + 1);
        }
    }

    Tensor Tensor::slice(const std::vector<Slice>& slices_in) const {
        // 对于维度不足的情况自动补全为全切片
        std::vector<Slice> slices = slices_in;
        if (slices.size() < shape.size()) {
            for (size_t i = slices.size(); i < shape.size(); i++) {
                slices.push_back(Slice::all(shape[i]));
            }
        }
        if (slices.size() > shape.size()) {
            throw std::invalid_argument("切片维度多于张量的实际维度");
        }

        // 验证各个维度的切片边界，并计算结果张量的形状
        std::vector<size_t> result_shape(slices.size());
        for (size_t i = 0; i < slices.size(); i++) {
            if (slices[i].start < 0 || slices[i].end > shape[i] || slices[i].start > slices[i].end) {
                throw std::out_of_range("第 " + std::to_string(i) + " 维度的切片超出张量边界");
            }
            // 计算该维度切片后的大小（向上取整）
            int dim_size = (slices[i].end - slices[i].start + slices[i].step - 1) / slices[i].step;
            result_shape[i] = dim_size;
        }

        // 根据计算得到的形状构造结果张量
        Tensor result(result_shape);

        // 初始化索引向量，分别用于原张量和结果张量
        std::vector<size_t> dst_index(result_shape.size(), 0);
        std::vector<size_t> src_index(shape.size(), 0);

        copySliceRecursive(*this, result, slices, dst_index, src_index, 0);

        return result;
    }

    //*************************************归一化与反归一化*************************************************************************

    // 归一化
    void Tensor::normalize() {
        if (data.empty()) throw std::runtime_error("这是空张量!");

        // 计算最小值和最大值
        data_min = *std::min_element(data.begin(), data.end());
        data_max = *std::max_element(data.begin(), data.end());

        // 防止除零错误
        if (data_max == data_min) throw std::runtime_error("所有的数据都是相同的!");

        // 归一化操作
        for (double& val : data) {
            val = (val - data_min) / (data_max - data_min);
        }
    }

    // 反归一化
    void Tensor::denormalize() {
        if (data.empty()) throw std::runtime_error("这是空张量!");

        // 检查输入数据是否在 [0,1] 之间
        for (const double& val : data) {
            if (val < 0.0 || val > 1.0) {
                throw std::runtime_error("这不是一个能够反归一化的张量，数据超过 [0,1]!");
            }
        }

        // 反归一化操作
        for (double& val : data) {
            val = val * (data_max - data_min) + data_min;
        }
    }
    //***********************************************打印Tensor****************************************************
    void Tensor::print() const{
        if (shape.size() == 2) {
            // 处理二维矩阵
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    std::cout << (*this)({ static_cast<size_t>(i), static_cast<size_t>(j) }) << " ";
                }
                std::cout << std::endl;
            }
        }
        else {
            // 处理其他维度
            std::cout << "[";
            for (size_t i = 0; i < data.size(); i++) {
                std::cout << data[i];
                if (i != data.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
    
    
    }
    //************************************************转置矩阵******************************
    Tensor Tensor::Transpose() const {
        if (shape.size() != 2)
            throw std::invalid_argument("只支持2D矩阵转置");

        Tensor transposed({ shape[1], shape[0] });  // 交换shape
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                transposed({ static_cast<size_t>(j), static_cast<size_t>(i) }) = (*this)({ static_cast<size_t>(i), static_cast<size_t>(j) });  // 访问当前 Tensor 的数据
            }
        }
        return transposed;
    }
//**********************************************reshape************************************************************
    Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
        // 计算原始总元素数
        size_t total_old = 1;
        for (size_t dim : shape)
            total_old *= dim;

        // 计算新形状总元素数
        size_t total_new = 1;
        for (size_t dim : new_shape)
            total_new *= dim;

        // 检查元素个数是否一致
        if (total_old != total_new) {
            throw std::runtime_error("reshape: total elements mismatch.");
        }

        // 创建一个新的 Tensor，拷贝当前数据，但是用新 shape 替换原来的
        Tensor result(*this);  // 复制当前 Tensor（数据和其他信息也复制）
        result.shape = new_shape;  // 更新新形状
        // 注意：如果有 strides 也可以重新计算，现阶段简单起见不做复杂处理

        return result;
    }



    //*******************************************索引**********************************************************************

    // 计算一维数组中的索引
    int Tensor::calculate_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size())  throw std::out_of_range("索引超出范围"); 
            int index = 0;
            int stride = 1;
            for (int i = shape.size() - 1; i >= 0; --i) {
                if (indices[i] >= 0 && indices[i] < shape[i]) {
                    index += indices[i] * stride;
                    stride *= shape[i];
                }
            }

            return index;
      
    }



    //*************************************************拼接矩阵*************************************************************************************
// 辅助函数：递归复制 src Tensor 的所有元素到 dst Tensor 中。
// 参数 offset 用于在拼接的轴上调整索引，
// 对于第一个 Tensor 传 0，第二个 Tensor 传 a.shape[axis]
    void copyTensorRecursive(const Tensor& src, Tensor& dst, int axis,
        const std::vector<size_t>& src_shape, std::vector<size_t>& index,
        int dim, int offset) {
        if (dim == src_shape.size()) {
            // 到达叶子节点，复制元素
            std::vector<size_t> dst_index = index;
            dst_index[axis] += offset;  // 在拼接轴上加上偏移量
            dst(dst_index) = src(index);
            return;
        }
        for (int i = 0; i < src_shape[dim]; ++i) {
            index[dim] = i;
            copyTensorRecursive(src, dst, axis, src_shape, index, dim + 1, offset);
        }
    }

    //沿指定轴拼接两个 Tensor
    Tensor concat(const Tensor& a, const Tensor& b, int axis) {
        // 检查两个 Tensor 的维度数是否一致
        if (a.shape.size() != b.shape.size()) {
            throw std::invalid_argument("形状不匹配: 维度数不同");
        }
        // 检查 axis 范围
        if (axis < 0 || axis >= a.shape.size()) {
            throw std::invalid_argument("axis 超出范围");
        }
        // 检查除拼接轴以外的维度是否相同
        for (size_t i = 0; i < a.shape.size(); i++) {
            if (i != static_cast<size_t>(axis) && a.shape[i] != b.shape[i]) {
                throw std::invalid_argument("形状不匹配: 非拼接轴上的维度不同");
            }
        }
        // 计算拼接后的新形状
        std::vector<size_t> new_shape = a.shape;
        new_shape[axis] += b.shape[axis];

        Tensor result(new_shape);

        // 用递归函数复制 a 和 b 的元素到 result 中
        std::vector<size_t> index(a.shape.size(), 0);
        // 复制 a 到 result，沿拼接轴偏移量为 0
        copyTensorRecursive(a, result, axis, a.shape, index, 0, 0);
        // 复制 b 到 result，沿拼接轴偏移量为 a.shape[axis]
        copyTensorRecursive(b, result, axis, b.shape, index, 0, a.shape[axis]);

        return result;
    }

//********************************************日志**************************************************
    //日志：用于检查某个张量的状态
    std::string Tensor::to_string() const {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            oss << shape[i];
            if (i != shape.size() - 1) oss << ", ";
        }
        oss << "], data=[";
        for (size_t i = 0; i < data.size(); ++i) {
            oss << data[i];
            if (i != data.size() - 1) oss << ", ";
        }
        oss << "])";
        return oss.str();
    }

//补充：
  //加法逻辑(广播)：
 
    Tensor Tensor::broadcast_add(const Tensor& a, const Tensor& b) {
        std::vector<size_t> result_shape = compute_broadcast_shape(a.shape, b.shape);
        Tensor result(result_shape);
        size_t total = 1;
        for (size_t d : result_shape)
            total *= d;

        for (size_t i = 0; i < total; ++i) {
            auto multi_index = get_multi_index(i, result_shape);
            auto idx_a = compute_operand_index(multi_index, a.shape);
            auto idx_b = compute_operand_index(multi_index, b.shape);
            result.data[i] = a.data[a.calculate_index(idx_a)] + b.data[b.calculate_index(idx_b)];
        }

        return result;
    }



    
