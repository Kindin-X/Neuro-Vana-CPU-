#pragma once
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <functional>
#include <memory>
#include <stdexcept>
#include <iostream>
#include<functional>

class Tensor;
//用于张量切片操作的结构体
struct Slice {
    int start;
    int end;
    int step;

    // 构造函数，默认步长为1
    Slice(int s, int e, int st = 1) : start(s), end(e), step(st) {}

    // 静态方法，用于表示某个维度的全切片
    static Slice all(int dim_size) {
        return Slice(0, dim_size, 1);
    }
};


//辅助函数声明：
// 
//广播辅助函数
std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b);
std::vector<size_t> get_multi_index(size_t flat_index, const std::vector<size_t>& shape);
std::vector<size_t> compute_operand_index(const std::vector<size_t>& multi_index, const std::vector<size_t>& operand_shape);
//张量拼接辅助函数
void copyTensorRecursive(const Tensor& src, Tensor& dst, int axis,
    const std::vector<size_t>& src_shape, std::vector<size_t>& index,
    int dim, int offset);
//用于切片的辅助函数
void copySliceRecursive(const Tensor& src, Tensor& dst,
    const std::vector<Slice>& slices,
    std::vector<size_t>& dst_index,
    std::vector<size_t>& src_index,
    int dim);



//智能指针的使用
using TensorPtr = std::shared_ptr<Tensor>;
using BackwardFn = std::function<void(TensorPtr)>;




class Tensor : public std::enable_shared_from_this<Tensor> {

protected:
    std::vector<double> data;// 数据存储，使用一维数组表示多维数据
    std::vector<size_t> strides;

    bool requires_grad = false;//是否需要梯度，默认false
    TensorPtr grad = nullptr;

    BackwardFn grad_fn = nullptr;
   
    size_t compute_flat_index(const std::vector<size_t>& indices) const;
public:
    std::string name = "";
    std::vector<size_t> shape; // 形状，每个维度的长度
    std::vector<TensorPtr> parents;//父节点
    // 使用工厂构造，禁止直接 new,因为图的多次连接涉及到share_from_this,防止崩溃
    static TensorPtr create(const std::vector<double>& values, const std::vector<size_t>& shape, bool requires_grad = false, const std::string& name = "") {
        TensorPtr t = std::shared_ptr<Tensor>(new Tensor(values, shape, requires_grad));
        t->name = name;
        return t;
    }
    //友元类
    friend class AdamWOptimizer;
    friend class AdamOptimizer;
    friend class SGDOptimizer;
    //友元函数
    
    friend void clip_gradients(const std::vector<TensorPtr>& parameters, double max_norm);

    // 构造函数：根据给定形状创建Tensor
    Tensor(const std::vector<size_t> shape,bool requires_grad=false);
    //给定数据块来创建tensor
    Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad = false);

    // 索引操作：返回指定位置元素的引用
    double& operator()(const std::vector<size_t>& indices);

    // 常量索引操作：返回指定位置元素的常量引用
    const double& operator()(const std::vector<size_t>& indices) const;

    //打印输出张量的形状
    //访问梯度接口
    void get_shape_print();
    std::vector<double> get_grad_data() const {
        if (!grad) throw std::runtime_error("Gradient not initialized");
        return grad->data;
    }
    void print_grad();
    //得到形状和数据
    const std::vector<size_t>& get_shape() const;
    std::vector<double>& get_data();
    const std::vector<double>& get_data() const;
    bool get_requires_grad() const;
    TensorPtr get_grad() const;
    //建立梯度
    
    void set_grad(TensorPtr grad_tensor);
    void init_grad();
    void set_grad_fn(BackwardFn fn);
    void backward();
    void zero_grad();
    std::string to_string() const;



    // 基础运算：元素级加法（支持广播机制与自动求导）
    friend TensorPtr add(TensorPtr a, TensorPtr b,bool is_sub);
    
    Tensor operator+(const Tensor& other) const;

    // 元素级乘法
    friend TensorPtr multiply(const TensorPtr& a, const TensorPtr& b);

    Tensor operator*(const Tensor& other) const;

    // 矩阵乘法（仅限2D Tensor）
    friend TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
    Tensor matmul(const Tensor& other) const;
    //矩阵转置
    Tensor Transpose() const;
     friend TensorPtr transpose(TensorPtr a, const std::vector<size_t>& dims);
    // 静态方法：创建全一Tensor
    static Tensor ones(const std::vector<size_t>& shape);
    // 工厂方法：创建全 0 张量
    static TensorPtr zeros(const std::vector<size_t>& shape, bool requires_grad = false, const std::string& name = "") {
        size_t total_size = 1;
        for (auto dim : shape) total_size *= dim;
        std::vector<double> values(total_size, 0.0);
        return Tensor::create(values, shape, requires_grad, name);
    }


    //逆矩阵
    Tensor inverse();
    //reshape
    Tensor reshape(const std::vector<size_t>&new_shape) const;
    //mean
    friend TensorPtr mean(const TensorPtr& a);
    //sum
    friend TensorPtr sum(const TensorPtr& a, const std::vector<size_t>& axes);
    //reshape
    friend TensorPtr reshape(TensorPtr a, const std::vector<size_t>& new_shape);
    //归一化
    void normalize();
    //反归一化
    void denormalize();

    //切片

    Tensor slice(const std::vector<Slice>& slices_in) const;
    //打印
    void print()const;

    //友元函数：
    // 切片复制
    friend void copySliceRecursive(const Tensor& src, Tensor& dst,
        const std::vector<Slice>& slices,
        std::vector<int>& dst_index,
        std::vector<int>& src_index,
        int dim);

    // 沿指定轴拼接两个Tensor
    friend Tensor concat(const Tensor& a, const Tensor& b, int axis);
    // 计算一维数组中的索引
    int calculate_index(const std::vector<size_t>& indices) const;
private:
    static double data_min;
    static double data_max;
   //张量加法
     static Tensor broadcast_add(const Tensor& a, const Tensor& b);

};

