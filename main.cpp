#include <iostream>
#include "tensor.h" 
#include"autograd.h"
#include"CSVLoader.h"
#include"LSTM.h"
#include"activation_function.h"
#include"train_model.h"
#include"predict_model.h"
#include"save_predictions_to_csv.h"
#include "visualize_prediction.h"
#include<cassert>
#include <iostream>


static double data_min=0;
static double data_max=0;

// 构造一个模拟的股票价格序列（带有趋势和波动）
std::vector<double> generate_synthetic_stock_data(size_t n) {
    std::vector<double> data;
    double price = 100.0; // 初始价
    for (size_t i = 0; i < n; ++i) {
        double trend = 0.2 * i; // 稳步上升趋势
        double noise = ((rand() % 100) / 100.0 - 0.5) * 2.0; // -1 到 1 的噪声
        double spike = (i % 15 == 0) ? 3.0 * ((rand() % 2 == 0) ? 1 : -1) : 0.0; // 模拟跳涨或跳水
        price += trend * 0.02 + noise + spike;
        data.push_back(price);
    }
    return data;
}


int main() {

    std::cout << "=== Test: Tensor Functionalities ===" << std::endl;

    // 构造基本张量并初始化数据
    std::vector<double> vec = { 1, 2, 3, 4, 5, 6 };
    TensorPtr a = Tensor::create(vec, { 2, 3 }, true, "A");
    TensorPtr b = Tensor::create(std::vector<double>{1, 1, 1}, { 1, 3 }, true, "B");

    // 广播加法 + 乘法 + 求和 + 均值 + reshape + 反向传播
    TensorPtr c = add(a, b);  // 广播加法
    TensorPtr d = multiply(c, c);  // 元素乘方
    TensorPtr e = mean(d);         // 标量输出
    e->backward();                 // 执行反向传播

    // 输出梯度
    std::cout << "A.grad: ";
    a->print_grad();
    std::cout << std::endl;

    // 测试切片
    Tensor sliced = a->slice({ Slice(0, 2), Slice(0, 2) });
    std::cout << "Sliced tensor:" << std::endl;
    sliced.print();

    // 测试拼接
    TensorPtr concat_tensor = std::make_shared<Tensor>(concat(*a, *a, 0));
    std::cout << "Concatenated tensor (axis=0):" << std::endl;
    concat_tensor->print();

    // 归一化与反归一化
    a->normalize();
    std::cout << "Normalized A:" << std::endl;
    a->print();
    a->denormalize();
    std::cout << "Denormalized A:" << std::endl;
    a->print();

    // 矩阵乘法与转置
    TensorPtr matA = Tensor::create({ 1, 2, 3, 4 }, { 2, 2 }, false, "MatA");
    TensorPtr matB = Tensor::create({ 5, 6, 7, 8 }, { 2, 2 }, false, "MatB");
    TensorPtr matC = matmul(matA, matB);
    std::cout << "MatMul result:" << std::endl;
    matC->print();

    // 反转置
    TensorPtr matCT = transpose(matC, { 1, 0 });
    std::cout << "Transpose:" << std::endl;
    matCT->print();

    // reshape
    TensorPtr reshaped = reshape(matC, { 4, 1 });
    std::cout << "Reshaped (4x1):" << std::endl;
    reshaped->print();

  

    // 最后转换为 vector<double>，与 `prices` 格式保持一致
    std::vector<double> test_input = reshaped->get_data();
    for (int i = 2; i < 8; ++i) {
        test_input.push_back(std::pow(3 * i, 2));  // 生成平方数序列
    }
    std::cout << "Final flat output:" << std::endl;
    for (auto v : test_input) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "=== Test: Model Functionalities ===" << std::endl;
    
    srand(time(nullptr));

    // === “股票价格预测”===
   // 1. 从CSV文件中加载数据
    std::string csv_file = "LSTM_training_data.csv";  // 输入CSV文件路径
    std::vector<double> raw_data = CSVLoader::load_single_column(csv_file);

    // 打印加载的数据进行验证
    std::cout << "Loaded " << raw_data.size() << " data points from CSV." << std::endl;

    // 2. 准备训练模型的输入数据
    std::vector<double> in_input = raw_data;  // 使用CSV中的原始数据作为输入
    int history_length = 20;  // 历史序列长度
    int predict_steps = 3;    // 预测步数
    int predict_steps_model = 10;//模型预测步数
    int hidden_size = 32;     // 隐藏层大小
    int num_layers = 2;       // LSTM层数
    int batch_size = 2;       // 批次大小
    int epochs = 40;         // 训练轮数
    double learning_rate = 0.00007;  // 学习率
    std::string loss_function = "smooth_l1_loss";  // 损失函数
    std::string optimizer_type = "adam";  // 优化器
    std::string model_prefix = "my_model";  // 模型保存前缀
    double teacher_forcing_ratio = 0.8;   // 教师强制比例
    double teacher_forcing_decay = 0.98;  // 教师强制衰减率

    // 3. 训练模型
    train_model(
        in_input,           // 原始时间序列数据
        history_length,     // 历史序列长度
        predict_steps,      // 预测步数
        hidden_size,        // 隐藏层大小
        num_layers,         // LSTM层数
        batch_size,         // 批次大小
        epochs,             // 训练轮数
        learning_rate,      // 学习率
        loss_function,      // 损失函数
        optimizer_type,     // 优化器
        model_prefix,       // 模型保存前缀
        teacher_forcing_ratio, // 教师强制比例
        teacher_forcing_decay // 教师强制衰减率
    );

    std::cout << "Training completed, model saved as " << model_prefix << "_final.txt" << std::endl;

    // 4. 使用训练好的模型进行预测
    int window_size = history_length; // 使用相同的历史序列长度作为窗口大小
    int mc_samples = 30;  // 使用30次MC采样来量化预测的不确定性

    // 预测数据是从原始数据中选取的最后几步
    std::vector<double> recent_seq(raw_data.end() - history_length, raw_data.end());

    // 使用 `run_model_prediction` 进行预测
    std::vector<double> future_predictions = run_model_prediction(
        recent_seq,        // 最近的序列数据
        predict_steps_model,     // 预测步数
        hidden_size,       // 隐藏层大小
        num_layers,        // LSTM层数
        window_size,       // 滑动窗口大小
        model_prefix,      // 模型文件路径（假设模型已经训练并保存）
        mc_samples         // 采样次数
    );

    // 打印预测结果
    std::cout << "Predicted values: ";
    for (double val : future_predictions) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    //可视化
    plot_predictions(recent_seq, future_predictions);

    // 5. 保存预测结果到 CSV 文件
    std::string output_file = "predictions.csv";  // 输出CSV文件路径
    save_predictions_to_csv(future_predictions, output_file);
    std::cout << "Predictions saved to " << output_file << std::endl;

    return 0;
}
     
