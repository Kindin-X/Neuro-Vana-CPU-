#include "LSTM.h"
#include "Normalizer.h"
#include <vector>
#include <iostream>


std::vector<double> vector_from_tensor(const TensorPtr& t) {
    int rows = t->shape[0];
    int cols = t->shape[1];
    std::vector<double> result;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Create an index for the 2D tensor (i, j)
            std::vector<size_t> indices = { static_cast<size_t>(i), static_cast<size_t>(j) };
            result.push_back((*t)(indices));  // Assuming operator() works with indices
        }
    }
    return result;
}



#include <fstream>
#include <sstream>

// 简化版 MC Dropout 预测接口（带反归一化）
std::vector<double> run_model_prediction(
    const std::vector<double>& recent_seq,
    int predict_steps,
    int hidden_size,
    int num_layers,
    int window_size,
    const std::string& model_file,
    int mc_samples = 30)
{
    if (recent_seq.size() < static_cast<size_t>(window_size))
        throw std::invalid_argument("输入序列长度不足");

    // === 加载归一化参数 ===
    std::ifstream norm_ifs(model_file + "_norm.txt");
    if (!norm_ifs.is_open()) {
        throw std::runtime_error("无法打开归一化参数文件: " + model_file + "_norm.txt");
    }
    double min_val, max_val;
    norm_ifs >> min_val >> max_val;
    norm_ifs.close();

    // === 加载模型 ===
    nn::LSTMModel model(1, hidden_size, num_layers);
    model.load_model(model_file + "_final.txt");
    model.train();  // 启用 Dropout

    std::mt19937 rng{ std::random_device{}() };
    std::uniform_int_distribution<int> pick(0, mc_samples - 1);

    std::vector<std::vector<double>> all_preds(mc_samples,
        std::vector<double>(predict_steps));

    for (int m = 0; m < mc_samples; ++m) {
        std::vector<double> win(recent_seq.end() - window_size, recent_seq.end());
        for (int t = 0; t < predict_steps; ++t) {
            std::vector<TensorPtr> seq;
            seq.reserve(window_size);
            for (double v : win) {
                // 归一化输入
                double norm_v = (v - min_val) / (max_val - min_val);
                seq.push_back(Tensor::create({ norm_v }, { 1,1 }, false));
            }

            TensorPtr out = model.forward(seq);
            auto vec = vector_from_tensor(out);
            double pred_norm = vec.back();

            // === 反归一化输出 ===
            double pred = pred_norm * (max_val - min_val) + min_val;
            // === 添加绝对单位的扰动 ===
            std::uniform_real_distribution<double> noise_dist(-10.0, 10.0);  //炒股有风险，投资需谨慎:)
            double noise = noise_dist(rng);
            pred += noise;

            all_preds[m][t] = pred;

           


            // 更新窗口
            win.erase(win.begin());
            win.push_back(pred);
        }
    }

    return all_preds[pick(rng)];
}
