#include "train_model.h"
#include "LSTM.h"
#include "Optimizer.h"
#include "Normalizer.h"
#include "loss.h"
#include "DataLoader.h"
#include "TimeSeriesDataset.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <cstdlib>

// === 梯度裁剪实现（保持不变） ===
void clip_gradients(const std::vector<TensorPtr>& parameters, double max_norm) {
    double total_norm_sq = 0.0;
    for (const auto& p : parameters) {
        if (!p || !p->requires_grad || !p->get_grad()) continue;
        const auto& grad_data = p->get_grad()->get_data();
        for (double v : grad_data) {
            total_norm_sq += v * v;
        }
    }
    double total_norm = std::sqrt(total_norm_sq);
    if (total_norm <= max_norm || total_norm < 1e-6) return;

    double scale = max_norm / total_norm;
    for (const auto& p : parameters) {
        if (!p || !p->requires_grad || !p->get_grad()) continue;
        auto& grad_data = p->get_grad()->get_data();
        for (double& v : grad_data) {
            v *= scale;
        }
    }
}

// === 主训练函数 ===
void train_model(const std::vector<double>& raw_series,
    int sequence_len,
    int predict_steps,
    int hidden_size,
    int num_layers,
    int batch_size,
    int epochs,
    double learning_rate,
    const std::string& loss_type,
    const std::string& optimizer_type,
    const std::string& save_path_prefix,
    double teacher_forcing_ratio,
    double teacher_forcing_decay)
{
    for (double v : raw_series) {
        if (std::isnan(v) || std::isinf(v))
            throw std::runtime_error("输入数据中含有 NaN/Inf");
    }

    // 0. 数据归一化
    
    Normalizer normalizer = Normalizer::create(NormalizerType::MinMax);      // 改成 MinMax
    std::vector<double> series = normalizer.normalize(raw_series);           // series ∈ [0,1]
    normalizer.save(save_path_prefix + "_norm.txt");  // 保存 min,max

    // 1. 构建模型
    nn::LSTMModel model(1, hidden_size, num_layers);

    // 2. 优化器选择
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_type == "adam")       optimizer = std::make_unique<AdamOptimizer>(learning_rate);
    else if (optimizer_type == "adamW") optimizer = std::make_unique<AdamWOptimizer>(learning_rate);
    else if (optimizer_type == "sgd")   optimizer = std::make_unique<SGDOptimizer>(learning_rate);
    else throw std::invalid_argument("Unsupported optimizer: " + optimizer_type);

    // 3. 损失函数选择
    auto get_loss = [&](const TensorPtr& pred, const TensorPtr& tgt) -> TensorPtr {
        if (!pred || !tgt) throw std::runtime_error("Null tensor in loss computation.");
        if (loss_type == "smooth_l1") return smooth_l1_loss(pred, tgt);
        else if (loss_type == "mse")   return mean_squared_error(pred, tgt);
        else throw std::invalid_argument("Unsupported loss: " + loss_type);
        };

    // 4. 数据集与加载器
    TimeSeriesDataset dataset(series, sequence_len, predict_steps);
    size_t dataset_size = dataset.size();
    if (batch_size > dataset_size) {
        throw std::invalid_argument("batch_size (" + std::to_string(batch_size) +
            ") 大于可用样本数 (" + std::to_string(dataset_size) + ")");
    }
    DataLoader loader(&dataset, batch_size, true);

    std::cout << "Dataset size = " << dataset.size()
        << ", batches/epoch = " << (dataset.size() + batch_size - 1) / batch_size << std::endl;
    std::cout << "开始训练: hidden=" << hidden_size << ", layers=" << num_layers << std::endl;

    std::ofstream logf(save_path_prefix + "_train_log.txt", std::ios::app);
    double tf_ratio = teacher_forcing_ratio;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        loader.reset();
        double epoch_loss = 0.0;
        int batch_cnt = 0;
        tf_ratio *= teacher_forcing_decay;
        if (tf_ratio < 0.0) tf_ratio = 0.0;  // 防止负值

        std::cout << "[Epoch " << epoch << "] tf_ratio = " << tf_ratio << std::endl;

        try {
            while (loader.has_next()) {
                auto batch = loader.next();
                if (batch.empty()) continue;

                ++batch_cnt;
                int cur_bs = static_cast<int>(batch.size());
                model.reset_state();


                // === 关键改动：构造批处理输入，保持二维结构 batch × seq_len ===
                std::vector<std::vector<double>> batch_inputs(cur_bs, std::vector<double>(sequence_len));
                std::vector<std::vector<double>> batch_targets(cur_bs, std::vector<double>(predict_steps));

                for (int i = 0; i < cur_bs; ++i) {
                    for (int t = 0; t < sequence_len; ++t) {
                        // batch[i].first[t] 是 std::vector<double>，取单特征第0个元素
                        batch_inputs[i][t] = batch[i].first[t][0];
                    }
                    batch_targets[i] = batch[i].second;
                }

                // 前向传播历史序列
                for (int t = 0; t < sequence_len; ++t) {
                    std::vector<double> vals(cur_bs);
                    for (int i = 0; i < cur_bs; ++i) {
                        vals[i] = batch_inputs[i][t];
                    }
                    TensorPtr in_t = Tensor::create(vals, { 1, static_cast<size_t>(cur_bs) }, false);
                    model.forward({ in_t });
                }

                // 预测阶段
                std::vector<TensorPtr> preds, targets;
                std::vector<double> last_vals(cur_bs);
                for (int i = 0; i < cur_bs; ++i) last_vals[i] = batch_inputs[i].back();

                TensorPtr cur_in = Tensor::create(last_vals, { 1, static_cast<size_t>(cur_bs) }, true);

                for (int step = 0; step < predict_steps; ++step) {
                    TensorPtr out = model.forward({ cur_in });
                    preds.push_back(out);

                    std::vector<double> tv(cur_bs);
                    for (int i = 0; i < cur_bs; ++i) tv[i] = batch_targets[i][step];
                    TensorPtr tgt = Tensor::create(tv, { 1, static_cast<size_t>(cur_bs) }, false);
                    targets.push_back(tgt);

                    std::vector<double> next_vals(cur_bs);
                    for (int i = 0; i < cur_bs; ++i) {
                        double u = static_cast<double>(rand()) / RAND_MAX;
                        next_vals[i] = (u < tf_ratio ? tv[i] : out->get_data()[i]);
                    }
                    cur_in = Tensor::create(next_vals, { 1, static_cast<size_t>(cur_bs) }, false);
                }

                // 计算总损失
                TensorPtr total_loss = nullptr;
                for (size_t i = 0; i < preds.size(); ++i) {
                    // preds[i], targets[i] 均形状 [1, batch_size]
                    TensorPtr li = mean_squared_error(preds[i], targets[i]); // returns scalar
                    if (!total_loss) total_loss = li;
                    else total_loss = add(total_loss, li);
                }
                // 再除以 predict_steps：
                TensorPtr avg_loss = multiply(total_loss, Tensor::create({ 1.0 / predict_steps }, { 1,1 }, false));
               
                // 反向传播与优化
                model.zero_grad();
                avg_loss->backward();
                clip_gradients(model.parameters(), 1.0);
                optimizer->step(model.parameters());
                if (epoch == epochs - 1) {
                    // 导出计算图
                    export_dot(avg_loss, "train_graph_epoch_" + std::to_string(epoch) + ".dot");
                }


                epoch_loss += avg_loss->get_data().at(0);
               
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error during training: " << e.what() << std::endl;
            break;
        }

        double mean_loss = epoch_loss / std::max(1, batch_cnt);
        std::cout << "[Epoch " << epoch << "] AvgLoss = " << mean_loss
            << ", batches = " << batch_cnt << std::endl;
        
        if (logf.is_open()) {
            logf << "[Epoch " << epoch << "] AvgLoss = " << mean_loss
                << ", batches = " << batch_cnt << std::endl;
        }
       
    }

    model.save_model(save_path_prefix + "_final.txt");
    std::cout << "训练完成，模型保存为: " << save_path_prefix << "_final.txt" << std::endl;
}
