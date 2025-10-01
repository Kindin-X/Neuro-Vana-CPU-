#pragma once
#include "tensor.h"
#include "autograd.h"
#include "activation_function.h"
#include "exceptions.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <sstream>
#include <random>
#include <stdexcept>  // 添加标准异常头文件

namespace nn {

    class LSTMCell {
    public:
        int input_size, hidden_size;

        // 输入门参数
        TensorPtr W_ix, W_ih, b_i;
        // 遗忘门参数
        TensorPtr W_fx, W_fh, b_f;
        // 候选门参数
        TensorPtr W_gx, W_gh, b_g;
        // 输出门参数
        TensorPtr W_ox, W_oh, b_o;

        TensorPtr h;  // 当前隐藏状态
        TensorPtr c;  // 当前细胞状态

        TensorPtr forward(const TensorPtr& x);
        void reset_state();
        LSTMCell(int input_size, int hidden_size);

        std::pair<TensorPtr, TensorPtr> forward(const TensorPtr& x,
            const TensorPtr& h_prev,
            const TensorPtr& c_prev);
    };

    class LSTMModel {
    public:
        std::vector<LSTMCell> layers;
        int num_layers;
        int input_size;  // 添加输入尺寸记录
        int hidden_size; // 添加隐藏尺寸记录

        bool training = true;
        void train() { training = true; }
        void eval() { training = false; }

        TensorPtr output_weight;
        TensorPtr output_bias;

        // 新增：单张量版本的重载，自动封装为长度为1的序列
        inline TensorPtr forward(const TensorPtr& x) {
            return forward(std::vector<TensorPtr>{ x });
        }

        LSTMModel(int input_size, int hidden_size, int num_layers = 1);
        TensorPtr forward(const std::vector<TensorPtr>& input_seq);
        std::vector<TensorPtr> parameters();
        void reset_state();
        void zero_grad() {
            for (auto& param : this->parameters()) {
                param->zero_grad();
            }
        }

        void save_model(const std::string& filename);
       void load_model(const std::string& filename);

    private:
        static std::vector<double> random_vec(int n, double limit) {
            std::vector<double> v(n);
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<> dis(-limit, limit);
            for (int i = 0; i < n; ++i) v[i] = dis(gen);
            return v;
        }
    };
}

