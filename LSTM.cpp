#include"LSTM.h"
#include "activation_function.h" 

namespace nn {

    LSTMCell::LSTMCell(int input_size, int hidden_size)
        : input_size(input_size), hidden_size(hidden_size)
    {
        double limit = std::sqrt(1.0 / (input_size + hidden_size));

        auto random_vec = [limit](int n) {
            std::vector<double> v(n);
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<> dis(-limit, limit);
            for (int i = 0; i < n; ++i) v[i] = dis(gen);
            return v;
            };

        // 使用静态转换确保尺寸安全
        size_t h_size = static_cast<size_t>(hidden_size);
        size_t i_size = static_cast<size_t>(input_size);

        // 初始化所有权重和偏置
        W_ix = Tensor::create(random_vec(h_size * i_size), { h_size, i_size }, true, "W_ix");
        W_ih = Tensor::create(random_vec(h_size * h_size), { h_size, h_size }, true, "W_ih");
        b_i = Tensor::create(std::vector<double>(h_size * 1, 1.0), { h_size, 1 }, true, "b_i");

        W_fx = Tensor::create(random_vec(h_size * i_size), { h_size, i_size }, true, "W_fx");
        W_fh = Tensor::create(random_vec(h_size * h_size), { h_size, h_size }, true, "W_fh");
        b_f = Tensor::create(std::vector<double>(h_size * 1, 1.0), { h_size, 1 }, true, "b_f");

        W_gx = Tensor::create(random_vec(h_size * i_size), { h_size, i_size }, true, "W_gx");
        W_gh = Tensor::create(random_vec(h_size * h_size), { h_size, h_size }, true, "W_gh");
        b_g = Tensor::create(std::vector<double>(h_size * 1, 1.0), { h_size, 1 }, true, "b_g");

        W_ox = Tensor::create(random_vec(h_size * i_size), { h_size, i_size }, true, "W_ox");
        W_oh = Tensor::create(random_vec(h_size * h_size), { h_size, h_size }, true, "W_oh");
        b_o = Tensor::create(std::vector<double>(h_size * 1, 1.0), { h_size, 1 }, true, "b_o");
    }

    TensorPtr LSTMCell::forward(const TensorPtr& x) {
        if (!h || !c || h->shape[1] != x->shape[1]) {
            size_t h_size = static_cast<size_t>(hidden_size);
            size_t batch_size = x->shape[1];
            h = Tensor::zeros({ h_size, batch_size }, true);
            c = Tensor::zeros({ h_size, batch_size }, true);
        }

        std::tie(h, c) = this->forward(x, h, c);
        return h;
    }

    void LSTMCell::reset_state() {
        h = nullptr;
        c = nullptr;
    }

    std::pair<TensorPtr, TensorPtr> LSTMCell::forward(const TensorPtr& x,
        const TensorPtr& h_prev,
        const TensorPtr& c_prev)
    {
        TensorPtr it_raw = add(add(matmul(W_ix, x), matmul(W_ih, h_prev)), b_i);
        TensorPtr it = layer_norm(it_raw);
        TensorPtr ft_raw = add(add(matmul(W_fx, x), matmul(W_fh, h_prev)), b_f);
        TensorPtr ft = layer_norm(it_raw);
        TensorPtr gt_raw = add(add(matmul(W_gx, x), matmul(W_gh, h_prev)), b_g);
        TensorPtr gt = layer_norm(it_raw);
        TensorPtr ot_raw = add(add(matmul(W_ox, x), matmul(W_oh, h_prev)), b_o);
        TensorPtr ot = layer_norm(it_raw);

        auto i = sigmoid(it);
        auto f = sigmoid(ft);
        auto g = tanh_activation(gt);
        auto o = sigmoid(ot);

        auto c = add(multiply(f, c_prev), multiply(i, g));
        auto h = multiply(o, tanh_activation(c));

        return { h, c };
    }

    // ===================== LSTMModel 实现 =====================
    LSTMModel::LSTMModel(int input_size, int hidden_size, int num_layers)
        : num_layers(num_layers), input_size(input_size), hidden_size(hidden_size)
    {
        layers.reserve(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            int layer_input_size = (i == 0) ? input_size : hidden_size;
            layers.emplace_back(layer_input_size, hidden_size);
        }

        // 输出层权重初始化
        output_weight = Tensor::create(
            random_vec(hidden_size, 0.08),
            { 1, static_cast<size_t>(hidden_size) },
            true,
            "W_out"
        );
        output_bias = Tensor::create({ 0.0 }, { 1 }, true, "b_out");
    }

    // 多层 LSTM forward 实现，保证所有运算均为 2D 矩阵
    TensorPtr LSTMModel::forward(const std::vector<TensorPtr>& input_seq) {
        if (input_seq.empty())
            throw std::invalid_argument("输入序列不能为空");

        size_t B = input_seq[0]->shape[1];  // batch size

        std::vector<TensorPtr> h(num_layers), c(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            h[l] = Tensor::zeros({ static_cast<size_t>(hidden_size), B });
            c[l] = Tensor::zeros({ static_cast<size_t>(hidden_size), B });
        }

        for (size_t t = 0; t < input_seq.size(); ++t) {
            TensorPtr x = input_seq[t];  // shape: (input_size, B)
            for (int l = 0; l < num_layers; ++l) {
                auto& lyr = layers[l];

                TensorPtr i_t =sigmoid(
                    add(matmul(lyr.W_ix, x), add(matmul(lyr.W_ih, h[l]), lyr.b_i))
                );
                TensorPtr f_t = sigmoid(
                    add(matmul(lyr.W_fx, x), add(matmul(lyr.W_fh, h[l]), lyr.b_f))
                );
                TensorPtr g_t = tanh_activation(
                    add(matmul(lyr.W_gx, x), add(matmul(lyr.W_gh, h[l]), lyr.b_g))
                );
                TensorPtr o_t = sigmoid(
                    add(matmul(lyr.W_ox, x), add(matmul(lyr.W_oh, h[l]), lyr.b_o))
                );

                c[l] = add(multiply(f_t, c[l]), multiply(i_t, g_t));
                h[l] = multiply(o_t, tanh_activation(c[l]));

                x = h[l];
            }
        }

        return h.back();  // shape: (hidden_size, B)
    }


    std::vector<TensorPtr> LSTMModel::parameters() {
        std::vector<TensorPtr> all_params;
        for (auto& layer : layers) {
            all_params.insert(all_params.end(), {
                layer.W_ix, layer.W_ih, layer.b_i,
                layer.W_fx, layer.W_fh, layer.b_f,
                layer.W_gx, layer.W_gh, layer.b_g,
                layer.W_ox, layer.W_oh, layer.b_o
                });
        }
        all_params.push_back(output_weight);
        all_params.push_back(output_bias);
        return all_params;
    }

    void LSTMModel::reset_state() {
        for (auto& layer : layers) {
            layer.reset_state();
        }
    }

    
   // ===================== 模型保存/加载工具函数 =====================
    void save_tensor(std::ofstream& ofs, const TensorPtr& t, const std::string& name) {
        ofs << name << "\n";

        // 写入张量形状
        ofs << t->shape.size();
        for (auto dim : t->shape) {
            ofs << " " << dim;
        }
        ofs << "\n";

        // 写入数据
        for (double v : t->get_data()) {
            ofs << std::setprecision(12) << v << " ";
        }
        ofs << "\n";
    }

    TensorPtr load_tensor(std::ifstream& ifs, const std::string& expected_name) {
        std::string name, line;

        std::getline(ifs, name);
        if (name != expected_name) {
            throw std::runtime_error("模型格式错误，预期: " + expected_name + "，实际: " + name);
        }

        // 读取形状行
        std::getline(ifs, line);
        std::istringstream shape_stream(line);
        size_t ndim;
        shape_stream >> ndim;
        std::vector<size_t> shape(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            shape_stream >> shape[i];
        }

        // 读取数据行
        std::getline(ifs, line);
        std::istringstream data_stream(line);
        std::vector<double> data;
        double val;
        while (data_stream >> val) {
            data.push_back(val);
        }

        // 校验数据长度是否匹配 shape
        size_t expected_size = 1;
        for (auto s : shape) expected_size *= s;
        if (data.size() != expected_size) {
            throw std::runtime_error("张量数据长度与 shape 不匹配: " + expected_name);
        }

        return Tensor::create(data, shape, true, expected_name);
    }


    // ===================== 模型保存/加载实现 =====================
    void LSTMModel::save_model(const std::string& filename) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) throw std::runtime_error("无法打开模型保存文件: " + filename);

        // 保存元数据
        ofs << "LSTMv1\n";
        ofs << input_size << " " << hidden_size << " " << num_layers << "\n";

        // 保存参数
        for (int i = 0; i < num_layers; ++i) {
            const auto& layer = layers[i];
            save_tensor(ofs, layer.W_ix, "W_ix_L" + std::to_string(i));
            save_tensor(ofs, layer.W_ih, "W_ih_L" + std::to_string(i));
            save_tensor(ofs, layer.b_i, "b_i_L" + std::to_string(i));

            save_tensor(ofs, layer.W_fx, "W_fx_L" + std::to_string(i));
            save_tensor(ofs, layer.W_fh, "W_fh_L" + std::to_string(i));
            save_tensor(ofs, layer.b_f, "b_f_L" + std::to_string(i));

            save_tensor(ofs, layer.W_gx, "W_gx_L" + std::to_string(i));
            save_tensor(ofs, layer.W_gh, "W_gh_L" + std::to_string(i));
            save_tensor(ofs, layer.b_g, "b_g_L" + std::to_string(i));

            save_tensor(ofs, layer.W_ox, "W_ox_L" + std::to_string(i));
            save_tensor(ofs, layer.W_oh, "W_oh_L" + std::to_string(i));
            save_tensor(ofs, layer.b_o, "b_o_L" + std::to_string(i));
        }

        save_tensor(ofs, output_weight, "W_out");
        save_tensor(ofs, output_bias, "b_out");
        ofs.close();
    }

    void LSTMModel::load_model(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs.is_open())
            throw std::runtime_error("无法打开模型文件: " + filename);

        // 检查版本
        std::string version;
        std::getline(ifs, version);
        if (version != "LSTMv1") {
            throw std::runtime_error("不支持的模型版本: " + version);
        }

        // 读取元数据
        int file_input_size, file_hidden_size, file_num_layers;
        ifs >> file_input_size >> file_hidden_size >> file_num_layers;
        // 跳过当前行剩余内容
        ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        // 验证模型兼容性
        if (file_input_size != input_size ||
            file_hidden_size != hidden_size ||
            file_num_layers != num_layers) {
            throw std::runtime_error("模型参数不匹配");
        }

        // 加载每层参数
        for (int i = 0; i < num_layers; ++i) {
            auto& layer = layers[i];
            layer.W_ix = load_tensor(ifs, "W_ix_L" + std::to_string(i));
            layer.W_ih = load_tensor(ifs, "W_ih_L" + std::to_string(i));
            layer.b_i = load_tensor(ifs, "b_i_L" + std::to_string(i));

            layer.W_fx = load_tensor(ifs, "W_fx_L" + std::to_string(i));
            layer.W_fh = load_tensor(ifs, "W_fh_L" + std::to_string(i));
            layer.b_f = load_tensor(ifs, "b_f_L" + std::to_string(i));

            layer.W_gx = load_tensor(ifs, "W_gx_L" + std::to_string(i));
            layer.W_gh = load_tensor(ifs, "W_gh_L" + std::to_string(i));
            layer.b_g = load_tensor(ifs, "b_g_L" + std::to_string(i));

            layer.W_ox = load_tensor(ifs, "W_ox_L" + std::to_string(i));
            layer.W_oh = load_tensor(ifs, "W_oh_L" + std::to_string(i));
            layer.b_o = load_tensor(ifs, "b_o_L" + std::to_string(i));
        }

        // 加载输出层参数
        output_weight = load_tensor(ifs, "W_out");
        output_bias = load_tensor(ifs, "b_out");

      
        ifs.close();
    }
}
