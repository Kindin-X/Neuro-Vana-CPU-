#include "Normalizer.h"
#include <fstream>
#include <cmath>
#include <stdexcept>

// === MinMaxNormalizer 实现 ===
class MinMaxNormalizer : public INormalizerStrategy {
private:
    double min_val = 0.0, max_val = 1.0;

public:
    std::vector<double> normalize(const std::vector<double>& data) override {
        min_val = *std::min_element(data.begin(), data.end());
        max_val = *std::max_element(data.begin(), data.end());
        std::vector<double> normed;
        for (double v : data)
            normed.push_back((v - min_val) / (max_val - min_val + 1e-8));
        return normed;
    }

    std::vector<double> denormalize(const std::vector<double>& normed) const override {
        std::vector<double> restored;
        for (double v : normed)
            restored.push_back(v * (max_val - min_val) + min_val);
        return restored;
    }

    void save(const std::string& path) const override {
        std::ofstream out(path);
        out << min_val << " " << max_val << "\n";
    }

    void load(const std::string& path) override {
        std::ifstream in(path);
        in >> min_val >> max_val;
    }
};

// === LogReturnNormalizer 实现 ===
class LogReturnNormalizer : public INormalizerStrategy {
private:
    double start_val = 1.0;

public:
    std::vector<double> normalize(const std::vector<double>& data) override {
        if (data.size() < 2) throw std::runtime_error("数据不足做 log return");
        start_val = data[0];
        std::vector<double> returns;
        for (size_t i = 1; i < data.size(); ++i)
            returns.push_back(std::log(data[i] / data[i - 1]));
        return returns;
    }

    std::vector<double> denormalize(const std::vector<double>& returns) const override {
        std::vector<double> restored;
        restored.push_back(start_val);
        for (double r : returns)
            restored.push_back(restored.back() * std::exp(r));
        return restored;
    }

    void save(const std::string& path) const override {
        std::ofstream out(path);
        out << start_val << "\n";
    }

    void load(const std::string& path) override {
        std::ifstream in(path);
        in >> start_val;
    }
};

// === Normalizer 工厂函数 ===
Normalizer Normalizer::create(NormalizerType type) {
    Normalizer normalizer;
    switch (type) {
    case NormalizerType::MinMax:
        normalizer.strategy = std::make_unique<MinMaxNormalizer>();
        break;
    case NormalizerType::LogReturn:
        normalizer.strategy = std::make_unique<LogReturnNormalizer>();
        break;
    default:
        throw std::invalid_argument("Unsupported normalizer type");
    }
    return normalizer;
}

// === Normalizer 委托接口 ===
std::vector<double> Normalizer::normalize(const std::vector<double>& data) {
    return strategy->normalize(data);
}
std::vector<double> Normalizer::denormalize(const std::vector<double>& normed) const {
    return strategy->denormalize(normed);
}
void Normalizer::save(const std::string& path) const {
    strategy->save(path);
}
void Normalizer::load(const std::string& path) {
    strategy->load(path);
}
