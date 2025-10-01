#pragma once
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

enum class NormalizerType {
    MinMax,
    LogReturn
};

class INormalizerStrategy {
public:
    virtual ~INormalizerStrategy() = default;

    virtual std::vector<double> normalize(const std::vector<double>& data) = 0;
    virtual std::vector<double> denormalize(const std::vector<double>& normed) const = 0;

    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

class Normalizer {
private:
    std::unique_ptr<INormalizerStrategy> strategy;

public:
    static Normalizer create(NormalizerType type);

    std::vector<double> normalize(const std::vector<double>& data);
    std::vector<double> denormalize(const std::vector<double>& normed) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
};
