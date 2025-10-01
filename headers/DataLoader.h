#pragma once
#include "Dataset.h"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

class DataLoader {
public:
    using Batch = std::vector<std::pair<std::vector<std::vector<double>>, std::vector<double>>>;

    DataLoader(Dataset* dataset, size_t batch_size, bool shuffle = true, bool drop_last = true);

    bool has_next() const;
    Batch next();
    void reset();

private:
    Dataset* dataset_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    size_t current_index_;
    std::vector<size_t> indices_;
    std::mt19937 rng_{ std::random_device{}() };
};
