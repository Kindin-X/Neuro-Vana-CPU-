#pragma once
#include "Dataset.h"

class TimeSeriesDataset : public Dataset {
public:
    TimeSeriesDataset(const std::vector<double>& series, size_t window_size, size_t predict_steps);

    size_t size() const override;
    std::pair<std::vector<std::vector<double>>, std::vector<double>> get_item(size_t index) const override;

private:
    std::vector<double> series_;
    size_t window_size_;
    size_t predict_steps_;
};


