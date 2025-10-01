// TimeSeriesDataset.cpp
#include "TimeSeriesDataset.h"

TimeSeriesDataset::TimeSeriesDataset(const std::vector<double>& series, size_t window_size, size_t predict_steps)
    : series_(series), window_size_(window_size), predict_steps_(predict_steps) {}

size_t TimeSeriesDataset::size() const {
    return series_.size() - window_size_ - predict_steps_ + 1;
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> TimeSeriesDataset::get_item(size_t index) const {
    std::vector<std::vector<double>> input;
    for (size_t i = 0; i < window_size_; ++i)
        input.push_back({ series_[index + i] });

    std::vector<double> targets;
    for (size_t i = 0; i < predict_steps_; ++i)
        targets.push_back(series_[index + window_size_ + i]);

    return { input, targets };
}

