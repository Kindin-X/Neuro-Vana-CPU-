#pragma once
#include <vector>
#include <utility>

class Dataset {
public:
    virtual size_t size() const = 0;
    virtual std::pair<std::vector<std::vector<double>>, std::vector<double>> get_item(size_t index) const = 0;
    virtual ~Dataset() = default;
};
