#pragma once
#include <vector>
#include <string>

class CSVLoader {
public:
    static std::vector<double> load_single_column(const std::string& filename);
};
