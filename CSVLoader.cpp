#include "CSVLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<double> CSVLoader::load_single_column(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("无法打开CSV文件: " + filename);

    std::vector<double> values;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');  // 只取第1列
        values.push_back(std::stod(cell));
    }

    return values;
}
