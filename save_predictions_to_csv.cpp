#include"save_predictions_to_csv.h"

void save_predictions_to_csv(const std::vector<double>& predictions, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    file << "step,value\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        file << i << "," << predictions[i] << "\n";
    }

    file.close();
    std::cout << "预测结果已保存到 " << filename << std::endl;
}
