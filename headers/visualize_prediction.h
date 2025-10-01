#pragma once
#include <vector>
#include <string>



// 绘图函数声明：传入实际值和预测值向量
void plot_predictions(const std::vector<double>& actual,
    const std::vector<double>& predicted,
    const std::string& title = "Stock Forecast");
