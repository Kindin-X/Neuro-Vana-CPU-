#define PYTHON_NO_DEBUG


#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

void plot_predictions(const std::vector<double>& actual,
    const std::vector<double>& predicted,
    const std::string& title) {
    std::vector<double> x_actual, x_pred;
    for (int i = 0; i < static_cast<int>(actual.size()); ++i)
        x_actual.push_back(static_cast<double>(i));
    for (int i = 0; i < static_cast<int>(predicted.size()); ++i)
        x_pred.push_back(static_cast<double>(i + actual.size()));

    // 替代 plot + label：使用 named_plot
    plt::figure_size(800, 400);
    plt::named_plot("Actual", x_actual, actual, "b-");
    plt::named_plot("Predicted", x_pred, predicted, "r--");

    plt::legend();
    plt::xlabel("Time Step");
    plt::ylabel("Value");
    plt::title(title);
    plt::grid(true);
    plt::show();
}

