#include"Loss.h"
#include <omp.h>


TensorPtr scalar_tensor(double v) {
    return Tensor::create({ v }, { 1, 1 }, false);
}

TensorPtr tensor_abs(const TensorPtr& t) {
    std::vector<double> data = t->get_data();
    std::vector<double> abs_data(data.size());
    int N = static_cast<int>(data.size());
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        abs_data[i] = std::abs(data[i]);
    return Tensor::create(abs_data, t->shape, false);
}



TensorPtr mean_squared_error(const TensorPtr& pred, const TensorPtr& target) {
    auto diff = add(pred, target, true);
    auto squared = multiply(diff, diff);
    return mean(squared);
}



TensorPtr smooth_l1_loss(const TensorPtr& pred, const TensorPtr& target) {
    TensorPtr diff = add(pred, target,true);
    TensorPtr abs_diff = tensor_abs(diff);

    std::vector<double> cond_data;
    for (double v : abs_diff->get_data())
        cond_data.push_back(v < 1.0 ? 1.0 : 0.0);
    TensorPtr condition = Tensor::create(cond_data, abs_diff->shape, false);

    // 0.5 * diff^2
    TensorPtr diff_sq = multiply(diff, diff);
    TensorPtr half = scalar_tensor(0.5);
    TensorPtr half_diff_sq = multiply(half, diff_sq);

    // abs_diff - 0.5
    TensorPtr shifted_abs = add(abs_diff, half,true);

    // where(cond, 0.5 * diff^2, abs_diff - 0.5)
    TensorPtr term1 = multiply(condition, half_diff_sq);
    TensorPtr one = scalar_tensor(1.0);
    TensorPtr inv_cond = add(one, condition,true);
    TensorPtr term2 = multiply(inv_cond, shifted_abs);

    return add(term1, term2);
}
