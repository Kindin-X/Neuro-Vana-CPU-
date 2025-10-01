#pragma once

#include "LSTM.h"
#include "Normalizer.h"
#include <vector>
#include <string>
#include <utility>
std::vector<double> run_model_prediction(
    const std::vector<double>& recent_seq,
    int predict_steps,
    int hidden_size,
    int num_layers,
    int window_size,
    const std::string& model_file,
    int mc_samples = 30);
