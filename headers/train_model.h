#pragma once
#include"tensor.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>



void train_model(const std::vector<double>& raw_series,
    int sequence_len,
    int predict_steps,
    int hidden_size,
    int num_layers,
    int batch_size,
    int epochs,
    double learning_rate,
    const std::string& loss_type = "smooth_l1",
    const std::string& optimizer_type = "adam",
    const std::string& save_path_prefix = "lstm_epoch",
    double teacher_forcing_ratio = 0.8,  // 新增：教师强制比例
    double teacher_forcing_decay = 0.99);
