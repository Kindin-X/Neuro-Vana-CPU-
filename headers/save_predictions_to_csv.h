#pragma once
#include <fstream>
#include"tensor.h"

void save_predictions_to_csv(const std::vector<double>& predictions, const std::string& filename);
