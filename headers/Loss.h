#pragma once
#include "tensor.h"
#include"autograd.h"
#include <cmath>

TensorPtr mean_squared_error(const TensorPtr& pred, const TensorPtr& target); 
TensorPtr smooth_l1_loss(const TensorPtr& pred, const TensorPtr& target); 
