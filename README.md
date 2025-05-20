# MyLSTM - A Minimal Autograd Framework with LSTM Support

> Lightweight C++ AI computation graph engine with autograd, tensor ops, and sequence modeling.

## 🔍 Overview

This is a self-developed deep learning backend framework written in C++, designed to support:
- Tensor structure with broadcasting
- Automatic differentiation (autograd)
- Matrix operations: add, multiply, matmul, slice, concat, normalize
- Computation graph export in `.dot` format
- Support for sequence models: LSTM
- Applied to tasks like stock curve prediction

## 📦 Features

- ⚙️ Pure C++ Implementation
- 🧠 LSTM with Backpropagation Through Time (BPTT)
- 🔁 Computation Graph Construction & Visualization
- 🚀 Performance-focused design, lightweight and modular

## 📂 Directory Structure
/my-lstm-project
│
├── src/ # Core framework source code
│ ├── tensor.cpp/h
| ├── autograd.cpp/h
| ├── activation_function.h
| ├── CSVLoader.cpp/h
| ├── DataLoader.cpp/h
| ├── Dataset.h
| ├── exceptions.h
│ ├── Loss.cpp/h
| ├── Optimizer.h
| ├── predict_model.cpp/h
| ├── save_predictions_to_csv.cpp/h
| ├──TimeSeriesDataset.cpp/h
| ├── train_model.cpp/h
│ └── lstm.cpp/h
│
├── examples/ # Usage examples (e.g. stock prediction)
├── include/ # Headers
├── tests/ # Unit tests
└── README.md


## 🛠️ Build Instructions

```bash
mkdir build && cd build
cmake ..
make
./run_example

