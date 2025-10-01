#pragma once
#include <stdexcept>
#include <string>

class EmptySequenceException : public std::runtime_error {
public:
    explicit EmptySequenceException(const std::string& msg = "输入为空.")
        : std::runtime_error(msg) {}
};

