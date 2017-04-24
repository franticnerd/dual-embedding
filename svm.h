#pragma once

#include <vector>

// In the following two functions, coeff serves both as starting point as well as return value
void LinearSVM(const std::vector<std::vector<double>*>& feature, const std::vector<int>& label,
               const std::vector<double>& penalty_coeff, const std::vector<double>& margin, std::vector<double>* coeff, bool l2);
void KernelSVM(const std::vector<std::vector<double>>& kernel, const std::vector<int>& label, 
               const std::vector<double>& penalty_coeff, const std::vector<double>& margin, std::vector<double>* coeff, bool l2);