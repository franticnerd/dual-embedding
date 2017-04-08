#pragma once

#include <vector>

// In the following two functions, coeff serves both as starting point as well as return value
void LinearSVM(const std::vector<std::vector<double>*>& feature, const std::vector<int>& label,
               double pos_penalty, double neg_penalty, std::vector<double>* coeff);
void KernelSVM(const std::vector<std::vector<double>>& kernel, const std::vector<int>& label, 
               double pos_penalty, double neg_penalty, std::vector<double>* coeff);