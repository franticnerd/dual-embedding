#pragma once

#include <vector>

inline double sqr(double x) {
    return x * x;
}

void RandomPermutation(std::vector<int>* vec);
double InnerProduct(const std::vector<double>& x, const std::vector<double>& y);
