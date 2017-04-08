#include "utility.h"
#include <vector>
#include <random>
#include <algorithm>

namespace {
    std::mt19937 gen(910109);
}   // anonymous namespace

void RandomPermutation(std::vector<int>* vec) {
    std::uniform_int_distribution<int> dist(0, vec->size() - 1);
    for (int i = 0; i < (int)vec->size(); ++i) {
        int t = dist(gen);
        std::swap(vec->at(i), vec->at(t));
    }
}

double InnerProduct(const std::vector<double>& x, const std::vector<double>& y) {
    double val = 0;
    for (int i = 0; i < (int)x.size(); ++i)
        val += x[i] * y[i];
    return val;
}