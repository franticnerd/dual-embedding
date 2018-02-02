#pragma once

#include <vector>

inline double sqr(double x) {
    return x * x;
}

inline double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

class Prop_Sampler {
    std::vector<double> ps;
  public:
    Prop_Sampler(const std::vector<double>& x);
    int GetSample();
};

void RandomPermutation(std::vector<int>* vec);
inline double InnerProduct(const double* x, const double* y, int dim) {
    double val = 0;
    for (int i = 0; i < dim; ++i)
        val += x[i] * y[i];
    return val;
}

double EvaluateF1(const std::vector<double>& positive, const std::vector<double>& negative);
double EvaluateAveragePrecision(const std::vector<double>& positive, const std::vector<double>& negative);
