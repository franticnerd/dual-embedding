#pragma once

#include <vector>

inline double sqr(double x) {
    return x * x;
}

class Prop_Sampler {
    std::vector<double> ps;
  public:
    Prop_Sampler(const std::vector<double>& x);
    int GetSample();
};

void RandomPermutation(std::vector<int>* vec);
double InnerProduct(const std::vector<double>& x, const std::vector<double>& y);
double EvaluateF1(const std::vector<double>& positive, const std::vector<double>& negative);
double EvaluateMAP(const std::vector<double>& positive, const std::vector<double>& negative);
