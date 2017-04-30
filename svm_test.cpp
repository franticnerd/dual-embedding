#include "unit_test.h"
#include "svm.h"
#include <vector>
#include <cassert>
#include <iostream>

std::vector<double> MakeFeature(double a, double b, double c) {
    std::vector<double> vec;
    vec.push_back(a);
    vec.push_back(b);
    vec.push_back(c);
    return vec;
}

std::vector<double> MakeVector(double a, double b, double c, double d) {
    std::vector<double> vec;
    vec.push_back(a);
    vec.push_back(b);
    vec.push_back(c);
    vec.push_back(d);
    return vec;
}

void LinearSVMTest() {
    std::vector<std::vector<double>> feature_vec;
    std::vector<int> label;
    std::vector<double> sqr_norm;
    feature_vec.push_back(MakeFeature(1, 1, 1)); label.push_back(1); sqr_norm.push_back(3);
    feature_vec.push_back(MakeFeature(2, 2, 3)); label.push_back(1); sqr_norm.push_back(17);
    feature_vec.push_back(MakeFeature(-1, -1, -1)); label.push_back(-1); sqr_norm.push_back(3);
    feature_vec.push_back(MakeFeature(-2, -2, -3)); label.push_back(-1); sqr_norm.push_back(17);

    std::vector<const double*> feature_ptr;
    for (int i = 0; i < (int)label.size(); ++i)
        feature_ptr.push_back(feature_vec[i].data());
    std::vector<double> coeff(4, 0), margin(4, 1), penalty_coeff(4, 1000), w(4, 0);
    LinearSVM(feature_ptr, sqr_norm, label, penalty_coeff, margin, &coeff, &w, 3, false);
    assert(fabs(w[0] - 0.3333) < 1e-3);
    assert(fabs(w[1] - 0.3333) < 1e-3);
    assert(fabs(w[2] - 0.3333) < 1e-3);
    assert(fabs(coeff[0] - coeff[2] - 0.3333) < 1e-3);
    assert(fabs(coeff[1]) < 1e-3);
    assert(fabs(coeff[3]) < 1e-3);
}

void KernelSVMTest() {
    std::vector<std::vector<double>> kernel;
    std::vector<int> label;
    kernel.push_back(MakeVector(3, 7, -3, -7)); label.push_back(1);
    kernel.push_back(MakeVector(7, 17, -7, -17)); label.push_back(1);
    kernel.push_back(MakeVector(-3, -7, 3, 7)); label.push_back(-1);
    kernel.push_back(MakeVector(-7, -17, 7, 17)); label.push_back(-1);

    std::vector<double> coeff(4, 0), margin(4, 1), penalty_coeff(4, 1000);
    KernelSVM(kernel, label, penalty_coeff, margin, &coeff, false);
    assert(fabs(coeff[0] - coeff[2] - 0.3333) < 1e-3);
    assert(fabs(coeff[1]) < 1e-3);
    assert(fabs(coeff[3]) < 1e-3);
}

void SVMTest() {
    LinearSVMTest();
    KernelSVMTest();
}