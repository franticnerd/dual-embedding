#include "unit_test.h"
#include "svm.h"
#include <vector>
#include <cassert>

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
    feature_vec.push_back(MakeFeature(1, 1, 1)); label.push_back(1);
    feature_vec.push_back(MakeFeature(2, 2, 3)); label.push_back(1);
    feature_vec.push_back(MakeFeature(-1, -1, -1)); label.push_back(-1);
    feature_vec.push_back(MakeFeature(-2, -2, -3)); label.push_back(-1);

    std::vector<std::vector<double>*> feature_ptr_vec;
    for (int i = 0; i < (int)label.size(); ++i)
        feature_ptr_vec.push_back(&feature_vec[i]);     
    std::vector<double> coeff(4, 0);
    LinearSVM(feature_ptr_vec, label, 1000, 1000, &coeff);
    assert(fabs(coeff[0] - 0.5) < 1e-3);
    assert(fabs(coeff[0]) < 1e-3);
    assert(fabs(coeff[0] + 0.5) < 1e-3);
    assert(fabs(coeff[0]) < 1e-3);
}

void KernelSVMTest() {
    std::vector<std::vector<double>> kernel;
    std::vector<int> label;
    kernel.push_back(MakeVector(3, 7, -3, -7)); label.push_back(1);
    kernel.push_back(MakeVector(7, 17, -7, -17)); label.push_back(1);
    kernel.push_back(MakeVector(-3, -7, 3, 7)); label.push_back(-1);
    kernel.push_back(MakeVector(-7, -17, 7, 17)); label.push_back(-1);

    std::vector<double> coeff(4, 0);
    KernelSVM(kernel, label, 1000, 1000, &coeff);
    assert(fabs(coeff[0] - 0.5) < 1e-3);
    assert(fabs(coeff[0]) < 1e-3);
    assert(fabs(coeff[0] + 0.5) < 1e-3);
    assert(fabs(coeff[0]) < 1e-3);
}

void SVMTest() {
    LinearSVMTest();
    KernelSVMTest();
}