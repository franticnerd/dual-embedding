#include "svm.h"
#include "utility.h"
#include <vector>   
#include <algorithm>
#include <iostream>

#define LINEAR_EPOCHS 2
#define KERNEL_EPOCHS 4
#define INFTY 1e10

// Dual Coordinate Descent
void LinearSVM(const std::vector<const double*>& feature, const std::vector<double>& feature_sqr_norm, const std::vector<int>& label,
    const std::vector<double>& penalty_coeff, const std::vector<double>& margin, std::vector<double>* coeff, 
    std::vector<double>* w, int dim, bool l2) {
    int feature_size = feature.size();
    std::fill(w->begin(), w->end(), 0);

    if (feature_size == 0) return;
    for (int i = 0; i < feature_size; ++i)
        if (fabs(coeff->at(i)) > 1e-4)
        for (int j = 0; j < dim; ++j)
            w->at(j) += feature[i][j] * coeff->at(i);

    std::vector<int> order(feature_size);
    for (int i = 0; i < feature_size; ++i)
        order[i] = i;
    for (int epoch = 0; epoch < LINEAR_EPOCHS; ++epoch) {
        RandomPermutation(&order);
        for (int i : order) {
            double G = label[i] * InnerProduct(w->data(), feature[i], dim) - margin[i];
            double U = (l2 ? INFTY : penalty_coeff[i]);
            double PG = G;
            if (coeff->at(i) == 0)
                PG = std::min(PG, (double)0);
            if (coeff->at(i) == U * label[i])
                PG = std::max(PG, (double)0);
            if (PG != 0) {
                double old_coeff = coeff->at(i);
                double Q = feature_sqr_norm[i] + (l2 ? 1 / penalty_coeff[i] : 0) / 2;
                double new_alpha = std::min(std::max(coeff->at(i) * label[i] - G / Q, (double)0), U);
                coeff->at(i) = new_alpha * label[i];
                for (int j = 0; j < dim; ++j)
                    w->at(j) += (coeff->at(i) - old_coeff) * feature[i][j];
            }
        }
    }
}

// Sequential Minimal Optimization
void KernelSVM(const std::vector<std::vector<double>>& kernel, const std::vector<int>& label,
    const std::vector<double>& penalty_coeff, const std::vector<double>& margin, std::vector<double>* coeff, bool l2) {
    std::vector<int> order(coeff->size());
    for (int i = 0; i < (int)coeff->size(); ++i)
        order[i] = i;

    std::vector<double> G(coeff->size(), 0);
    for (int i = 0; i < (int)coeff->size(); ++i)
        if (fabs(coeff->at(i)) > 1e-3)
            for (int j = 0; j < (int)coeff->size(); ++j)
                G[j] += label[j] * kernel[i][j] * coeff->at(i);
    for (int i = 0; i < (int)coeff->size(); ++i)
        G[i] -= margin[i];

    for (int epoch = 0; epoch < KERNEL_EPOCHS; ++epoch) {
        RandomPermutation(&order);
        for (int i : order) {
            double U = (l2 ? INFTY : penalty_coeff[i]);
            double PG = G[i];
            if (coeff->at(i) == 0)
                PG = std::min(PG, (double)0);
            if (coeff->at(i) == U * label[i])
                PG = std::max(PG, (double)0);
            if (PG != 0) {
                double old_coeff = coeff->at(i);
                double Q = kernel[i][i] + (l2 ? 1 / penalty_coeff[i] : 0) / 2;
                double new_alpha = std::min(std::max(coeff->at(i) * label[i] - G[i] / Q, (double)0), U);
                coeff->at(i) = new_alpha * label[i];
                for (int j = 0; j < (int)coeff->size(); ++j)
                    G[j] += label[j] * kernel[i][j] * (coeff->at(i) - old_coeff);
            }
        }
    }
}