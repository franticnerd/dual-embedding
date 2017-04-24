#include "svm.h"
#include "utility.h"
#include <vector>   
#include <algorithm>
#include <iostream>

#define LINEAR_EPOCHS 2
#define KERNEL_EPOCHS 10
#define INFTY 1e10

// Dual Coordinate Descent
void LinearSVM(const std::vector<std::vector<double>*>& feature, const std::vector<int>& label,
    const std::vector<double>& penalty_coeff, const std::vector<double>& margin, std::vector<double>* coeff, bool l2) {
    if (feature.size() == 0) return;
    int dim = feature[0]->size();
    std::vector<double> w(dim, 0);
    for (int i = 0; i < (int)coeff->size(); ++i)
        if (fabs(coeff->at(i)) > 1e-4)
        for (int j = 0; j < dim; ++j)
            w[j] += feature[i]->at(j) * coeff->at(i);
    std::vector<double> Q(coeff->size(), 0);
    for (int i = 0; i < (int)coeff->size(); ++i) {
        for (int j = 0; j < dim; ++j)
            Q[i] += sqr(feature[i]->at(j));
        Q[i] += (l2 ? 1 / penalty_coeff[i] : 0) / 2;
    }

    std::vector<int> order(coeff->size());
    for (int i = 0; i < (int)coeff->size(); ++i)
        order[i] = i;
    for (int epoch = 0; epoch < LINEAR_EPOCHS; ++epoch) {
        RandomPermutation(&order);
        for (int i : order) {
            double G = label[i] * InnerProduct(w, *(feature[i])) - margin[i];
            double U = (l2 ? INFTY : penalty_coeff[i]);
            double PG = G;
            if (coeff->at(i) == 0)
                PG = std::min(PG, (double)0);
            if (coeff->at(i) == U * label[i])
                PG = std::max(PG, (double)0);
            if (PG != 0) {
                double old_coeff = coeff->at(i);
                double new_alpha = std::min(std::max(coeff->at(i) * label[i] - G / Q[i], (double)0), U);
                coeff->at(i) = new_alpha * label[i];
                for (int j = 0; j < dim; ++j)
                    w[j] += (coeff->at(i) - old_coeff) * feature[i]->at(j);
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