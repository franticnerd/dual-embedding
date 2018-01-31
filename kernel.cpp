#include "base.h"
#include "utility.h"
#include "svm.h"
#include <vector>
#include <algorithm>
#include <iostream>

#define EPOCHS 5

class KernelEmbedding : public Model {
    int size_;
    const double neg_penalty_, regularizer_;
    std::vector<std::vector<double>> kernel;
    std::vector<std::vector<double>> coeff;

    void UpdateEmbedding(const Graph& positive, const Graph& negative, int x);
public:
    KernelEmbedding(const Graph& graph, const Graph& negative, double neg_penalty, double regularizer);
    double Evaluate(int x, int y);
};

void KernelEmbedding::UpdateEmbedding(const Graph& positive, const Graph& negative, int x) {
    std::vector<std::vector<double>> local;
    std::vector<int> label, instance;
    std::vector<double> penalty_coeff, margin;
    for (int i : positive.edge[x]) {
        instance.push_back(i);
        label.push_back(1);
        penalty_coeff.push_back(1 / regularizer_);
        margin.push_back(1);
    }
    for (int i : negative.edge[x]) {
        instance.push_back(i);
        label.push_back(-1);
        penalty_coeff.push_back(neg_penalty_ / regularizer_);
        margin.push_back(0);
    }
    local.resize(instance.size());
    for (int i = 0; i < (int)instance.size(); ++i) {
        local[i].resize(instance.size());
        for (int j = 0; j < (int)instance.size(); ++j)
            local[i][j] = kernel[instance[i]][instance[j]];
    }

    KernelSVM(local, label, penalty_coeff, margin, &coeff[x], false);
    for (int i = 0; i < size_; ++i)
        if (i != x) {
            double val = 0;
            for (int j = 0; j < (int)instance.size(); ++j)
                val += kernel[instance[j]][i] * coeff[x][j];
            kernel[x][i] = kernel[i][x] = val;
        }
    double val = 0;
    for (int i = 0; i < (int)instance.size(); ++i)
        val += kernel[x][instance[i]] * coeff[x][i];
    kernel[x][x] = val;
}

KernelEmbedding::KernelEmbedding(const Graph& graph, const Graph& negative, double neg_penalty, double regularizer) :
    size_(graph.size),
    neg_penalty_(neg_penalty),
    regularizer_(regularizer) {
    kernel.resize(size_);
    for (int i = 0; i < size_; ++i)
        kernel[i].resize(size_, 0);
    for (int i = 0; i < size_; ++i) {
        for (int x : graph.edge[i])
            kernel[i][x] = 1;
        kernel[i][i] = graph.edge[i].size();
    }

    coeff.resize(size_);
    for (int i = 0; i < size_; ++i)
        coeff[i].resize(graph.edge[i].size() + negative.edge[i].size());

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        int cnt = 0;
        std::cout << "epoch " << i << "\n";
        RandomPermutation(&order);
        for (int j : order) {
            UpdateEmbedding(graph, negative, j);
            if (++cnt % 100 == 0)
                std::cout << "executed " << cnt << "\n";
        }
    }
}

double KernelEmbedding::Evaluate(int x, int y) {
    return kernel[x][y];
}

Model* GetKernelEmbedding(const Graph& graph, const Graph& negative, double neg_penalty, double regularizer) {
    return new KernelEmbedding(graph, negative, neg_penalty, regularizer);
}