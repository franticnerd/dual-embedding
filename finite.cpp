#include "base.h"
#include "utility.h"
#include "svm.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

#define EPOCHS 10

class FiniteEmbedding : public Model {
    int size_, dim_;
    const double neg_penalty_, regularizer_, deg_norm_pow_;
    std::vector<std::vector<double>> embedding;
    std::vector<std::vector<double>> coeff;

    void UpdateEmbedding(const Graph& positive, const Graph& negative, int x);
  public:
    FiniteEmbedding(const Graph& graph, const Graph& negative, int dimension, double neg_penalty, double regularizer, double neg_norm_pow);
    double Evaluate(int x, int y);
};

void FiniteEmbedding::UpdateEmbedding(const Graph& positive, const Graph& negative, int x) {
    double deg_norm = pow(std::max((int)positive.edge[x].size(), 1), deg_norm_pow_);

    std::vector<std::vector<double>*> feature;
    std::vector<int> label;
    std::vector<double> penalty_coeff, margin;
    for (int i : positive.edge[x]) {
        feature.push_back(&embedding[i]);
        label.push_back(1);
        penalty_coeff.push_back(deg_norm / regularizer_);
        margin.push_back(1);
    }
    for (int i : negative.edge[x]) {
        feature.push_back(&embedding[i]);
        label.push_back(-1);
        penalty_coeff.push_back(neg_penalty_ * deg_norm / regularizer_);
        margin.push_back(0);
    }
    LinearSVM(feature, label, penalty_coeff, margin, &coeff[x], false);
    for (int i = 0; i < dim_; ++i) {
        double val = 0;
        for (int j = 0; j < (int)feature.size(); ++j)
            val += coeff[x][j] * feature[j]->at(i);
        embedding[x][i] = val;
    }
}

FiniteEmbedding::FiniteEmbedding(const Graph& graph, const Graph& negative, int dimension, double neg_penalty, double regularizer, double deg_norm_pow) :
    size_(graph.size),
    dim_(dimension),
    neg_penalty_(neg_penalty), 
    regularizer_(regularizer), 
    deg_norm_pow_(deg_norm_pow) {
    std::uniform_real_distribution<double> dist(-1, 1);
    embedding.resize(size_);
    for (int i = 0; i < size_; ++i)
        embedding[i].resize(dim_);
    for (int i = 0; i < size_; ++i)
        for (int j = 0; j < dim_; ++j)
            embedding[i][j] = dist(gen);

    coeff.resize(size_);
    for (int i = 0; i < size_; ++i)
        coeff[i].resize(graph.edge[i].size() + negative.edge[i].size());

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        RandomPermutation(&order);
        for (int j : order)
            UpdateEmbedding(graph, negative, j);
    }
}

double FiniteEmbedding::Evaluate(int x, int y) {
    return InnerProduct(embedding[x], embedding[y]);
}

Model* GetFiniteEmbedding(const Graph& graph, const Graph& negative, int dimension, double neg_penalty, double regularizer, double deg_norm_pow) {
    return new FiniteEmbedding(graph, negative, dimension, neg_penalty, regularizer, deg_norm_pow);
}