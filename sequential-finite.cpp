#include "base.h"
#include "utility.h"
#include "svm.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

#define EPOCHS 10
#define POS_PENALTY 0.5
#define NEG_PENALTY 0.05

class SequentialFiniteEmbedding : public Model {
    int size_, dim_;
    std::vector<std::vector<double>> embedding;
    std::vector<std::vector<double>> coeff;
    std::vector<bool> estimated;

    void UpdateEmbedding(const Graph& positive, const Graph& negative, int x);
public:
    SequentialFiniteEmbedding(const Graph& graph, const Graph& negative, int dimension);
    double Evaluate(int x, int y);
};

void SequentialFiniteEmbedding::UpdateEmbedding(const Graph& positive, const Graph& negative, int x) {
    std::vector<std::vector<double>*> feature;
    std::vector<int> label;
    for (int i : positive.edge[x])
    if (estimated[i]) {
        feature.push_back(&embedding[i]);
        label.push_back(1);
    }
    for (int i : negative.edge[x]) 
    if (estimated[i]) {
        feature.push_back(&embedding[i]);
        label.push_back(-1);
    }
    coeff[x].resize(label.size());

    for (int i = 0; i < EPOCHS; ++i)
        LinearSVM(feature, label, POS_PENALTY, NEG_PENALTY, &coeff[x], false);
    for (int i = 0; i < dim_; ++i) {
        double val = 0;
        for (int j = 0; j < (int)feature.size(); ++j)
            val += coeff[x][j] * feature[j]->at(i);
        embedding[x][i] = val;
    }
    std::uniform_real_distribution<double> dist(-1 / sqrt(dim_), 1 / sqrt(dim_));
    for (int j = 0; j < dim_; ++j)
        embedding[x][j] += dist(gen);
    estimated[x] = true;
}

SequentialFiniteEmbedding::SequentialFiniteEmbedding(const Graph& graph, const Graph& negative, int dimension) :
    size_(graph.size),
    dim_(dimension) {
    embedding.resize(size_);
    for (int i = 0; i < size_; ++i)
        embedding[i].resize(dim_);

    coeff.resize(size_);
    estimated.resize(size_, false);

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    RandomPermutation(&order);
    for (int j : order)
        UpdateEmbedding(graph, negative, j);
}

double SequentialFiniteEmbedding::Evaluate(int x, int y) {
    return InnerProduct(embedding[x], embedding[y]);
}

Model* GetSequentialFiniteEmbedding(const Graph& graph, const Graph& negative, int dimension) {
    return new SequentialFiniteEmbedding(graph, negative, dimension);
}