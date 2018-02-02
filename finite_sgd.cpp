#include "base.h"
#include "utility.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

#define EPOCHS 100

class FiniteSGD : public Model {
    int size_, dim_;
    const double neg_penalty_, regularizer_;
    std::vector<std::vector<double>> embedding;
    std::vector<double> sqr_norm;

    void UpdateEmbedding(const Graph& positive, const Graph& negative, int x, double learn_rate);
  public:
    FiniteSGD(const Graph& graph, const Graph& negative, int dimension, double neg_penalty, double regularizer);
    double Evaluate(int x, int y);
    const std::vector<double>& GetEmbedding(int x) { return embedding[x]; }
};

void FiniteSGD::UpdateEmbedding(const Graph& positive, const Graph& negative, int x, double learn_rate) {
    double *vx = embedding[x].data();
    for (int i : positive.edge[x]) {
        const double* feature = embedding[i].data();
        double ip = InnerProduct(vx, feature, dim_);
        double coeff = -sigmoid(-ip);
        for (int j = 0; j < dim_; ++j)
            vx[j] -= learn_rate * coeff * feature[j];
    }
    for (int i : negative.edge[x]) {
        const double* feature = embedding[i].data();
        double ip = InnerProduct(vx, feature, dim_);
        double coeff = sigmoid(ip);
        for (int j = 0; j < dim_; ++j)
            vx[j] -= learn_rate * coeff * feature[j];
    }
    for (int j = 0; j < dim_; ++j)
        vx[j] -= 2 * regularizer_ * learn_rate * vx[j];
}

FiniteSGD::FiniteSGD(const Graph& graph, const Graph& negative, int dimension, double neg_penalty, double regularizer) :
    size_(graph.size),
    dim_(dimension),
    neg_penalty_(neg_penalty), 
    regularizer_(regularizer) {
    
    std::uniform_real_distribution<double> dist(-1, 1);
    embedding.resize(size_);
    for (int i = 0; i < size_; ++i)
        embedding[i].resize(dim_);
    for (int i = 0; i < size_; ++i)
        for (int j = 0; j < dim_; ++j)
            embedding[i][j] = dist(gen);

    sqr_norm.resize(size_);
    for (int i = 0; i < size_; ++i)
        sqr_norm[i] = InnerProduct(embedding[i].data(), embedding[i].data(), dim_);

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        double learn_rate = 1 / sqrt(i + 10);
        RandomPermutation(&order);
        for (int j : order)
            UpdateEmbedding(graph, negative, j, learn_rate);
    }
}

double FiniteSGD::Evaluate(int x, int y) {
    return InnerProduct(embedding[x].data(), embedding[y].data(), dim_);
}

Model* GetFiniteSGD(const Graph& graph, const Graph& negative, int dimension, double neg_penalty, double regularizer) {
    return new FiniteSGD(graph, negative, dimension, neg_penalty, regularizer);
}