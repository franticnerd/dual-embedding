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

class DirectedFiniteEmbedding : public Model {
    int size_, dim_;
    const double neg_penalty_, regularizer_;
    
    std::vector<std::vector<double>> in_embedding, out_embedding, combined_embedding;
    std::vector<double> in_sqr_norm, out_sqr_norm;
    std::vector<std::vector<double>> in_coeff, out_coeff;

    void UpdateInEmbedding(const DGraph& positive, const DGraph& negative, int x);
    void UpdateOutEmbedding(const DGraph& positive, const DGraph& negative, int x);
public:
    DirectedFiniteEmbedding(const DGraph& graph, const DGraph& negative, int dimension, double neg_penalty, double regularizer);
    double Evaluate(int x, int y);
    const std::vector<double>& GetEmbedding(int x) { return combined_embedding[x]; }
};

void DirectedFiniteEmbedding::UpdateInEmbedding(const DGraph& positive, const DGraph& negative, int x) {
    std::vector<const double*> feature;
    std::vector<int> label;
    std::vector<double> penalty_coeff, margin, f_sqr_norm;
    for (int i : positive.in_edge[x]) {
        feature.push_back(out_embedding[i].data());
        label.push_back(1);
        penalty_coeff.push_back(1 / regularizer_);
        margin.push_back(1);
        f_sqr_norm.push_back(out_sqr_norm[i]);
    }
    for (int i : negative.in_edge[x]) {
        feature.push_back(out_embedding[i].data());
        label.push_back(-1);
        penalty_coeff.push_back(neg_penalty_ / regularizer_);
        margin.push_back(0);
        f_sqr_norm.push_back(out_sqr_norm[i]);
    }
    LinearSVM(feature, f_sqr_norm, label, penalty_coeff, margin, &in_coeff[x], &in_embedding[x], dim_, false);
    in_sqr_norm[x] = InnerProduct(in_embedding[x].data(), in_embedding[x].data(), dim_);
}

void DirectedFiniteEmbedding::UpdateOutEmbedding(const DGraph& positive, const DGraph& negative, int x) {
    std::vector<const double*> feature;
    std::vector<int> label;
    std::vector<double> penalty_coeff, margin, f_sqr_norm;
    for (int i : positive.out_edge[x]) {
        feature.push_back(in_embedding[i].data());
        label.push_back(1);
        penalty_coeff.push_back(1 / regularizer_);
        margin.push_back(1);
        f_sqr_norm.push_back(in_sqr_norm[i]);
    }
    for (int i : negative.out_edge[x]) {
        feature.push_back(in_embedding[i].data());
        label.push_back(-1);
        penalty_coeff.push_back(neg_penalty_ / regularizer_);
        margin.push_back(0);
        f_sqr_norm.push_back(in_sqr_norm[i]);
    }
    LinearSVM(feature, f_sqr_norm, label, penalty_coeff, margin, &out_coeff[x], &out_embedding[x], dim_, false);
    out_sqr_norm[x] = InnerProduct(out_embedding[x].data(), out_embedding[x].data(), dim_);
}

DirectedFiniteEmbedding::DirectedFiniteEmbedding(const DGraph& graph, const DGraph& negative, 
    int dimension, double neg_penalty, double regularizer) :
    size_(graph.size),
    dim_(dimension),
    neg_penalty_(neg_penalty),
    regularizer_(regularizer) {

    std::uniform_real_distribution<double> dist(-1, 1);
    in_embedding.resize(size_);
    out_embedding.resize(size_);
    for (int i = 0; i < size_; ++i) {
        in_embedding[i].resize(dim_);
        out_embedding[i].resize(dim_);
    }
    for (int i = 0; i < size_; ++i)
        for (int j = 0; j < dim_; ++j) {
            in_embedding[i][j] = dist(gen);
            out_embedding[i][j] = dist(gen);
        }

    in_sqr_norm.resize(size_);
    out_sqr_norm.resize(size_);
    for (int i = 0; i < size_; ++i) {
        in_sqr_norm[i] = InnerProduct(in_embedding[i].data(), in_embedding[i].data(), dim_);
        out_sqr_norm[i] = InnerProduct(out_embedding[i].data(), out_embedding[i].data(), dim_);
    }

    in_coeff.resize(size_);
    out_coeff.resize(size_);
    for (int i = 0; i < size_; ++i) {
        in_coeff[i].resize(graph.in_edge[i].size() + negative.in_edge[i].size());
        out_coeff[i].resize(graph.out_edge[i].size() + negative.out_edge[i].size());
    }

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        RandomPermutation(&order);
        for (int j : order) {
            UpdateInEmbedding(graph, negative, j);
            UpdateOutEmbedding(graph, negative, j);
        }
    }

    combined_embedding.resize(size_);
    for (int i = 0; i < size_; ++i) {
        for (double v : in_embedding[i])
            combined_embedding[i].push_back(v);
        for (double v : out_embedding[i])
            combined_embedding[i].push_back(v);
    }
}

double DirectedFiniteEmbedding::Evaluate(int x, int y) {
    return InnerProduct(out_embedding[x].data(), in_embedding[y].data(), dim_);
}

Model* GetDirectedFiniteEmbedding(const DGraph& graph, const DGraph& negative, int dimension, double neg_penalty, double regularizer) {
    return new DirectedFiniteEmbedding(graph, negative, dimension, neg_penalty, regularizer);
}