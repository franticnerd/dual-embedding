#include "base.h"
#include "utility.h"
#include "svm.h"
#include <vector>
#include <map>
#include <algorithm>

#define EPOCHS 10

struct SparseFeature {
    int index;
    double value;
    SparseFeature(int index_, double value_) : index(index_), value(value_) {}
};

class SparseEmbedding : public Model {
    int size_;
    const double neg_penalty_, regularizer_, deg_norm_pow_;
    std::vector<std::vector<SparseFeature>> embedding;
    std::vector<std::vector<double>> coeff;
    std::vector<double> feature_buffer;

    void UpdateEmbedding(const Graph& positive, const Graph& negative, int x);
public:
    SparseEmbedding(const Graph& graph, const Graph& negative, double neg_penalty, double regularizer, double deg_norm_pow);
    double Evaluate(int x, int y);
};

void SparseEmbedding::UpdateEmbedding(const Graph& positive, const Graph& negative, int x) {
    double deg_norm = pow(std::max((int)positive.edge[x].size(), 1), deg_norm_pow_);

    std::vector<int> label, instance;
    std::vector<double> penalty_coeff, margin;
    for (int i : positive.edge[x]) {
        instance.push_back(i);
        label.push_back(1);
        penalty_coeff.push_back(deg_norm / regularizer_);
        margin.push_back(1);
    }
    for (int i : negative.edge[x]) {
        instance.push_back(i);
        label.push_back(-1);
        penalty_coeff.push_back(neg_penalty_ * deg_norm / regularizer_);
        margin.push_back(0);
    }

    std::vector<std::vector<double>> feature;
    std::vector<double> sqr_norm;
    for (int i : instance) {
        int index = 0;
        std::vector<double> vec(embedding[x].size());
        for (const auto& p : embedding[i])
            feature_buffer[p.index] = p.value;
        for (const auto& p : embedding[x])
            vec[index ++] = feature_buffer[p.index];
        for (const auto& p : embedding[i])
            feature_buffer[p.index] = 0;
        sqr_norm.push_back(InnerProduct(vec.data(), vec.data(), vec.size()));
        feature.push_back(std::move(vec));
    }
    std::vector<double*> feature_ptr;
    for (int i = 0; i < (int)instance.size(); ++i)
        feature_ptr.push_back(feature[i].data());

    std::vector<double> val(embedding[x].size(), 0);
    LinearSVM(feature_ptr, sqr_norm, label, penalty_coeff, margin, &coeff[x], &val, val.size(), false);

    for (int i = 0; i < (int)embedding[x].size(); ++i)
        embedding[x][i].value = val[i];
}

SparseEmbedding::SparseEmbedding(const Graph& graph, const Graph& negative, double neg_penalty, double regularizer, double deg_norm_pow) :
    size_(graph.size),
    neg_penalty_(neg_penalty),
    regularizer_(regularizer), 
    deg_norm_pow_(deg_norm_pow) {
    embedding.resize(size_);
    for (int i = 0; i < size_; ++i) {
        embedding[i].clear();
        embedding[i].push_back(SparseFeature(i, sqrt(graph.edge[i].size())));
        for (int j : graph.edge[i])
            embedding[i].push_back(SparseFeature(j, 1));
    }

    feature_buffer.resize(size_, 0);
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

double SparseEmbedding::Evaluate(int x, int y) {
    for (const auto& p : embedding[x])
        feature_buffer[p.index] = p.value;
    double val = 0;
    for (const auto& p : embedding[y])
        val += p.value * feature_buffer[p.index];
    for (const auto& p : embedding[x])
        feature_buffer[p.index] = 0;
    return val;
}

Model* GetSparseEmbedding(const Graph& graph, const Graph& negative, double neg_penalty, double regularizer, double deg_norm_pow) {
    return new SparseEmbedding(graph, negative, neg_penalty, regularizer, deg_norm_pow);
}