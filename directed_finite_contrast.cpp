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

struct ContrastEdgePair {
    int b, c, d, label;
    ContrastEdgePair(int b_, int c_, int d_, int label_) :
        b(b_), c(c_), d(d_), label(label_) {}
};

typedef std::vector<std::vector<ContrastEdgePair>> ContrastEdgeAdjacencyList;

class DirectedFiniteContrastEmbedding : public Model {
    int size_, dim_;
    const double regularizer_;
    std::vector<std::vector<double>> in_embedding, out_embedding, combined_embedding;
    std::vector<std::vector<double>> in_coeff, out_coeff;
    std::vector<double> in_sqr_norm, out_sqr_norm;

    void UpdateInEmbedding(const ContrastEdgeAdjacencyList& table, int x);
    void UpdateOutEmbedding(const ContrastEdgeAdjacencyList& table, int x);
public:
    DirectedFiniteContrastEmbedding(const DGraph& graph, const DGraph& negative, int sample_ratio, int dimension, double regularizer);
    double Evaluate(int x, int y);
    const std::vector<double>& GetEmbedding(int x) { return combined_embedding[x]; }
};

void DirectedFiniteContrastEmbedding::UpdateInEmbedding(const ContrastEdgeAdjacencyList& table, int x) {
    std::vector<const double*> feature;
    std::vector<int> label;
    std::vector<double> margin, penalty_coeff, f_sqr_norm;
    for (const ContrastEdgePair& pair : table[x]) {
        feature.push_back(out_embedding[pair.b].data());
        label.push_back(pair.label);
        margin.push_back(1 + pair.label * InnerProduct(out_embedding[pair.c].data(), in_embedding[pair.d].data(), dim_));
        penalty_coeff.push_back(1 / regularizer_);
        f_sqr_norm.push_back(out_sqr_norm[pair.b]);
    }
    LinearSVM(feature, f_sqr_norm, label, penalty_coeff, margin, &in_coeff[x], &in_embedding[x], dim_, false);
    in_sqr_norm[x] = InnerProduct(in_embedding[x].data(), in_embedding[x].data(), dim_);
}

void DirectedFiniteContrastEmbedding::UpdateOutEmbedding(const ContrastEdgeAdjacencyList& table, int x) {
    std::vector<const double*> feature;
    std::vector<int> label;
    std::vector<double> margin, penalty_coeff, f_sqr_norm;
    for (const ContrastEdgePair& pair : table[x]) {
        feature.push_back(in_embedding[pair.b].data());
        label.push_back(pair.label);
        margin.push_back(1 + pair.label * InnerProduct(out_embedding[pair.c].data(), in_embedding[pair.d].data(), dim_));
        penalty_coeff.push_back(1 / regularizer_);
        f_sqr_norm.push_back(in_sqr_norm[pair.b]);
    }
    LinearSVM(feature, f_sqr_norm, label, penalty_coeff, margin, &out_coeff[x], &out_embedding[x], dim_, false);
    out_sqr_norm[x] = InnerProduct(out_embedding[x].data(), out_embedding[x].data(), dim_);
}

DirectedFiniteContrastEmbedding::DirectedFiniteContrastEmbedding(const DGraph& graph, const DGraph& negative, int sample_ratio, int dimension, double regularizer) :
    size_(graph.size),
    dim_(dimension),
    regularizer_(regularizer) {

    std::uniform_real_distribution<double> dist_d(-1, 1);
    in_embedding.resize(size_);
    out_embedding.resize(size_);
    for (int i = 0; i < size_; ++i) {
        in_embedding[i].resize(dim_);
        out_embedding[i].resize(dim_);
    }
    for (int i = 0; i < size_; ++i)
        for (int j = 0; j < dim_; ++j) {
            in_embedding[i][j] = dist_d(gen);
            out_embedding[i][j] = dist_d(gen);
        }

    // Construct Contrast Pair Adjacency List
    ContrastEdgeAdjacencyList in_table(size_), out_table(size_);
    std::vector<std::pair<int, int>> edge_list;
    for (int x = 0; x < size_; ++x)
        for (int y : negative.out_edge[x])
            edge_list.push_back(std::make_pair(x, y));
    std::uniform_int_distribution<int> dist_i(0, edge_list.size() - 1);
    for (int a = 0; a < size_; ++a)
        for (int b : graph.out_edge[a]) {
            int cnt = 0;
            while (1) {
                int i = dist_i(gen);
                int c = edge_list[i].first, d = edge_list[i].second;
                if (a == c || a == d || b == c || b == d) continue;
                out_table[a].push_back(ContrastEdgePair(b, c, d, 1));
                in_table[b].push_back(ContrastEdgePair(a, c, d, 1));
                out_table[c].push_back(ContrastEdgePair(d, a, b, -1));
                in_table[d].push_back(ContrastEdgePair(c, a, b, -1));
                if (++cnt == sample_ratio)
                    break;
            }
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
        in_coeff[i].resize(in_table[i].size());
        out_coeff[i].resize(out_table[i].size());
    }

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        RandomPermutation(&order);
        for (int j : order) {
            UpdateInEmbedding(in_table, j);
            UpdateOutEmbedding(out_table, j);
        }
    }

    combined_embedding.resize(size_);    for (int i = 0; i < size_; ++i) {        for (double v : in_embedding[i])            combined_embedding[i].push_back(v);        for (double v : out_embedding[i])            combined_embedding[i].push_back(v);    }
}

double DirectedFiniteContrastEmbedding::Evaluate(int x, int y) {
    return InnerProduct(out_embedding[x].data(), in_embedding[y].data(), dim_);
}

Model* GetDirectedFiniteContrastEmbedding(const DGraph& graph, const DGraph& negative, int sample_ratio, int dimension, double regularizer) {
    return new DirectedFiniteContrastEmbedding(graph, negative, sample_ratio, dimension, regularizer);
}